"""Reproducible batch execution and result summaries for radar attacks."""

import csv
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml


NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
VALUE_OPTIONS = (
    'cfg_file',
    'batch_size',
    'workers',
    'seed',
    'ckpt',
    'epsilon',
    'attack_domain',
    'attack_feature',
    'attack_type',
    'pgd_steps',
    'step_size',
    'epsilon_xyz',
    'epsilon_rcs',
    'epsilon_doppler',
    'epsilon_time',
    'voxel_mode',
    'num_samples',
    'sample_strategy',
    'adv_dir',
    'adv_format',
    'score_threshold',
    'vod_devkit',
    'vod_label_dir',
    'vod_score_threshold',
    'launcher',
    'local_rank',
    'set',
)
FLAG_OPTIONS = (
    'random_start',
    'save_adv',
    'vod_eval',
    'no_vod_eval',
)
SUMMARY_COLUMNS = (
    'name',
    'status',
    'attack_domain',
    'attack_type',
    'attack_feature',
    'epsilon_default',
    'epsilon_xyz',
    'epsilon_rcs',
    'epsilon_doppler',
    'epsilon_time',
    'pgd_steps',
    'voxel_mode',
    'seed',
    'total_samples',
    'original_recall',
    'attacked_recall',
    'recall_drop',
    'attack_success_rate_sample',
    'max_abs_perturbation',
    'entire_clean_3d_map',
    'entire_adversarial_3d_map',
    'entire_3d_map_drop',
    'roi_clean_3d_map',
    'roi_adversarial_3d_map',
    'roi_3d_map_drop',
    'result_file',
)


@dataclass(frozen=True)
class Experiment:
    name: str
    parameters: Dict
    extra_tag: str
    output_dir: Path

    @property
    def result_path(self) -> Path:
        return self.output_dir / 'attack_results.json'


@dataclass(frozen=True)
class Campaign:
    name: str
    source_path: Path
    cfg_file: str
    output_dir: Path
    experiments: Sequence[Experiment]

    @property
    def state_path(self) -> Path:
        return self.output_dir / 'experiment_state.json'


def _validate_name(value: str, field: str) -> str:
    value = str(value)
    if not NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f'{field} must match {NAME_PATTERN.pattern}; got {value!r}'
        )
    return value


def load_campaign(path: Path) -> Dict:
    path = Path(path).expanduser().resolve()
    with path.open('r', encoding='utf-8') as campaign_file:
        payload = yaml.safe_load(campaign_file)
    if not isinstance(payload, dict):
        raise ValueError('campaign YAML must contain a mapping')
    if payload.get('version') != 1:
        raise ValueError('campaign YAML version must be 1')
    _validate_name(payload.get('name', ''), 'campaign name')

    common = payload.get('common')
    experiments = payload.get('experiments')
    if not isinstance(common, dict):
        raise ValueError('campaign common must be a mapping')
    if 'cfg_file' not in common or 'ckpt' not in common:
        raise ValueError('campaign common requires cfg_file and ckpt')
    if not isinstance(experiments, list) or not experiments:
        raise ValueError('campaign experiments must be a non-empty list')

    names = []
    for item in experiments:
        if not isinstance(item, dict):
            raise ValueError('every experiment must be a mapping')
        names.append(_validate_name(item.get('name', ''), 'experiment name'))
    if len(set(names)) != len(names):
        raise ValueError('experiment names must be unique')
    return payload


def _resolve_repo_path(value, repo_root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def _normalize_cfg_file(value, repo_root: Path, tools_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    candidates = [path] if path.is_absolute() else [tools_dir / path, repo_root / path]
    resolved = next((item.resolve() for item in candidates if item.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f'config file not found: {value}')
    try:
        return resolved.relative_to(tools_dir.resolve()).as_posix()
    except ValueError as error:
        raise ValueError('cfg_file must be located under the tools directory') from error


def _runner_output_root(cfg_file: str, repo_root: Path) -> Path:
    parts = Path(cfg_file).parts
    if len(parts) < 2:
        raise ValueError(f'cfg_file must include a config group: {cfg_file}')
    group = Path(*parts[1:-1]) if len(parts) > 2 else Path()
    return repo_root / 'output' / group / Path(cfg_file).stem


def _normalize_parameters(parameters: Dict, repo_root: Path, tools_dir: Path) -> Dict:
    allowed = set(VALUE_OPTIONS) | set(FLAG_OPTIONS)
    unknown = sorted(set(parameters) - allowed - {'name'})
    if unknown:
        raise ValueError(f'unsupported attack options: {unknown}')
    normalized = dict(parameters)
    normalized.pop('name', None)
    normalized['cfg_file'] = _normalize_cfg_file(
        normalized['cfg_file'], repo_root, tools_dir
    )
    for path_key in ('ckpt', 'vod_devkit', 'vod_label_dir', 'adv_dir'):
        if normalized.get(path_key) is not None:
            normalized[path_key] = _resolve_repo_path(
                normalized[path_key], repo_root
            )
    if not Path(normalized['ckpt']).is_file():
        raise FileNotFoundError(f'checkpoint not found: {normalized["ckpt"]}')
    if normalized.get('vod_eval') and normalized.get('no_vod_eval'):
        raise ValueError('vod_eval and no_vod_eval cannot both be true')
    if normalized.get('vod_eval') is False:
        normalized.pop('vod_eval')
        normalized['no_vod_eval'] = True
    if normalized.get('no_vod_eval') is False:
        normalized.pop('no_vod_eval')
    for option in FLAG_OPTIONS:
        if option not in normalized:
            continue
        if not isinstance(normalized[option], bool):
            raise ValueError(f'{option} must be a boolean')
        if normalized[option] is False:
            normalized.pop(option)
    return normalized


def build_campaign(
    source_path: Path,
    payload: Dict,
    repo_root: Path,
    tools_dir: Path,
) -> Campaign:
    name = _validate_name(payload['name'], 'campaign name')
    common = dict(payload['common'])
    experiments: List[Experiment] = []
    cfg_file = None

    for item in payload['experiments']:
        experiment_name = _validate_name(item['name'], 'experiment name')
        merged = {**common, **{key: value for key, value in item.items() if key != 'name'}}
        if 'extra_tag' in merged:
            raise ValueError('extra_tag is managed by the campaign runner')
        parameters = _normalize_parameters(merged, repo_root, tools_dir)
        if cfg_file is None:
            cfg_file = parameters['cfg_file']
        elif parameters['cfg_file'] != cfg_file:
            raise ValueError('all experiments in a campaign must use one cfg_file')

        extra_tag = f'{name}/{experiment_name}'
        output_dir = _runner_output_root(cfg_file, repo_root) / extra_tag
        experiments.append(
            Experiment(
                name=experiment_name,
                parameters=parameters,
                extra_tag=extra_tag,
                output_dir=output_dir,
            )
        )

    return Campaign(
        name=name,
        source_path=Path(source_path).resolve(),
        cfg_file=cfg_file,
        output_dir=_runner_output_root(cfg_file, repo_root) / name,
        experiments=experiments,
    )


def build_attack_command(
    experiment: Experiment,
    python_executable: str,
    attack_entry: Path,
) -> List[str]:
    command = [str(python_executable), str(Path(attack_entry).resolve())]
    parameters = experiment.parameters
    for option in VALUE_OPTIONS:
        if option == 'set':
            continue
        value = parameters.get(option)
        if value is None:
            continue
        command.append(f'--{option}')
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    for option in FLAG_OPTIONS:
        if parameters.get(option) is True:
            command.append(f'--{option}')
    command.extend(('--extra_tag', experiment.extra_tag))
    set_values = parameters.get('set')
    if set_values is not None:
        if not isinstance(set_values, (list, tuple)):
            raise ValueError('set must be a list of config key/value items')
        command.append('--set')
        command.extend(str(item) for item in set_values)
    return command


def _nested(mapping: Dict, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _load_result(path: Path) -> Optional[Dict]:
    try:
        with Path(path).open('r', encoding='utf-8') as result_file:
            payload = json.load(result_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get('metrics'), dict):
        return None
    return payload


def result_matches_experiment(
    experiment: Experiment,
    payload: Optional[Dict] = None,
) -> bool:
    payload = (
        payload
        if payload is not None
        else _load_result(experiment.result_path)
    )
    if payload is None:
        return False
    attack = payload.get('attack', {})
    for key, expected in experiment.parameters.items():
        if key == 'no_vod_eval':
            actual = not bool(attack.get('vod_eval', True))
        elif key == 'set':
            actual = attack.get('set_cfgs')
        else:
            actual = attack.get(key)
        if actual != expected:
            return False
    return attack.get('extra_tag') == experiment.extra_tag


def result_to_row(
    experiment: Experiment,
    state: Optional[Dict] = None,
) -> Dict:
    payload = _load_result(experiment.result_path)
    state = state or {}
    state_entry = state.get('experiments', {}).get(experiment.name, {})
    if payload is not None:
        status = (
            'completed'
            if result_matches_experiment(experiment, payload)
            else 'stale'
        )
    else:
        status = state_entry.get('status', 'pending')
    attack = payload.get('attack', {}) if payload else experiment.parameters
    metrics = payload.get('metrics', {}) if payload else {}
    vod = metrics.get('vod_official', {})

    row = {
        'name': experiment.name,
        'status': status,
        'attack_domain': attack.get('attack_domain'),
        'attack_type': attack.get('attack_type'),
        'attack_feature': attack.get('attack_feature'),
        'epsilon_default': attack.get('epsilon'),
        'epsilon_xyz': attack.get('epsilon_xyz'),
        'epsilon_rcs': attack.get('epsilon_rcs'),
        'epsilon_doppler': attack.get('epsilon_doppler'),
        'epsilon_time': attack.get('epsilon_time'),
        'pgd_steps': attack.get('pgd_steps'),
        'voxel_mode': attack.get('voxel_mode'),
        'seed': attack.get('seed'),
        'total_samples': metrics.get('total_samples'),
        'original_recall': metrics.get('original_recall'),
        'attacked_recall': metrics.get('attacked_recall'),
        'recall_drop': metrics.get('recall_drop'),
        'attack_success_rate_sample': metrics.get('attack_success_rate_sample'),
        'max_abs_perturbation': metrics.get('max_abs_perturbation'),
        'entire_clean_3d_map': _nested(vod, 'clean', 'entire_area', '3d', 'mAP'),
        'entire_adversarial_3d_map': _nested(
            vod, 'adversarial', 'entire_area', '3d', 'mAP'
        ),
        'entire_3d_map_drop': _nested(
            vod, 'absolute_drop', 'entire_area', '3d', 'mAP'
        ),
        'roi_clean_3d_map': _nested(vod, 'clean', 'roi', '3d', 'mAP'),
        'roi_adversarial_3d_map': _nested(
            vod, 'adversarial', 'roi', '3d', 'mAP'
        ),
        'roi_3d_map_drop': _nested(
            vod, 'absolute_drop', 'roi', '3d', 'mAP'
        ),
        'result_file': str(experiment.result_path) if payload else None,
    }
    return {column: row.get(column) for column in SUMMARY_COLUMNS}


def _format_cell(value) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def write_summaries(campaign: Campaign, state: Optional[Dict] = None) -> Sequence[Dict]:
    campaign.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result_to_row(experiment, state) for experiment in campaign.experiments]
    csv_path = campaign.output_dir / 'summary.csv'
    markdown_path = campaign.output_dir / 'summary.md'

    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with markdown_path.open('w', encoding='utf-8') as markdown_file:
        markdown_file.write('| ' + ' | '.join(SUMMARY_COLUMNS) + ' |\n')
        markdown_file.write('| ' + ' | '.join('---' for _ in SUMMARY_COLUMNS) + ' |\n')
        for row in rows:
            values = [_format_cell(row[column]).replace('|', '\\|') for column in SUMMARY_COLUMNS]
            markdown_file.write('| ' + ' | '.join(values) + ' |\n')
    return rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state(campaign: Campaign) -> Dict:
    try:
        with campaign.state_path.open('r', encoding='utf-8') as state_file:
            state = json.load(state_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = {}
    state.setdefault('version', 1)
    state.setdefault('campaign', campaign.name)
    state.setdefault('source', str(campaign.source_path))
    state.setdefault('experiments', {})
    return state


def _write_state(campaign: Campaign, state: Dict) -> None:
    campaign.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = campaign.state_path.with_suffix('.json.tmp')
    with temporary_path.open('w', encoding='utf-8') as state_file:
        json.dump(state, state_file, indent=2, ensure_ascii=False)
    temporary_path.replace(campaign.state_path)


def run_campaign(
    campaign: Campaign,
    repo_root: Path,
    attack_entry: Path,
    python_executable: str = sys.executable,
    selected_names: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    force: bool = False,
    keep_going: bool = False,
    summary_only: bool = False,
) -> int:
    selected = set(selected_names or ())
    known_names = {experiment.name for experiment in campaign.experiments}
    unknown_names = sorted(selected - known_names)
    if unknown_names:
        raise ValueError(f'unknown experiment names: {unknown_names}')

    experiments = [
        experiment
        for experiment in campaign.experiments
        if not selected or experiment.name in selected
    ]
    state = _load_state(campaign)
    if summary_only:
        write_summaries(campaign, state)
        return 0

    failed = False
    for experiment in experiments:
        command = build_attack_command(
            experiment, python_executable, attack_entry
        )
        existing_result = _load_result(experiment.result_path)
        if (
            existing_result is not None
            and result_matches_experiment(experiment, existing_result)
            and not force
        ):
            print(f'[skip] {experiment.name}: result already exists')
            continue
        if existing_result is not None and not force:
            print(f'[stale] {experiment.name}: parameters changed; rerunning')
        print(f'[run] {experiment.name}')
        print(f'      {shlex.join(command)}')
        if dry_run:
            continue

        state_entry = state['experiments'].setdefault(experiment.name, {})
        state_entry.update(
            {
                'status': 'running',
                'started_at': _utc_now(),
                'command': command,
            }
        )
        _write_state(campaign, state)
        completed = subprocess.run(command, cwd=Path(repo_root))
        state_entry.update(
            {
                'status': 'completed' if completed.returncode == 0 else 'failed',
                'finished_at': _utc_now(),
                'return_code': completed.returncode,
            }
        )
        _write_state(campaign, state)
        write_summaries(campaign, state)

        if completed.returncode != 0:
            failed = True
            if not keep_going:
                break

    if not dry_run:
        write_summaries(campaign, state)
    return 1 if failed else 0
