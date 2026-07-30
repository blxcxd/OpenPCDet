"""Adapter for the official View-of-Delft detection evaluator."""

import sys
import types
from pathlib import Path
from typing import Dict, Sequence


DEFAULT_CLASS_NAMES = ('Car', 'Pedestrian', 'Cyclist')
METRIC_SUFFIXES = {
    '3d': '3d_all',
    'bev': 'bev_all',
    'aos': 'aos_all',
}


def prepare_prediction_directory(output_dir: Path) -> Path:
    """Create a prediction directory and remove stale KITTI text files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for prediction_file in output_dir.glob('*.txt'):
        prediction_file.unlink()
    return output_dir


def resolve_vod_label_dir(dataset, override=None) -> Path:
    if override is not None:
        label_dir = Path(override).expanduser().resolve()
    else:
        label_dir = (
            Path(dataset.root_path).expanduser().resolve()
            / 'training'
            / 'label_2'
        )
    if not label_dir.is_dir():
        raise FileNotFoundError(f'VoD label directory not found: {label_dir}')
    return label_dir


def _load_evaluation_class(devkit_path: Path):
    """Load only the official evaluation package, skipping optional visuals."""
    devkit_path = Path(devkit_path).expanduser().resolve()
    vod_path = devkit_path / 'vod'
    evaluation_path = vod_path / 'evaluation' / 'evaluate.py'
    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f'VoD evaluator not found under {devkit_path}'
        )

    loaded_vod = sys.modules.get('vod')
    loaded_paths = list(getattr(loaded_vod, '__path__', []))
    if str(vod_path) not in loaded_paths:
        for module_name in list(sys.modules):
            if module_name == 'vod' or module_name.startswith('vod.'):
                del sys.modules[module_name]

        vod_package = types.ModuleType('vod')
        vod_package.__path__ = [str(vod_path)]
        vod_package.__package__ = 'vod'
        sys.modules['vod'] = vod_package

        from vod.common.file_handling import get_frame_list_from_folder

        vod_package.get_frame_list_from_folder = get_frame_list_from_folder

    from vod.evaluation import Evaluation

    return Evaluation


def summarize_vod_results(
    raw_results: Dict,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> Dict:
    summary = {}
    for area in ('entire_area', 'roi'):
        if area not in raw_results:
            raise KeyError(f'VoD evaluator result is missing "{area}"')
        summary[area] = {}
        for metric_name, suffix in METRIC_SUFFIXES.items():
            class_values = {
                class_name: float(
                    raw_results[area][f'{class_name}_{suffix}']
                )
                for class_name in class_names
            }
            aggregate_name = 'mAOS' if metric_name == 'aos' else 'mAP'
            summary[area][metric_name] = {
                **class_values,
                aggregate_name: (
                    sum(class_values.values()) / len(class_values)
                ),
            }
    return summary


def _metric_difference(clean: Dict, adversarial: Dict) -> Dict:
    difference = {}
    for area in ('entire_area', 'roi'):
        difference[area] = {}
        for metric_name in METRIC_SUFFIXES:
            difference[area][metric_name] = {
                key: float(clean[area][metric_name][key])
                - float(adversarial[area][metric_name][key])
                for key in clean[area][metric_name]
            }
    return difference


def _relative_metric_difference(clean: Dict, drop: Dict) -> Dict:
    relative = {}
    for area in ('entire_area', 'roi'):
        relative[area] = {}
        for metric_name in METRIC_SUFFIXES:
            relative[area][metric_name] = {}
            for key, value in drop[area][metric_name].items():
                clean_value = float(clean[area][metric_name][key])
                relative[area][metric_name][key] = (
                    float(value) / clean_value if clean_value > 0 else 0.0
                )
    return relative


def evaluate_vod_pair(
    clean_prediction_dir: Path,
    adversarial_prediction_dir: Path,
    label_dir: Path,
    devkit_path: Path,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    score_threshold: float = -1.0,
) -> Dict:
    clean_prediction_dir = Path(clean_prediction_dir).resolve()
    adversarial_prediction_dir = Path(adversarial_prediction_dir).resolve()
    clean_frames = {
        path.stem for path in clean_prediction_dir.glob('*.txt')
    }
    adversarial_frames = {
        path.stem for path in adversarial_prediction_dir.glob('*.txt')
    }
    if not clean_frames:
        raise ValueError('no clean prediction files were generated')
    if clean_frames != adversarial_frames:
        raise ValueError(
            'clean and adversarial prediction frame sets do not match'
        )

    Evaluation = _load_evaluation_class(devkit_path)
    evaluator = Evaluation(test_annotation_file=str(Path(label_dir).resolve()))
    class_ids = list(range(len(class_names)))
    clean_raw = evaluator.evaluate(
        result_path=str(clean_prediction_dir),
        current_class=class_ids,
        score_thresh=score_threshold,
    )
    adversarial_raw = evaluator.evaluate(
        result_path=str(adversarial_prediction_dir),
        current_class=class_ids,
        score_thresh=score_threshold,
    )
    clean = summarize_vod_results(clean_raw, class_names)
    adversarial = summarize_vod_results(adversarial_raw, class_names)
    drop = _metric_difference(clean, adversarial)

    return {
        'frames': len(clean_frames),
        'score_threshold': float(score_threshold),
        'clean': clean,
        'adversarial': adversarial,
        'absolute_drop': drop,
        'relative_drop': _relative_metric_difference(clean, drop),
    }
