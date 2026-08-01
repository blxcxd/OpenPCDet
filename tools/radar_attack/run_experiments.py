"""CLI entry point for YAML-defined 4D-radar attack campaigns."""

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) in sys.path:
    sys.path.remove(str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from radar_attack.experiments.runner import (
    build_campaign,
    load_campaign,
    run_campaign,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run and summarize a YAML campaign of radar attacks'
    )
    parser.add_argument('campaign', help='path to campaign YAML')
    parser.add_argument(
        '--only',
        nargs='+',
        default=None,
        help='run only the named experiments',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--force',
        action='store_true',
        help='rerun experiments even when attack_results.json exists',
    )
    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='continue after an experiment fails',
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='regenerate summaries without running attacks',
    )
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch each attack',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.campaign).expanduser()
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    payload = load_campaign(source_path)
    campaign = build_campaign(
        source_path=source_path,
        payload=payload,
        repo_root=REPO_ROOT,
        tools_dir=TOOLS_DIR,
    )
    print(f'Campaign: {campaign.name}')
    print(f'Experiments: {len(campaign.experiments)}')
    print(f'Output: {campaign.output_dir}')
    return run_campaign(
        campaign=campaign,
        repo_root=REPO_ROOT,
        attack_entry=Path(__file__).with_name('run_attack.py'),
        python_executable=args.python,
        selected_names=args.only,
        dry_run=args.dry_run,
        force=args.force,
        keep_going=args.keep_going,
        summary_only=args.summary_only,
    )


if __name__ == '__main__':
    raise SystemExit(main())
