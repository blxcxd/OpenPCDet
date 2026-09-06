"""Prepare and audit temporal metadata for VoD five-sweep Radar attacks."""

import argparse
import json
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) in sys.path:
    sys.path.remove(str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from radar_attack.temporal import (  # noqa: E402
    DEFAULT_TARGET_CLASSES,
    TemporalSweepResolver,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Recover source-sweep transforms, validate zero-delta '
            'reconstruction, and assign historical target tracks.'
        )
    )
    parser.add_argument(
        '--dataset_root', type=Path,
        default=Path('data/view_of_delft'),
    )
    parser.add_argument('--split', default='val')
    parser.add_argument('--frame_ids', nargs='+')
    parser.add_argument('--max_frames', type=int)
    parser.add_argument(
        '--cache_dir', type=Path,
        default=Path('output/radar_attack/temporal_sweep_cache'),
    )
    parser.add_argument(
        '--report_file', type=Path,
        default=Path(
            'output/radar_attack/temporal_sweep_cache/preparation_report.json'
        ),
    )
    parser.add_argument(
        '--target_classes', nargs='+',
        default=list(DEFAULT_TARGET_CLASSES),
    )
    parser.add_argument('--box_margin', type=float, default=0.0)
    parser.add_argument('--max_residual_m', type=float, default=2e-5)
    return parser.parse_args()


def selected_frames(args):
    if args.frame_ids:
        frames = [str(value).zfill(5) for value in args.frame_ids]
    else:
        split_file = (
            args.dataset_root / 'lidar/ImageSets' / f'{args.split}.txt'
        )
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        frames = split_file.read_text().split()
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError('--max_frames must be positive')
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError('no frames selected')
    return frames


def main():
    args = parse_args()
    frames = selected_frames(args)
    resolver = TemporalSweepResolver(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        max_residual_m=args.max_residual_m,
    )
    per_frame = []
    maximum_transform_error = 0.0
    maximum_roundtrip_error = 0.0
    cache_hits = 0
    sweep_count = 0
    target_count = 0
    assigned_points = 0
    historical_assigned_points = 0
    for frame_id in frames:
        context = resolver.resolve(frame_id)
        zero = context.zero_delta_diagnostics()
        assignments = resolver.assign_tracks(
            context,
            target_classes=args.target_classes,
            box_margin=args.box_margin,
        )
        transform_error = max(
            record.residual_max_m for record in context.sweeps
        )
        maximum_transform_error = max(
            maximum_transform_error, transform_error
        )
        maximum_roundtrip_error = max(
            maximum_roundtrip_error,
            zero['measurement_roundtrip_max_m'],
        )
        cache_hits += int(context.cache_hit)
        sweep_count += len(context.sweeps)
        diagnostics = assignments.diagnostics
        target_count += int(diagnostics['current_target_count'])
        assigned_points += int(diagnostics['assigned_target_point_count'])
        historical_assigned_points += int(
            diagnostics['historical_assigned_target_point_count']
        )
        if not zero['zero_mask_exact'] or zero['non_xyz_change_count']:
            raise RuntimeError(
                f'{frame_id}: zero perturbation did not preserve clean input'
            )
        per_frame.append({
            'frame_id': context.frame_id,
            'cache_hit': context.cache_hit,
            'transform_max_residual_m': transform_error,
            **zero,
            **diagnostics,
        })

    report = {
        'version': 1,
        'dataset_root': str(args.dataset_root.resolve()),
        'split': args.split,
        'frame_count': len(frames),
        'sweep_count': sweep_count,
        'cache_hits': cache_hits,
        'cache_misses': len(frames) - cache_hits,
        'target_classes': list(args.target_classes),
        'current_target_count': target_count,
        'assigned_target_point_count': assigned_points,
        'historical_assigned_target_point_count': historical_assigned_points,
        'maximum_transform_residual_m': maximum_transform_error,
        'maximum_measurement_roundtrip_error_m': maximum_roundtrip_error,
        'zero_delta_exact_for_all_frames': True,
        'non_xyz_change_count': 0,
        'frames': per_frame,
    }
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    )
    print(
        f"Temporal preparation passed: frames={len(frames)}, "
        f"sweeps={sweep_count}, targets={target_count}, "
        f"assigned_points={assigned_points}, "
        f"historical_assigned_points={historical_assigned_points}"
    )
    print(
        f"max_transform_residual={maximum_transform_error:.9g} m, "
        f"max_measurement_roundtrip_error={maximum_roundtrip_error:.9g} m"
    )
    print(
        f"cache_hits={cache_hits}, cache_misses={len(frames)-cache_hits}, "
        f"report={args.report_file}"
    )


if __name__ == '__main__':
    main()
