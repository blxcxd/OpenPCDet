"""Analyze model-input feature distributions for 4D-radar point clouds."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) in sys.path:
    sys.path.remove(str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

from radar_attack.adapters.openpcdet import get_feature_names
from radar_attack.analysis import (
    DEFAULT_QUANTILES,
    StreamingFeatureStatistics,
    extract_voxelized_features,
    format_statistics_table,
)


def parse_config():
    parser = argparse.ArgumentParser(
        description=(
            'Compute feature statistics over processed 4D-radar model inputs'
        )
    )
    parser.add_argument(
        '--cfg_file',
        required=True,
        help='dataset/model config relative to tools, for example cfgs/...',
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='maximum number of frames; default analyzes the full split',
    )
    parser.add_argument(
        '--max_quantile_points',
        type=int,
        default=1_000_000,
        help='maximum uniformly sampled points used for quantiles',
    )
    parser.add_argument(
        '--point_scope',
        choices=['voxelized', 'processed'],
        default='voxelized',
        help=(
            'voxelized: points actually retained by hard voxelization; '
            'processed: all points left by dataset preprocessing'
        ),
    )
    parser.add_argument(
        '--quantiles',
        type=float,
        nargs='+',
        default=list(DEFAULT_QUANTILES),
        help='quantile probabilities in [0, 1]',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='JSON path; relative paths are resolved from the repository root',
    )
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        default=None,
        nargs=argparse.REMAINDER,
        help='override config keys',
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    if args.workers < 0:
        parser.error('--workers must be non-negative')
    if args.seed < 0:
        parser.error('--seed must be non-negative')
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error('--num_samples must be positive')
    if args.max_quantile_points <= 0:
        parser.error('--max_quantile_points must be positive')
    if not args.quantiles or any(
        value < 0 or value > 1 for value in args.quantiles
    ):
        parser.error('--quantiles must contain values in [0, 1]')
    if len(set(args.quantiles)) != len(args.quantiles):
        parser.error('--quantiles must not contain duplicates')

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    return args, cfg


def _resolve_output_path(args) -> Path:
    if args.output is None:
        return (
            cfg.ROOT_DIR
            / 'output'
            / 'radar_attack'
            / cfg.TAG
            / 'feature_statistics.json'
        )
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = cfg.ROOT_DIR / output_path
    return output_path.resolve()


def main():
    os.chdir(TOOLS_DIR)
    args, dataset_cfg = parse_config()
    np.random.seed(args.seed)

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=dataset_cfg.DATA_CONFIG,
        class_names=dataset_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        seed=args.seed,
        training=False,
    )
    feature_names = get_feature_names(dataset_cfg.DATA_CONFIG)
    accumulator = StreamingFeatureStatistics(
        feature_names=feature_names,
        quantiles=args.quantiles,
        max_quantile_points=args.max_quantile_points,
        seed=args.seed,
    )

    frame_limit = (
        len(dataset)
        if args.num_samples is None
        else min(args.num_samples, len(dataset))
    )
    processed_frames = 0
    processed_raw_points = 0
    for batch_dict in dataloader:
        remaining = frame_limit - processed_frames
        if remaining <= 0:
            break

        batch_frames = int(batch_dict['batch_size'])
        accepted_frames = min(batch_frames, remaining)
        points = np.asarray(batch_dict['points'])
        if points.ndim != 2 or points.shape[1] != len(feature_names) + 1:
            raise ValueError(
                f'collated points have shape {points.shape}; expected '
                f'[N, {len(feature_names) + 1}]'
            )
        if accepted_frames < batch_frames:
            points = points[points[:, 0] < accepted_frames]
        processed_raw_points += int(points.shape[0])

        if args.point_scope == 'processed':
            feature_values = points[:, 1:]
        else:
            voxel_coords = np.asarray(batch_dict['voxel_coords'])
            voxel_mask = voxel_coords[:, 0] < accepted_frames
            feature_values = extract_voxelized_features(
                np.asarray(batch_dict['voxels'])[voxel_mask],
                np.asarray(batch_dict['voxel_num_points'])[voxel_mask],
            )
            if feature_values.shape[1] != len(feature_names):
                raise ValueError(
                    f'voxel features have shape {feature_values.shape}; '
                    f'expected [N, {len(feature_names)}]'
                )
        accumulator.update(feature_values)
        processed_frames += accepted_frames

        print(
            f'\rProcessed {processed_frames}/{frame_limit} frames, '
            f'{accumulator.total_points} {args.point_scope} points',
            end='',
            flush=True,
        )
    print()

    statistics = accumulator.compute()
    output_path = _resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'config': str(Path(args.cfg_file).resolve()),
        'dataset_root': str(Path(dataset.root_path).resolve()),
        'split': getattr(dataset, 'split', None),
        'processed_frames': processed_frames,
        'dataset_frames': len(dataset),
        'point_scope': args.point_scope,
        'processed_raw_points': processed_raw_points,
        'selected_points': statistics['total_points'],
        'selected_fraction_of_processed': (
            statistics['total_points'] / processed_raw_points
            if processed_raw_points
            else 0.0
        ),
        'point_stage': (
            'hard-voxel entries retained after range and capacity filtering'
            if args.point_scope == 'voxelized'
            else (
                'all points after feature encoding, FOV filtering, and '
                'DATA_PROCESSOR point filtering'
            )
        ),
        'statistics': statistics,
    }
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)

    print(format_statistics_table(statistics))
    print(
        f'Quantiles: {statistics["quantile_sampling"]} over '
        f'{statistics["quantile_sample_points"]} points'
    )
    for feature_name, values in statistics['features'].items():
        if (
            values['unique_values'] is not None
            and 0 < values['unique_count'] <= 16
        ):
            print(
                f'Discrete/low-cardinality feature {feature_name}: '
                f'{values["unique_values"]}'
            )
    print(f'Saved JSON: {output_path}')


if __name__ == '__main__':
    main()
