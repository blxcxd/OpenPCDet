"""OpenPCDet experiment runner for 4D-radar adversarial attacks."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import datetime
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Sampler

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from radar_attack.adapters.openpcdet import (
    PointCloudVoxelizer,
    get_feature_names,
    get_voxel_settings,
)
from radar_attack.attacks.gradient import point_cloud_attack
from radar_attack.attacks.iadv import build_iadv_attack_mask, iadv_rcs_attack
from radar_attack.attacks.iou_s_original import original_iou_s_point_attack
from radar_attack.attacks.measurement import (
    TEMPORAL_PARAMETER_MODES,
    build_measurement_attack_mask,
    build_temporal_measurement_attack_mask,
    radar_measurement_geometry_attack,
    radar_temporal_measurement_geometry_attack,
)
from radar_attack.attacks.objective import ObjectEvidenceObjective
from radar_attack.attacks.voxel import voxel_attack
from radar_attack.evaluation import (
    AdversarialPointCloudWriter,
    DetectionAttackMetrics,
    build_target_screening_context,
    compare_target_object_endpoints,
    evaluate_vod_pair,
    merge_target_diagnostics,
    prepare_prediction_directory,
    resolve_vod_label_dir,
    summarize_current_sweep,
    set_temporal_screening_assignments,
    target_attack_diagnostics,
    target_correlations,
    write_correlation_output,
    write_current_sweep_outputs,
)
from radar_attack.temporal import (
    TemporalSweepResolver,
    build_temporal_batch_data,
)


DEFAULT_OBJECT_IOU_THRESHOLDS = {
    'Car': 0.5,
    'Pedestrian': 0.25,
    'Cyclist': 0.25,
}


def parse_class_thresholds(values):
    thresholds = {}
    for item in values or []:
        if '=' not in item:
            raise ValueError(
                f'invalid class threshold {item!r}; expected CLASS=VALUE'
            )
        class_name, raw_threshold = item.split('=', 1)
        class_name = class_name.strip()
        if not class_name or class_name in thresholds:
            raise ValueError(f'invalid or duplicate class name in {item!r}')
        try:
            threshold = float(raw_threshold)
        except ValueError as error:
            raise ValueError(f'invalid IoU threshold in {item!r}') from error
        if not 0 <= threshold <= 1:
            raise ValueError(f'IoU threshold must be in [0, 1]: {item!r}')
        thresholds[class_name] = threshold
    return thresholds


def resolve_object_targets(args, class_names):
    available = list(class_names)
    selected = list(args.target_classes)
    if len(selected) != len(set(selected)):
        raise ValueError(f'duplicate target classes: {selected}')
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            f'unknown target classes {unknown}; available classes: {available}'
        )

    configured = dict(DEFAULT_OBJECT_IOU_THRESHOLDS)
    configured.update(parse_class_thresholds(args.object_iou_thresholds))
    missing = [name for name in selected if name not in configured]
    if missing:
        raise ValueError(
            f'no default IoU threshold for {missing}; provide '
            '--object_iou_thresholds CLASS=VALUE'
        )
    ids = [available.index(name) + 1 for name in selected]
    return selected, ids, {
        class_id: configured[class_name]
        for class_name, class_id in zip(selected, ids)
    }


class OrderedIndexSampler(Sampler):
    """Yield a reproducible, explicitly ordered validation subset."""

    def __init__(self, indices):
        self.indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def select_sample_indices(dataset_size, num_samples, strategy, seed):
    if dataset_size <= 0:
        raise ValueError('dataset_size must be positive')
    if num_samples is None or num_samples >= dataset_size:
        return np.arange(dataset_size, dtype=np.int64)
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')
    if strategy == 'first':
        return np.arange(num_samples, dtype=np.int64)
    if strategy == 'uniform':
        return np.floor(
            (np.arange(num_samples, dtype=np.float64) + 0.5)
            * dataset_size
            / num_samples
        ).astype(np.int64)
    if strategy == 'random':
        indices = np.random.default_rng(seed).choice(
            dataset_size, size=num_samples, replace=False
        )
        return np.sort(indices.astype(np.int64, copy=False))
    raise ValueError(f'unknown sample strategy: {strategy}')


def build_subset_dataloader(dataset, args, indices):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        sampler=OrderedIndexSampler(indices),
        collate_fn=dataset.collate_batch,
        drop_last=False,
        timeout=0,
        worker_init_fn=partial(common_utils.worker_init_fn, seed=args.seed),
    )


def parse_config():
    parser = argparse.ArgumentParser(description='FGSM Attack on PointPillars for 4D Radar')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for attack')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=1024,
                        help='random seed for reproducible attacks')
    parser.add_argument('--extra_tag', type=str, default='fgsm_attack_radar', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--epsilon', type=float, default=0.05, help='FGSM epsilon (perturbation size)')
    parser.add_argument('--attack_domain', type=str, default='voxel',
                        choices=['voxel', 'point'],
                        help='attack voxel tensor or raw 4D-radar points')
    parser.add_argument('--attack_feature', type=str, default='all',
                        choices=['all', 'xyz', 'doppler', 'intensity', 'rcs', 'time'],
                        help='which features to perturb')
    parser.add_argument(
        '--attack_space', choices=['feature', 'radar_measurement'],
        default='feature',
        help=(
            'feature uses the existing independent input channels; '
            'radar_measurement optimizes current-sweep range/angles and '
            'reconstructs XYZ'
        ),
    )
    parser.add_argument('--attack_type', type=str, default='fgsm',
                        choices=['fgsm', 'pgd', 'iadv', 'iou_s_original'],
                        help=(
                            'attack type: FGSM, PGD, I-ADV-RCS, or the '
                            'paper-faithful IoU-S point perturbation'
                        ))
    parser.add_argument('--pgd_steps', type=int, default=5, help='PGD steps')
    parser.add_argument('--step_size', type=float, default=None,
                        help='point PGD step size (default: 2 * epsilon / steps)')
    parser.add_argument('--epsilon_xyz', type=float, default=None,
                        help='point attack xyz budget, overriding epsilon')
    parser.add_argument('--epsilon_rcs', type=float, default=None,
                        help='point attack RCS budget, overriding epsilon')
    parser.add_argument('--epsilon_doppler', type=float, default=None,
                        help='point attack Doppler budget, overriding epsilon')
    parser.add_argument('--epsilon_time', type=float, default=None,
                        help='point attack timestamp budget, overriding epsilon')
    parser.add_argument('--epsilon_range', type=float, default=None,
                        help='measurement-space range budget in metres')
    parser.add_argument('--epsilon_azimuth_deg', type=float, default=None,
                        help='measurement-space azimuth budget in degrees')
    parser.add_argument('--epsilon_elevation_deg', type=float, default=None,
                        help='measurement-space elevation budget in degrees')
    parser.add_argument('--step_size_range', type=float, default=None,
                        help='measurement PGD range step in metres')
    parser.add_argument('--step_size_azimuth_deg', type=float, default=None,
                        help='measurement PGD azimuth step in degrees')
    parser.add_argument('--step_size_elevation_deg', type=float, default=None,
                        help='measurement PGD elevation step in degrees')
    parser.add_argument('--random_start', action='store_true',
                        help='use a random PGD start for point attack')
    parser.add_argument(
        '--current_sweep_only', action='store_true',
        help=(
            'for feature-space point attacks, restrict the clean target mask '
            'to time=0 points retained by clean hard voxelization'
        ),
    )
    parser.add_argument(
        '--temporal_mode',
        choices=['none', *TEMPORAL_PARAMETER_MODES],
        default='none',
        help=(
            'all non-none modes attack every available source sweep; '
            'point_independent gives each target point its own measurement '
            'delta, object_per_sweep shares within each target and sweep, '
            'and track_shared shares across the full tracking ID'
        ),
    )
    parser.add_argument(
        '--temporal_dataset_root', type=str, default=None,
        help=(
            'VoD root containing radar, radar_5frames and lidar; default '
            'is inferred from the configured dataset root'
        ),
    )
    parser.add_argument(
        '--temporal_cache_dir', type=str, default=None,
        help='temporal transform cache directory',
    )
    parser.add_argument(
        '--temporal_max_residual_m', type=float, default=2e-5,
        help='maximum accepted single-to-accumulated rigid residual',
    )
    parser.add_argument('--voxel_mode', type=str, default='fixed',
                        choices=['fixed', 'revoxelize'],
                        help='point topology: fixed pillars or per-step revoxelization')
    parser.add_argument('--point_scope', choices=['scene', 'gt_boxes'],
                        default='scene',
                        help='FGSM/PGD point region; defaults to the full scene')
    parser.add_argument(
        '--target_classes', nargs='+', default=['Car'],
        help=(
            'one or more object classes to attack and evaluate, e.g. '
            '--target_classes Car Pedestrian Cyclist'
        ),
    )
    parser.add_argument('--point_box_margin', type=float, default=0.0,
                        help='FGSM/PGD target GT box margin in metres')
    parser.add_argument(
        '--point_target_selection',
        choices=['all_gt', 'clean_detected'],
        default='all_gt',
        help=(
            'select every target-class GT or only targets detected on the '
            'clean input; clean_detected enables loss-controlled comparisons'
        ),
    )
    parser.add_argument(
        '--attack_loss',
        choices=[
            'training', 'object_evidence', 'object_hybrid', 'object_iou_s'
        ],
        default='training',
        help='gradient objective for point FGSM/PGD',
    )
    parser.add_argument(
        '--object_loss_iou_threshold', type=float, default=None,
        help=(
            'optional shared clean-detection IoU for object-loss targets; '
            'default uses the class-specific evaluation thresholds'
        ),
    )
    parser.add_argument('--object_loss_candidate_margin', type=float, default=1.0,
                        help='margin around target boxes for fixed anchor candidates')
    parser.add_argument('--object_loss_candidate_topk', type=int, default=32,
                        help='nearest-anchor fallback when a target has no candidates')
    parser.add_argument('--object_loss_temperature', type=float, default=1.0,
                        help='LogSumExp temperature of the object-evidence loss')
    parser.add_argument('--hybrid_localization_weight', type=float, default=1.0,
                        help='weight of target-specific box regression divergence')
    parser.add_argument('--hybrid_localization_topk', type=int, default=32,
                        help='clean high-IoU anchors used by the hybrid localization term')
    parser.add_argument('--iou_s_candidate_topk', type=int, default=32,
                        help='clean high-IoU anchors retained per IoU-S target')
    parser.add_argument('--iou_s_score_weight', type=float, default=1.0,
                        help='weight of the IoU-S confidence-suppression term')
    parser.add_argument('--iou_s_iou_weight', type=float, default=1.0,
                        help='weight of the IoU-S differentiable 3D-IoU term')
    parser.add_argument('--iou_s_log_epsilon', type=float, default=1e-6,
                        help='numerical epsilon inside IoU-S logarithms')
    parser.add_argument('--iou_s_original_steps', type=int, default=500,
                        help='official IoU-S point-perturbation Adam steps')
    parser.add_argument('--iou_s_original_lr', type=float, default=0.01,
                        help='official IoU-S point-perturbation Adam learning rate')
    parser.add_argument('--iou_s_original_init_noise', type=float, default=0.01,
                        help='positive uniform XYZ initialization amplitude')
    parser.add_argument('--iou_s_original_distance_weight', type=float, default=1.0,
                        help='weight of Chamfer plus global XYZ L2 regularization')
    parser.add_argument('--iou_s_original_log_epsilon', type=float, default=1e-8,
                        help='official numerical epsilon inside IoU-S logarithms')
    parser.add_argument('--iou_s_original_chamfer_chunk_size', type=int,
                        default=1024,
                        help='point chunk size for exact Chamfer computation')
    parser.add_argument(
        '--iou_s_original_return_policy',
        choices=['joint_best', 'last'],
        default='joint_best',
        help=(
            'returned IoU-S iterate: official joint distance/total best, '
            'or the final Adam iterate for diagnosis'
        ),
    )
    parser.add_argument('--iadv_steps', type=int, default=10,
                        help='I-ADV iteration count')
    parser.add_argument('--iadv_scope', choices=['gt_boxes', 'scene'],
                        default='gt_boxes',
                        help='I-ADV target points: target-class GT boxes or full scene')
    parser.add_argument(
        '--iadv_neighbor_scope',
        choices=['object', 'attack_union', 'scene'],
        default='object',
        help=(
            'I-ADV PCA/fusion scope: per target object (default), legacy '
            'union of attacked points, or full model-active scene'
        ),
    )
    parser.add_argument('--iadv_attack_voxel_size', type=float, default=0.1,
                        help='edge length in metres for I-ADV gradient-fusion cubes')
    parser.add_argument('--iadv_mu', type=float, default=1.0,
                        help='I-ADV momentum decay factor')
    parser.add_argument('--iadv_lambda', type=float, default=1000.0,
                        help='I-ADV reflectivity gradient enhancement factor')
    parser.add_argument('--iadv_d_max', type=float, default=75.0,
                        help='I-ADV maximum sensor range in metres')
    parser.add_argument('--iadv_k_neighbors', type=int, default=16,
                        help='PCA k-nearest points including the query point')
    parser.add_argument('--iadv_min_neighbors', type=int, default=3,
                        help='minimum valid points for PCA; otherwise distance-only fallback')
    parser.add_argument('--iadv_neighbor_radius', type=float, default=None,
                        help='optional maximum KD-tree neighbour distance in metres')
    parser.add_argument('--iadv_gradient_norm', choices=['l1', 'l2'], default='l1',
                        help='per-sample RCS gradient normalization; L1 follows MI-FGSM')
    parser.add_argument('--iadv_box_margin', type=float, default=0.0,
                        help='margin in metres around target GT boxes')
    parser.add_argument('--iadv_rcs_min', type=float, default=None,
                        help='optional lower legal bound for adversarial RCS')
    parser.add_argument('--iadv_rcs_max', type=float, default=None,
                        help='optional upper legal bound for adversarial RCS')
    parser.add_argument(
        '--object_iou_thresholds', nargs='*', default=None,
        metavar='CLASS=VALUE',
        help=(
            'class-specific strict IoU thresholds; defaults are '
            'Car=0.5 Pedestrian=0.25 Cyclist=0.25'
        ),
    )
    parser.add_argument(
        '--object_score_threshold', type=float, default=None,
        help=(
            'prediction score threshold for object outcomes; default uses '
            'MODEL.POST_PROCESSING.SCORE_THRESH'
        ),
    )
    parser.add_argument('--num_samples', type=int, default=None, help='number of samples to attack')
    parser.add_argument('--sample_strategy', choices=['first', 'uniform', 'random'],
                        default='first',
                        help='validation subset selection when num_samples is set')
    parser.add_argument('--save_adv', action='store_true', default=False, help='save adversarial samples')
    parser.add_argument('--adv_dir', type=str, default=None,
                        help='directory for adversarial raw points (default: experiment output/adversarial_points)')
    parser.add_argument('--adv_format', choices=['npy', 'bin'], default='npy',
                        help='saved adversarial point-cloud format')
    parser.add_argument('--vod_eval', dest='vod_eval', action='store_true',
                        default=True, help='run the official VoD clean/adv evaluation')
    parser.add_argument('--no_vod_eval', dest='vod_eval', action='store_false',
                        help='skip the official VoD evaluation')
    parser.add_argument('--vod_devkit', type=str, default='~/VoD-evaluation',
                        help='path to the official View-of-Delft devkit')
    parser.add_argument('--vod_label_dir', type=str, default=None,
                        help='VoD label_2 directory (default: dataset training/label_2)')
    parser.add_argument('--vod_score_threshold', type=float, default=-1.0,
                        help='official evaluator score filter; -1 keeps all model outputs')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    if args.epsilon < 0:
        parser.error('--epsilon must be non-negative')
    if args.seed < 0:
        parser.error('--seed must be non-negative')
    if args.pgd_steps <= 0:
        parser.error('--pgd_steps must be positive')
    if args.iadv_steps <= 0:
        parser.error('--iadv_steps must be positive')
    if args.iou_s_original_steps <= 0:
        parser.error('--iou_s_original_steps must be positive')
    if args.iou_s_original_lr <= 0:
        parser.error('--iou_s_original_lr must be positive')
    if args.iou_s_original_init_noise < 0:
        parser.error('--iou_s_original_init_noise must be non-negative')
    if args.iou_s_original_distance_weight < 0:
        parser.error('--iou_s_original_distance_weight must be non-negative')
    if not 0 < args.iou_s_original_log_epsilon < 1:
        parser.error('--iou_s_original_log_epsilon must be in (0, 1)')
    if args.iou_s_original_chamfer_chunk_size <= 0:
        parser.error('--iou_s_original_chamfer_chunk_size must be positive')
    if args.step_size is not None and args.step_size <= 0:
        parser.error('--step_size must be positive')
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error('--num_samples must be positive')
    if args.object_score_threshold is not None and not 0 <= args.object_score_threshold <= 1:
        parser.error('--object_score_threshold must be in [0, 1]')
    if args.object_loss_iou_threshold is not None and not 0 <= args.object_loss_iou_threshold <= 1:
        parser.error('--object_loss_iou_threshold must be in [0, 1]')
    try:
        parse_class_thresholds(args.object_iou_thresholds)
    except ValueError as error:
        parser.error(str(error))
    if args.save_adv and args.attack_domain != 'point':
        parser.error('--save_adv requires --attack_domain point')
    if args.attack_type == 'iadv':
        if args.attack_domain != 'point':
            parser.error('--attack_type iadv requires --attack_domain point')
        if args.attack_feature not in {'rcs', 'intensity'}:
            parser.error('--attack_type iadv requires --attack_feature rcs')
        if args.epsilon_rcs is None:
            parser.error('--attack_type iadv requires explicit --epsilon_rcs')
        if args.random_start:
            parser.error('I-ADV Algorithm 1 does not use --random_start')
        if args.iadv_neighbor_scope == 'object' and args.iadv_scope != 'gt_boxes':
            parser.error(
                '--iadv_neighbor_scope object requires --iadv_scope gt_boxes'
            )
    if args.attack_type == 'iou_s_original':
        if args.attack_domain != 'point':
            parser.error('--attack_type iou_s_original requires --attack_domain point')
        if args.attack_space != 'feature' or args.attack_feature != 'xyz':
            parser.error(
                '--attack_type iou_s_original requires '
                '--attack_space feature --attack_feature xyz'
            )
        if args.voxel_mode != 'revoxelize':
            parser.error(
                '--attack_type iou_s_original requires --voxel_mode revoxelize'
            )
        if args.point_scope != 'scene':
            parser.error(
                '--attack_type iou_s_original requires --point_scope scene'
            )
        if args.point_target_selection != 'all_gt':
            parser.error(
                '--attack_type iou_s_original requires '
                '--point_target_selection all_gt'
            )
        if args.attack_loss != 'training':
            parser.error(
                '--attack_loss is not used by iou_s_original; leave it at training'
            )
        if args.batch_size != 1:
            parser.error('--attack_type iou_s_original requires --batch_size 1')
        if args.random_start:
            parser.error(
                'iou_s_original has its own positive uniform initialization; '
                'do not use --random_start'
            )
        if args.current_sweep_only or args.temporal_mode != 'none':
            parser.error(
                'iou_s_original attacks the full accumulated scene and does '
                'not use current-sweep or temporal parameter restrictions'
            )
        if args.step_size is not None or args.epsilon_xyz is not None:
            parser.error(
                'iou_s_original uses Adam and distance regularization, not '
                '--step_size or --epsilon_xyz'
            )
        if args.launcher != 'none':
            parser.error('--attack_type iou_s_original requires --launcher none')
    if args.attack_space == 'radar_measurement':
        if args.attack_domain != 'point':
            parser.error('--attack_space radar_measurement requires --attack_domain point')
        if args.attack_type not in {'fgsm', 'pgd'}:
            parser.error('--attack_space radar_measurement supports FGSM/PGD only')
        if args.attack_feature != 'xyz':
            parser.error('--attack_space radar_measurement requires --attack_feature xyz')
        if args.point_scope == 'scene' and (
            args.temporal_mode != 'point_independent'
        ):
            parser.error(
                'scene-wide radar measurement attack requires '
                '--temporal_mode point_independent'
            )
        if args.voxel_mode != 'revoxelize':
            parser.error(
                '--attack_space radar_measurement requires '
                '--voxel_mode revoxelize'
            )
        if args.step_size is not None:
            parser.error(
                'measurement attack uses the three measurement-specific '
                'step-size options, not --step_size'
            )
        required_budgets = (
            'epsilon_range',
            'epsilon_azimuth_deg',
            'epsilon_elevation_deg',
        )
        missing = [name for name in required_budgets if getattr(args, name) is None]
        if missing:
            parser.error(
                '--attack_space radar_measurement requires explicit '
                + ', '.join(f'--{name}' for name in missing)
            )
    if args.attack_loss in {
        'object_evidence', 'object_hybrid', 'object_iou_s'
    }:
        if args.attack_domain != 'point' or args.attack_type == 'iadv':
            parser.error(
                'object attack losses support point FGSM/PGD only'
            )
        if args.point_scope != 'gt_boxes':
            parser.error(
                'object attack losses require --point_scope gt_boxes'
            )
        if args.point_target_selection != 'clean_detected':
            parser.error(
                'object attack losses require '
                '--point_target_selection clean_detected'
            )
    if (args.point_target_selection == 'clean_detected'
            and (args.attack_domain != 'point'
                 or args.attack_type == 'iadv'
                 or args.point_scope != 'gt_boxes')):
        parser.error(
            '--point_target_selection clean_detected supports point FGSM/PGD '
            'with --point_scope gt_boxes only'
        )
    if args.current_sweep_only and (
        args.attack_domain != 'point'
        or args.attack_type == 'iadv'
        or args.point_scope != 'gt_boxes'
    ):
        parser.error(
            '--current_sweep_only supports point FGSM/PGD with '
            '--point_scope gt_boxes only'
        )
    if args.temporal_mode != 'none':
        if args.attack_space != 'radar_measurement':
            parser.error(
                'non-none --temporal_mode requires '
                '--attack_space radar_measurement'
            )
        if args.current_sweep_only:
            parser.error(
                'non-none --temporal_mode conflicts with '
                '--current_sweep_only'
            )
        if args.launcher != 'none':
            parser.error(
                'non-none --temporal_mode currently requires '
                '--launcher none'
            )
    if args.temporal_max_residual_m <= 0:
        parser.error('--temporal_max_residual_m must be positive')
    if args.iadv_attack_voxel_size <= 0:
        parser.error('--iadv_attack_voxel_size must be positive')
    if args.iadv_mu < 0:
        parser.error('--iadv_mu must be non-negative')
    if args.iadv_lambda < 0:
        parser.error('--iadv_lambda must be non-negative')
    if args.iadv_d_max <= 0:
        parser.error('--iadv_d_max must be positive')
    if args.iadv_k_neighbors <= 0:
        parser.error('--iadv_k_neighbors must be positive')
    if args.iadv_min_neighbors < 3:
        parser.error('--iadv_min_neighbors must be at least 3')
    if args.iadv_min_neighbors > args.iadv_k_neighbors:
        parser.error('--iadv_min_neighbors must not exceed --iadv_k_neighbors')
    if args.iadv_neighbor_radius is not None and args.iadv_neighbor_radius <= 0:
        parser.error('--iadv_neighbor_radius must be positive')
    if args.iadv_box_margin < 0:
        parser.error('--iadv_box_margin must be non-negative')
    if args.point_box_margin < 0:
        parser.error('--point_box_margin must be non-negative')
    if args.object_loss_candidate_margin < 0:
        parser.error('--object_loss_candidate_margin must be non-negative')
    if args.object_loss_candidate_topk <= 0:
        parser.error('--object_loss_candidate_topk must be positive')
    if args.object_loss_temperature <= 0:
        parser.error('--object_loss_temperature must be positive')
    if args.hybrid_localization_weight < 0:
        parser.error('--hybrid_localization_weight must be non-negative')
    if (args.attack_loss == 'object_hybrid'
            and args.hybrid_localization_weight <= 0):
        parser.error(
            '--attack_loss object_hybrid requires positive '
            '--hybrid_localization_weight'
        )
    if args.hybrid_localization_topk <= 0:
        parser.error('--hybrid_localization_topk must be positive')
    if args.iou_s_candidate_topk <= 0:
        parser.error('--iou_s_candidate_topk must be positive')
    if args.iou_s_score_weight < 0 or args.iou_s_iou_weight < 0:
        parser.error('--iou_s_score_weight and --iou_s_iou_weight must be non-negative')
    if (args.attack_loss == 'object_iou_s'
            and args.iou_s_score_weight == 0
            and args.iou_s_iou_weight == 0):
        parser.error('--attack_loss object_iou_s requires a positive IoU-S weight')
    if not 0 < args.iou_s_log_epsilon < 1:
        parser.error('--iou_s_log_epsilon must be in (0, 1)')
    if (args.iadv_rcs_min is not None and args.iadv_rcs_max is not None
            and args.iadv_rcs_min > args.iadv_rcs_max):
        parser.error('--iadv_rcs_min must not exceed --iadv_rcs_max')
    if args.vod_eval and args.launcher != 'none':
        parser.error('official VoD evaluation currently requires --launcher none')
    if (args.num_samples is not None and args.sample_strategy != 'first'
            and args.launcher != 'none'):
        parser.error('uniform/random sample selection requires --launcher none')
    for name in ['epsilon_xyz', 'epsilon_rcs', 'epsilon_doppler', 'epsilon_time']:
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f'--{name} must be non-negative')
    for name in (
        'epsilon_range', 'epsilon_azimuth_deg', 'epsilon_elevation_deg',
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f'--{name} must be non-negative')
    for name in (
        'step_size_range', 'step_size_azimuth_deg',
        'step_size_elevation_deg',
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'--{name} must be positive')

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.measurement_budget_semantics = (
        'research_digital_not_validated_physical_sensor_uncertainty'
    )

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def evaluate_attack(model, dataloader, args, logger, output_dir):
    model.eval()
    if (args.point_target_selection == 'clean_detected'
            and model.dense_head.__class__.__name__ != 'AnchorHeadSingle'):
        raise NotImplementedError(
            'clean-detected target selection and object losses currently '
            'support AnchorHeadSingle only; got '
            f'{model.dense_head.__class__.__name__}'
        )
    metrics = DetectionAttackMetrics()
    writer = None
    dataset = dataloader.dataset
    feature_names = get_feature_names(cfg.DATA_CONFIG)
    target_class_names, target_class_ids, object_iou_thresholds = (
        resolve_object_targets(args, dataset.class_names)
    )
    selected_classes = dict(zip(target_class_ids, target_class_names))
    all_class_names = {
        class_id: class_name
        for class_id, class_name in enumerate(dataset.class_names, start=1)
    }
    model_score_threshold = float(cfg.MODEL.POST_PROCESSING.SCORE_THRESH)
    object_score_threshold = (
        model_score_threshold
        if args.object_score_threshold is None
        else float(args.object_score_threshold)
    )
    if not 0 <= object_score_threshold <= 1:
        raise ValueError('resolved object score threshold must be in [0, 1]')
    if object_score_threshold < model_score_threshold:
        raise ValueError(
            '--object_score_threshold cannot be lower than '
            f'MODEL.POST_PROCESSING.SCORE_THRESH={model_score_threshold:g}; '
            'lower-score predictions have already been removed'
        )
    object_loss_iou_thresholds = dict(object_iou_thresholds)
    if args.object_loss_iou_threshold is not None:
        object_loss_iou_thresholds = {
            class_id: float(args.object_loss_iou_threshold)
            for class_id in target_class_ids
        }
    logger.info('Object target classes: %s', ', '.join(target_class_names))
    logger.info(
        'Object IoU thresholds: %s',
        ', '.join(
            f'{selected_classes[class_id]}={threshold:g}'
            for class_id, threshold in object_iou_thresholds.items()
        ),
    )
    logger.info('Object score threshold: %g', object_score_threshold)
    clean_prediction_dir = None
    adversarial_prediction_dir = None
    vod_label_dir = None
    temporal_resolver = None

    if args.temporal_mode != 'none':
        configured_root = Path(dataset.root_path).expanduser().resolve()
        temporal_dataset_root = (
            Path(args.temporal_dataset_root).expanduser().resolve()
            if args.temporal_dataset_root is not None
            else configured_root.parent
        )
        temporal_cache_dir = (
            Path(args.temporal_cache_dir).expanduser().resolve()
            if args.temporal_cache_dir is not None
            else cfg.ROOT_DIR / 'output/radar_attack/temporal_sweep_cache'
        )
        temporal_resolver = TemporalSweepResolver(
            dataset_root=temporal_dataset_root,
            cache_dir=temporal_cache_dir,
            max_residual_m=args.temporal_max_residual_m,
        )
        logger.info('Temporal mode: %s', args.temporal_mode)
        logger.info('Temporal VoD root: %s', temporal_dataset_root)
        logger.info('Temporal transform cache: %s', temporal_cache_dir)

    if args.vod_eval:
        prediction_root = output_dir / 'vod_predictions'
        clean_prediction_dir = prepare_prediction_directory(
            prediction_root / 'clean'
        )
        adversarial_prediction_dir = prepare_prediction_directory(
            prediction_root / 'adversarial'
        )
        vod_label_dir = resolve_vod_label_dir(
            dataset,
            args.vod_label_dir,
        )
        logger.info('Official VoD labels: %s', vod_label_dir)
        logger.info('Clean VoD predictions: %s', clean_prediction_dir)
        logger.info(
            'Adversarial VoD predictions: %s',
            adversarial_prediction_dir,
        )
        if model_score_threshold > 0:
            logger.warning(
                'MODEL.POST_PROCESSING.SCORE_THRESH=%s removes lower-score '
                'predictions before VoD AP evaluation',
                model_score_threshold,
            )

    if args.attack_domain == 'point':
        voxel_size, max_points, max_voxels = get_voxel_settings(cfg.DATA_CONFIG)
        epsilon_overrides = {
            'xyz': args.epsilon_xyz,
            'rcs': args.epsilon_rcs,
            'doppler': args.epsilon_doppler,
            'time': args.epsilon_time,
        }
        if args.save_adv:
            adversarial_dir = (
                Path(args.adv_dir)
                if args.adv_dir is not None
                else output_dir / 'adversarial_points'
            )
            writer = AdversarialPointCloudWriter(
                output_dir=adversarial_dir,
                feature_names=feature_names,
                file_format=args.adv_format,
                run_metadata={
                    'seed': args.seed,
                    'attack_domain': args.attack_domain,
                    'attack_type': args.attack_type,
                    'attack_feature': args.attack_feature,
                    'attack_space': args.attack_space,
                    'epsilon': args.epsilon,
                    'epsilon_xyz': args.epsilon_xyz,
                    'epsilon_rcs': args.epsilon_rcs,
                    'epsilon_doppler': args.epsilon_doppler,
                    'epsilon_time': args.epsilon_time,
                    'epsilon_range': args.epsilon_range,
                    'epsilon_azimuth_deg': args.epsilon_azimuth_deg,
                    'epsilon_elevation_deg': args.epsilon_elevation_deg,
                    'step_size_range': args.step_size_range,
                    'step_size_azimuth_deg': args.step_size_azimuth_deg,
                    'step_size_elevation_deg': args.step_size_elevation_deg,
                    'measurement_budget_semantics': (
                        args.measurement_budget_semantics
                    ),
                    'temporal_mode': args.temporal_mode,
                    'temporal_dataset_root': args.temporal_dataset_root,
                    'temporal_cache_dir': args.temporal_cache_dir,
                    'temporal_max_residual_m': (
                        args.temporal_max_residual_m
                    ),
                    'pgd_steps': args.pgd_steps,
                    'step_size': args.step_size,
                    'random_start': args.random_start,
                    'voxel_mode': args.voxel_mode,
                    'sample_strategy': args.sample_strategy,
                    'point_scope': args.point_scope,
                    'target_classes': target_class_names,
                    'point_box_margin': args.point_box_margin,
                    'point_target_selection': args.point_target_selection,
                    'attack_loss': args.attack_loss,
                    'object_loss_iou_threshold': (
                        args.object_loss_iou_threshold
                    ),
                    'object_loss_candidate_margin': (
                        args.object_loss_candidate_margin
                    ),
                    'object_loss_candidate_topk': (
                        args.object_loss_candidate_topk
                    ),
                    'object_loss_temperature': args.object_loss_temperature,
                    'hybrid_localization_weight': (
                        args.hybrid_localization_weight
                    ),
                    'hybrid_localization_topk': (
                        args.hybrid_localization_topk
                    ),
                    'iou_s_candidate_topk': args.iou_s_candidate_topk,
                    'iou_s_score_weight': args.iou_s_score_weight,
                    'iou_s_iou_weight': args.iou_s_iou_weight,
                    'iou_s_log_epsilon': args.iou_s_log_epsilon,
                    'iou_s_original_steps': args.iou_s_original_steps,
                    'iou_s_original_lr': args.iou_s_original_lr,
                    'iou_s_original_init_noise': (
                        args.iou_s_original_init_noise
                    ),
                    'iou_s_original_distance_weight': (
                        args.iou_s_original_distance_weight
                    ),
                    'iou_s_original_log_epsilon': (
                        args.iou_s_original_log_epsilon
                    ),
                    'iou_s_original_chamfer_chunk_size': (
                        args.iou_s_original_chamfer_chunk_size
                    ),
                    'iou_s_original_return_policy': (
                        args.iou_s_original_return_policy
                    ),
                    'iadv_steps': args.iadv_steps,
                    'iadv_scope': args.iadv_scope,
                    'iadv_neighbor_scope': args.iadv_neighbor_scope,
                    'iadv_attack_voxel_size': args.iadv_attack_voxel_size,
                    'iadv_mu': args.iadv_mu,
                    'iadv_lambda': args.iadv_lambda,
                    'iadv_d_max': args.iadv_d_max,
                    'iadv_k_neighbors': args.iadv_k_neighbors,
                    'iadv_min_neighbors': args.iadv_min_neighbors,
                    'iadv_neighbor_radius': args.iadv_neighbor_radius,
                    'iadv_gradient_norm': args.iadv_gradient_norm,
                    'iadv_box_margin': args.iadv_box_margin,
                    'iadv_rcs_min': args.iadv_rcs_min,
                    'iadv_rcs_max': args.iadv_rcs_max,
                    'object_iou_thresholds': {
                        selected_classes[class_id]: threshold
                        for class_id, threshold in object_iou_thresholds.items()
                    },
                    'object_score_threshold': object_score_threshold,
                },
            )
            logger.info('Adversarial raw points will be saved to %s', adversarial_dir)

    if args.num_samples is not None:
        total_samples_expected = min(args.num_samples, len(dataloader.dataset))
    else:
        total_samples_expected = len(dataloader.dataset)

    progress_bar = tqdm.tqdm(
        total=total_samples_expected,
        leave=True,
        desc='Attack Evaluation',
        dynamic_ncols=True,
    )

    for batch_dict in dataloader:
        if (
            args.num_samples is not None
            and metrics.total_samples >= args.num_samples
        ):
            break

        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts_original, ret_dict_original = model(dict(batch_dict))
        if clean_prediction_dir is not None:
            dataset.generate_prediction_dicts(
                batch_dict,
                pred_dicts_original,
                dataset.class_names,
                output_path=clean_prediction_dir,
            )

        objective = None
        clean_object_evidence = None
        target_context = None
        target_diagnostics = None
        attack_point_mask = None
        clean_records_by_batch = {}
        if args.attack_domain == 'point':
            original_points = batch_dict['points'].detach().clone()
            voxelizer = PointCloudVoxelizer(
                cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                voxel_size,
                max_points,
                max_voxels,
                int(batch_dict['batch_size']),
            )
            if args.attack_type == 'iadv':
                attack_output = iadv_rcs_attack(
                    model=model,
                    batch_dict=batch_dict,
                    voxelizer=voxelizer,
                    feature_names=feature_names,
                    epsilon_rcs=args.epsilon_rcs,
                    steps=args.iadv_steps,
                    step_size=args.step_size,
                    attack_voxel_size=args.iadv_attack_voxel_size,
                    momentum_decay=args.iadv_mu,
                    gradient_enhancement=args.iadv_lambda,
                    d_max=args.iadv_d_max,
                    k_neighbors=args.iadv_k_neighbors,
                    min_neighbors=args.iadv_min_neighbors,
                    neighbor_radius=args.iadv_neighbor_radius,
                    gradient_norm=args.iadv_gradient_norm,
                    scope=args.iadv_scope,
                    neighbor_scope=args.iadv_neighbor_scope,
                    target_class_ids=(
                        target_class_ids if args.iadv_scope == 'gt_boxes' else None
                    ),
                    box_margin=args.iadv_box_margin,
                    rcs_min=args.iadv_rcs_min,
                    rcs_max=args.iadv_rcs_max,
                )
            elif args.attack_type == 'iou_s_original':
                attack_output = original_iou_s_point_attack(
                    model=model,
                    batch_dict=batch_dict,
                    voxelizer=voxelizer,
                    steps=args.iou_s_original_steps,
                    learning_rate=args.iou_s_original_lr,
                    initial_noise=args.iou_s_original_init_noise,
                    distance_weight=args.iou_s_original_distance_weight,
                    log_epsilon=args.iou_s_original_log_epsilon,
                    chamfer_chunk_size=(
                        args.iou_s_original_chamfer_chunk_size
                    ),
                    return_policy=args.iou_s_original_return_policy,
                )
            else:
                point_mask = None
                allowed_rows = None
                temporal_data = None
                if args.point_target_selection == 'clean_detected':
                    for batch_index, clean_prediction in enumerate(
                        pred_dicts_original
                    ):
                        _, clean_records = compare_target_object_endpoints(
                            clean_prediction=clean_prediction,
                            adversarial_prediction=clean_prediction,
                            gt_boxes=batch_dict['gt_boxes'][batch_index],
                            target_classes=selected_classes,
                            iou_thresholds=object_iou_thresholds,
                            frame_id=batch_dict['frame_id'][batch_index],
                            object_score_threshold=object_score_threshold,
                            class_names=all_class_names,
                            batch_index=batch_index,
                        )
                        clean_records_by_batch[batch_index] = clean_records
                    objective = ObjectEvidenceObjective.from_clean_predictions(
                        model=model,
                        batch_dict=batch_dict,
                        clean_predictions=pred_dicts_original,
                        target_class_ids=target_class_ids,
                        iou_thresholds=object_loss_iou_thresholds,
                        candidate_margin=args.object_loss_candidate_margin,
                        candidate_topk=args.object_loss_candidate_topk,
                        temperature=args.object_loss_temperature,
                        point_box_margin=args.point_box_margin,
                        localization_weight=(
                            args.hybrid_localization_weight
                            if args.attack_loss == 'object_hybrid'
                            else 0.0
                        ),
                        localization_topk=args.hybrid_localization_topk,
                        objective_type=(
                            args.attack_loss
                            if args.attack_loss != 'training'
                            else 'object_evidence'
                        ),
                        iou_s_candidate_topk=args.iou_s_candidate_topk,
                        iou_s_score_weight=args.iou_s_score_weight,
                        iou_s_iou_weight=args.iou_s_iou_weight,
                        iou_s_log_epsilon=args.iou_s_log_epsilon,
                    )
                    allowed_rows = {
                        batch_index: [
                            int(record['gt_row']) for record in records
                        ]
                        for batch_index, records
                        in clean_records_by_batch.items()
                    }
                    objective = objective.restricted_to_gt_rows(
                        allowed_rows,
                        batch_dict,
                        point_box_margin=args.point_box_margin,
                    )
                    target_context = build_target_screening_context(
                        points=original_points,
                        batch_dict=batch_dict,
                        clean_records_by_batch=clean_records_by_batch,
                        feature_names=feature_names,
                        voxelizer=voxelizer,
                        box_margin=args.point_box_margin,
                    )
                    point_mask = target_context.target_mask
                    clean_object_evidence = objective.evidence_by_target(model)
                elif args.point_scope == 'gt_boxes':
                    point_mask = build_iadv_attack_mask(
                        original_points,
                        batch_dict,
                        scope='gt_boxes',
                        target_class_ids=target_class_ids,
                        box_margin=args.point_box_margin,
                    )
                target_mask = point_mask
                mask_stats = None
                if (
                    args.attack_space == 'radar_measurement'
                    and args.temporal_mode != 'none'
                ):
                    temporal_data = build_temporal_batch_data(
                        resolver=temporal_resolver,
                        batched_points=(
                            original_points.detach().cpu().numpy()
                        ),
                        frame_ids=batch_dict['frame_id'],
                        gt_boxes=(
                            batch_dict['gt_boxes'].detach().cpu().numpy()
                        ),
                        target_classes=target_class_names,
                        dataset_classes=dataset.class_names,
                        allowed_gt_rows=allowed_rows,
                        box_margin=args.point_box_margin,
                    )
                    point_mask, raw_mask_stats = (
                        build_temporal_measurement_attack_mask(
                            original_points,
                            temporal_data,
                            voxelizer,
                            point_scope=args.point_scope,
                        )
                    )
                    mask_stats = {
                        'temporal_attack_target_points': raw_mask_stats[
                            'measurement_target_points'
                        ],
                        'temporal_attack_current_points': raw_mask_stats[
                            'measurement_current_target_points'
                        ],
                        'temporal_attack_historical_points': raw_mask_stats[
                            'measurement_historical_target_points'
                        ],
                        'temporal_attack_clean_active_points': raw_mask_stats[
                            'measurement_clean_active_target_points'
                        ],
                    }
                    if target_context is not None:
                        set_temporal_screening_assignments(
                            target_context,
                            torch.as_tensor(
                                temporal_data.gt_rows,
                                dtype=torch.long,
                                device=original_points.device,
                            ),
                            torch.as_tensor(
                                temporal_data.source_xyz,
                                dtype=original_points.dtype,
                                device=original_points.device,
                            ),
                            torch.as_tensor(
                                temporal_data.rotations,
                                dtype=original_points.dtype,
                                device=original_points.device,
                            ),
                        )
                elif (
                    args.attack_space == 'radar_measurement'
                    or args.current_sweep_only
                ):
                    point_mask, raw_mask_stats = build_measurement_attack_mask(
                        original_points,
                        feature_names,
                        target_mask,
                        voxelizer,
                    )
                    mask_stats = {
                        'current_sweep_target_points': raw_mask_stats[
                            'measurement_target_points'
                        ],
                        'current_sweep_time0_target_points': raw_mask_stats[
                            'measurement_current_target_points'
                        ],
                        'current_sweep_history_target_points': raw_mask_stats[
                            'measurement_historical_target_points'
                        ],
                        'current_sweep_clean_active_target_points': raw_mask_stats[
                            'measurement_clean_active_current_target_points'
                        ],
                    }
                attack_point_mask = point_mask
                loss_fn = (
                    objective
                    if args.attack_loss in {
                        'object_evidence', 'object_hybrid', 'object_iou_s'
                    }
                    else None
                )
                loss_stats = dict(objective.stats) if objective is not None else {}
                if mask_stats is not None:
                    loss_stats.update(mask_stats)
                loss_stats = loss_stats or None
                if args.attack_space == 'radar_measurement':
                    measurement_kwargs = {
                        'model': model,
                        'batch_dict': batch_dict,
                        'voxelizer': voxelizer,
                        'attack_type': args.attack_type,
                        'epsilon_range': args.epsilon_range,
                        'epsilon_azimuth': math.radians(
                            args.epsilon_azimuth_deg
                        ),
                        'epsilon_elevation': math.radians(
                            args.epsilon_elevation_deg
                        ),
                        'pgd_steps': args.pgd_steps,
                        'step_size_range': args.step_size_range,
                        'step_size_azimuth': (
                            math.radians(args.step_size_azimuth_deg)
                            if args.step_size_azimuth_deg is not None
                            else None
                        ),
                        'step_size_elevation': (
                            math.radians(args.step_size_elevation_deg)
                            if args.step_size_elevation_deg is not None
                            else None
                        ),
                        'random_start': args.random_start,
                        'attack_mask': point_mask,
                        'attack_mask_stats': mask_stats,
                        'loss_fn': loss_fn,
                        'loss_stats': loss_stats,
                    }
                    if args.temporal_mode != 'none':
                        attack_output = (
                            radar_temporal_measurement_geometry_attack(
                                temporal_data=temporal_data,
                                temporal_mode=args.temporal_mode,
                                point_scope=args.point_scope,
                                **measurement_kwargs,
                            )
                        )
                    else:
                        attack_output = radar_measurement_geometry_attack(
                            feature_names=feature_names,
                            target_mask=target_mask,
                            **measurement_kwargs,
                        )
                else:
                    attack_output = point_cloud_attack(
                        model=model,
                        batch_dict=batch_dict,
                        voxelizer=voxelizer,
                        feature_names=feature_names,
                        attack_type=args.attack_type,
                        attack_feature=args.attack_feature,
                        epsilon=args.epsilon,
                        epsilon_overrides=epsilon_overrides,
                        pgd_steps=args.pgd_steps,
                        step_size=args.step_size,
                        random_start=args.random_start,
                        voxel_mode=args.voxel_mode,
                        point_mask=point_mask,
                        loss_fn=loss_fn,
                        loss_stats=loss_stats,
                    )
                if target_context is not None:
                    target_diagnostics = target_attack_diagnostics(
                        target_context,
                        attack_output.adv_points,
                        attack_point_mask,
                    )
            batch_dict['points'] = attack_output.adv_points
            batch_dict.update(attack_output.model_inputs)
            metrics.update_perturbation(attack_output.stats)
            if writer is not None:
                writer.save_batch(
                    original_points,
                    attack_output.adv_points,
                    batch_dict,
                )
        else:
            batch_dict['voxels'] = voxel_attack(
                model=model,
                batch_dict=batch_dict,
                epsilon=args.epsilon,
                attack_type=args.attack_type,
                attack_feature=args.attack_feature,
                steps=args.pgd_steps,
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                feature_names=feature_names,
            )

        with torch.no_grad():
            pred_dicts_attacked, ret_dict_attacked = model(dict(batch_dict))
        adversarial_object_evidence = (
            objective.evidence_by_target(model)
            if objective is not None
            else None
        )
        if adversarial_prediction_dir is not None:
            dataset.generate_prediction_dicts(
                batch_dict,
                pred_dicts_attacked,
                dataset.class_names,
                output_path=adversarial_prediction_dir,
            )

        metrics.update_predictions(
            pred_dicts_original,
            pred_dicts_attacked,
            ret_dict_original,
            ret_dict_attacked,
        )
        for batch_index, (clean_prediction, adversarial_prediction) in enumerate(
            zip(pred_dicts_original, pred_dicts_attacked)
        ):
            target_count, endpoint_records = compare_target_object_endpoints(
                clean_prediction=clean_prediction,
                adversarial_prediction=adversarial_prediction,
                gt_boxes=batch_dict['gt_boxes'][batch_index],
                target_classes=selected_classes,
                iou_thresholds=object_iou_thresholds,
                frame_id=batch_dict['frame_id'][batch_index],
                object_score_threshold=object_score_threshold,
                class_names=all_class_names,
                batch_index=batch_index,
                clean_evidence=clean_object_evidence,
                adversarial_evidence=adversarial_object_evidence,
            )
            if target_diagnostics is not None:
                merge_target_diagnostics(
                    endpoint_records, target_diagnostics
                )
            metrics.update_object_endpoints(target_count, endpoint_records)
        progress_bar.update(len(pred_dicts_original))

    progress_bar.close()
    results = metrics.compute()
    for class_name in target_class_names:
        results['object_outcomes']['by_class'].setdefault(
            class_name,
            {
                'eligible_clean_objects': 0,
                'counts': {
                    'still_correct': 0,
                    'misclassification': 0,
                    'localization_failure': 0,
                    'pure_hiding': 0,
                },
                'rates': {
                    'object_failure_asr': 0.0,
                    'pure_hiding_asr': 0.0,
                    'misclassification_rate': 0.0,
                    'localization_failure_rate': 0.0,
                    'still_correct_rate': 0.0,
                },
            },
        )
    results['object_outcomes']['target_classes'] = target_class_names
    results['object_outcomes']['iou_thresholds'] = {
        selected_classes[class_id]: threshold
        for class_id, threshold in object_iou_thresholds.items()
    }
    results['object_outcomes']['score_threshold'] = object_score_threshold
    if metrics.object_endpoint_records and all(
        'num_active_time0_points' in record
        for record in metrics.object_endpoint_records
    ):
        results['current_sweep_summary'] = summarize_current_sweep(
            metrics.object_endpoint_records
        )
        results['current_sweep_files'] = write_current_sweep_outputs(
            metrics.object_endpoint_records, output_dir
        )
        results['target_correlations'] = target_correlations(
            metrics.object_endpoint_records
        )
        results['target_correlation_file'] = write_correlation_output(
            results['target_correlations'], output_dir
        )
    object_endpoint_path = output_dir / 'object_endpoint_metrics.csv'
    metrics.write_object_endpoints(object_endpoint_path)
    results['object_endpoint_file'] = str(object_endpoint_path)
    if clean_prediction_dir is not None:
        logger.info('Running official VoD evaluator for clean predictions...')
        results['vod_official'] = evaluate_vod_pair(
            clean_prediction_dir=clean_prediction_dir,
            adversarial_prediction_dir=adversarial_prediction_dir,
            label_dir=vod_label_dir,
            devkit_path=Path(args.vod_devkit),
            class_names=dataset.class_names,
            score_threshold=args.vod_score_threshold,
        )
        results['vod_official']['full_validation_set'] = (
            results['vod_official']['frames'] == len(dataset)
        )

    logger.info('=' * 70)
    logger.info('Attack Results for 4D Radar Data:')
    logger.info(f'Attack Domain: {args.attack_domain}')
    logger.info(f'Attack Type: {args.attack_type.upper()}')
    if args.attack_type == 'iou_s_original':
        logger.info('Default Epsilon: unused (no hard projection)')
    else:
        logger.info(f'Default Epsilon: {args.epsilon}')
    logger.info(f'Attack Feature: {args.attack_feature}')
    logger.info(f'Attack Space: {args.attack_space}')
    if args.attack_domain == 'point' and args.attack_type not in {
        'iadv', 'iou_s_original'
    }:
        logger.info(f'Attack Loss: {args.attack_loss}')
        logger.info(f'Point Target Selection: {args.point_target_selection}')
    if args.attack_type == 'pgd':
        logger.info(f'PGD Steps: {args.pgd_steps}')
    elif args.attack_type == 'iadv':
        logger.info(f'I-ADV Steps: {args.iadv_steps}')
        logger.info(f'I-ADV Scope: {args.iadv_scope}')
        logger.info(f'I-ADV Neighbor Scope: {args.iadv_neighbor_scope}')
        logger.info('I-ADV Target Classes: %s', ', '.join(target_class_names))
    elif args.attack_type == 'iou_s_original':
        logger.info('Original IoU-S Adam Steps: %d', args.iou_s_original_steps)
        logger.info('Original IoU-S Adam LR: %g', args.iou_s_original_lr)
        logger.info(
            'Original IoU-S init / distance weight: %g / %g',
            args.iou_s_original_init_noise,
            args.iou_s_original_distance_weight,
        )
        logger.info('Original IoU-S Point Scope: full scene')
        logger.info('Original IoU-S Hard Epsilon Projection: disabled')
        logger.info(
            'Original IoU-S Return Policy: %s',
            args.iou_s_original_return_policy,
        )
        logger.info(
            'Original IoU-S evaluation target classes: %s',
            ', '.join(target_class_names),
        )
    if args.attack_domain == 'point':
        for group in ('xyz', 'rcs', 'doppler', 'time'):
            value = getattr(args, f'epsilon_{group}')
            if value is not None:
                logger.info('Epsilon %s: %s', group, value)
        logger.info(f'Voxel Mode: {args.voxel_mode}')
        if args.attack_space == 'radar_measurement':
            logger.info('Temporal Mode: %s', args.temporal_mode)
            logger.info(
                'Measurement Sweep: %s',
                {
                    'none': 'current (time == 0)',
                    'point_independent': (
                        'all available, point-independent'
                    ),
                    'object_per_sweep': (
                        'all available, object-shared within each sweep'
                    ),
                    'track_shared': 'all available, track-shared',
                }[args.temporal_mode],
            )
            logger.info('Range Epsilon [m]: %s', args.epsilon_range)
            logger.info(
                'Azimuth / Elevation Epsilon [deg]: %s / %s',
                args.epsilon_azimuth_deg,
                args.epsilon_elevation_deg,
            )
        logger.info(f'Max |delta|: {results["max_abs_perturbation"]:.6f}')
        logger.info(f'Mean |delta|: {results["mean_abs_perturbation"]:.6f}')
        if writer is not None:
            logger.info(f'Saved adversarial point clouds: {writer.saved_samples}')
            logger.info(f'Adversarial manifest: {writer.manifest_path}')
        diagnostics = results.get('attack_diagnostics', {})
        if args.attack_space == 'radar_measurement':
            if args.temporal_mode != 'none':
                logger.info(
                    'Temporal parameter groups (all / active): %d / %d',
                    int(diagnostics.get('temporal_parameter_groups', 0.0)),
                    int(diagnostics.get(
                        'temporal_active_parameter_groups', 0.0
                    )),
                )
            logger.info(
                'Measurement max delta range [m] / azimuth [deg] / '
                'elevation [deg]: %.6f / %.6f / %.6f',
                diagnostics.get('measurement_max_abs_delta_range', 0.0),
                math.degrees(diagnostics.get(
                    'measurement_max_abs_delta_azimuth_rad', 0.0
                )),
                math.degrees(diagnostics.get(
                    'measurement_max_abs_delta_elevation_rad', 0.0
                )),
            )
            logger.info(
                'Measurement Cartesian displacement max / mean L2 [m]: '
                '%.6f / %.6f',
                diagnostics.get('measurement_max_xyz_l2', 0.0),
                diagnostics.get('measurement_mean_xyz_l2', 0.0),
            )
            logger.info(
                'Measurement current active points / modified points: %d / %d',
                int(diagnostics.get(
                    'measurement_clean_active_current_target_points', 0.0
                )),
                int(diagnostics.get(
                    'measurement_modified_current_points', 0.0
                )),
            )
            logger.info(
                'Measurement historical / non-target / non-geometry '
                'modifications: %d / %d / %d',
                int(diagnostics.get(
                    'measurement_historical_modification_count', 0.0
                )),
                int(diagnostics.get(
                    'measurement_non_target_modification_count', 0.0
                )),
                int(diagnostics.get(
                    'measurement_non_geometry_modification_count', 0.0
                )),
            )
        if 'object_evidence_targets' in diagnostics:
            logger.info(
                'Clean-detected attack targets: %d; mean candidates/target: %.1f',
                int(diagnostics['object_evidence_targets']),
                diagnostics['object_evidence_mean_candidates_per_target'],
            )
        if args.attack_loss == 'object_iou_s':
            logger.info(
                'Radar Object IoU-S targets: %d; mean candidates/target: %.1f; '
                'score/IoU weights: %g/%g',
                int(diagnostics.get('object_iou_s_targets', 0.0)),
                diagnostics.get(
                    'object_iou_s_mean_candidates_per_target', 0.0
                ),
                args.iou_s_score_weight,
                args.iou_s_iou_weight,
            )
        if args.attack_type == 'iou_s_original':
            logger.info(
                'Original IoU-S mean predictions / GT-prediction pairs '
                'per step: %.1f / %.1f',
                diagnostics.get(
                    'iou_s_original_mean_predictions_per_step', 0.0
                ),
                diagnostics.get('iou_s_original_mean_pairs_per_step', 0.0),
            )
            logger.info(
                'Original IoU-S best attack / distance / total loss: '
                '%.6f / %.6f / %.6f',
                diagnostics.get('iou_s_original_best_attack_loss', 0.0),
                diagnostics.get('iou_s_original_best_distance_loss', 0.0),
                diagnostics.get('iou_s_original_best_total_loss', 0.0),
            )
            logger.info(
                'Original IoU-S mean best step / early <=10 / early <=100: '
                '%.1f / %.3f / %.3f',
                diagnostics.get('iou_s_original_best_step', 0.0),
                diagnostics.get('iou_s_original_best_step_le_10', 0.0),
                diagnostics.get('iou_s_original_best_step_le_100', 0.0),
            )
            logger.info(
                'Original IoU-S actual joint-best / last attack loss per pair: '
                '%.6f / %.6f',
                diagnostics.get(
                    'iou_s_original_joint_best_endpoint_attack_loss_per_pair',
                    0.0,
                ),
                diagnostics.get(
                    'iou_s_original_last_attack_loss_per_pair', 0.0
                ),
            )
            logger.info(
                'Original IoU-S prediction retention / switch / count-change '
                'fractions: %.3f / %.3f / %.3f',
                diagnostics.get(
                    'iou_s_original_mean_prediction_retention', 0.0
                ),
                diagnostics.get(
                    'iou_s_original_prediction_switch_fraction', 0.0
                ),
                diagnostics.get(
                    'iou_s_original_prediction_count_change_fraction', 0.0
                ),
            )
    logger.info(f'Total Samples: {results["total_samples"]}')
    logger.info(f'Original Recall@0.5: {results["original_recall"]:.4f}')
    logger.info(f'Attacked Recall@0.5: {results["attacked_recall"]:.4f}')
    outcomes = results['object_outcomes']
    counts = outcomes['counts']
    logger.info(
        'Object Failure ASR: %.4f; Pure Hiding ASR: %.4f (%d/%d)',
        results['object_failure_asr'],
        results['pure_hiding_asr'],
        counts['pure_hiding'],
        outcomes['eligible_clean_objects'],
    )
    logger.info(
        'Object outcomes: still correct=%d, pure hiding=%d, '
        'misclassification=%d, localization failure=%d',
        counts['still_correct'], counts['pure_hiding'],
        counts['misclassification'], counts['localization_failure'],
    )
    for class_name, class_result in outcomes['by_class'].items():
        class_counts = class_result['counts']
        logger.info(
            '%s outcomes: eligible=%d, hiding=%d (%.4f), miscls=%d, loc=%d, correct=%d',
            class_name,
            class_result['eligible_clean_objects'],
            class_counts['pure_hiding'],
            class_result['rates']['pure_hiding_asr'],
            class_counts['misclassification'],
            class_counts['localization_failure'],
            class_counts['still_correct'],
        )
    endpoint = results['object_endpoint_metrics']
    logger.info(
        'Object endpoint mean IoU drop / decreased fraction: %.6f / %.4f',
        endpoint['max_iou_drop']['mean'] or 0.0,
        endpoint['max_iou_drop']['positive_fraction'] or 0.0,
    )
    logger.info(
        'Object endpoint mean matched-score drop / decreased fraction: '
        '%.6f / %.4f',
        endpoint['match_score_drop']['mean'] or 0.0,
        endpoint['match_score_drop']['positive_fraction'] or 0.0,
    )
    if endpoint['object_evidence_drop']['count']:
        logger.info(
            'Object endpoint mean evidence drop / decreased fraction: '
            '%.6f / %.4f',
            endpoint['object_evidence_drop']['mean'],
            endpoint['object_evidence_drop']['positive_fraction'],
        )
    logger.info('Object endpoint CSV: %s', results['object_endpoint_file'])
    logger.info(f'Recall Drop: {results["recall_drop"]:.4f}')
    if 'vod_official' in results:
        vod_results = results['vod_official']
        for area in ('entire_area', 'roi'):
            clean_map = vod_results['clean'][area]['3d']['mAP']
            adversarial_map = vod_results['adversarial'][area]['3d']['mAP']
            map_drop = vod_results['absolute_drop'][area]['3d']['mAP']
            logger.info(
                'VoD %s 3D mAP clean/adv/drop: %.4f / %.4f / %.4f',
                area,
                clean_map,
                adversarial_map,
                map_drop,
            )
        if not vod_results['full_validation_set']:
            logger.warning(
                'VoD AP used a %d-frame subset and is not directly comparable '
                'with full validation-set results',
                vod_results['frames'],
            )
    logger.info('=' * 70)
    return results


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            18888, args.local_rank, backend='nccl'
        )
        dist_test = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_attack_%s_%s.txt' % (args.attack_type, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start 4D Radar Attack Logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    selected_indices = select_sample_indices(
        len(test_set), args.num_samples, args.sample_strategy, args.seed
    )
    if args.num_samples is not None:
        test_loader = build_subset_dataloader(test_set, args, selected_indices)
        args.sample_indices = selected_indices.tolist()
        logger.info(
            'Selected %d validation frames with strategy=%s and seed=%d',
            len(selected_indices),
            args.sample_strategy,
            args.seed,
        )
        logger.info('Validation sample indices: %s', args.sample_indices)
    else:
        args.sample_indices = None

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger)
    model.cuda()

    logger.info('Starting attack evaluation on 4D radar data...')
    results = evaluate_attack(model, test_loader, args, logger, output_dir)
    result_payload = {
        'attack': vars(args),
        'metrics': results,
    }
    with open(output_dir / 'attack_results.json', 'w') as result_file:
        json.dump(result_payload, result_file, indent=2)

    diagnostics = results.get('attack_diagnostics', {})
    with open(output_dir / 'attack_results.txt', 'w') as f:
        f.write('4D Radar Attack Results\n')
        f.write('=' * 40 + '\n')
        f.write(f'Attack Domain: {args.attack_domain}\n')
        f.write(f'Attack Type: {args.attack_type}\n')
        if args.attack_type == 'iou_s_original':
            f.write('Default Epsilon: unused (no hard projection)\n')
        else:
            f.write(f'Default Epsilon: {args.epsilon}\n')
        f.write(f'Attack Feature: {args.attack_feature}\n')
        f.write(f'Attack Space: {args.attack_space}\n')
        if args.attack_domain == 'point' and args.attack_type not in {
            'iadv', 'iou_s_original'
        }:
            f.write(f'Attack Loss: {args.attack_loss}\n')
            f.write(
                f'Point Target Selection: {args.point_target_selection}\n'
            )
        if args.attack_type == 'pgd':
            f.write(f'PGD Steps: {args.pgd_steps}\n')
        elif args.attack_type == 'iadv':
            f.write(f'I-ADV Steps: {args.iadv_steps}\n')
            f.write(f'I-ADV Scope: {args.iadv_scope}\n')
            f.write(f'I-ADV Neighbor Scope: {args.iadv_neighbor_scope}\n')
            f.write(
                'I-ADV Target Classes: '
                f'{", ".join(results["object_outcomes"]["target_classes"])}\n'
            )
        elif args.attack_type == 'iou_s_original':
            f.write(
                f'Original IoU-S Adam Steps: {args.iou_s_original_steps}\n'
            )
            f.write(f'Original IoU-S Adam LR: {args.iou_s_original_lr}\n')
            f.write(
                'Original IoU-S Init Noise: '
                f'{args.iou_s_original_init_noise}\n'
            )
            f.write(
                'Original IoU-S Distance Weight: '
                f'{args.iou_s_original_distance_weight}\n'
            )
            f.write('Original IoU-S Point Scope: full scene\n')
            f.write('Original IoU-S Hard Epsilon Projection: disabled\n')
            f.write(
                'Original IoU-S Return Policy: '
                f'{args.iou_s_original_return_policy}\n'
            )
        if args.attack_domain == 'point':
            for group in ('xyz', 'rcs', 'doppler', 'time'):
                value = getattr(args, f'epsilon_{group}')
                if value is not None:
                    f.write(f'Epsilon {group}: {value}\n')
            f.write(f'Voxel Mode: {args.voxel_mode}\n')
            if args.attack_space == 'radar_measurement':
                f.write(f'Temporal Mode: {args.temporal_mode}\n')
                f.write(
                    'Measurement Sweep: '
                    + {
                        'none': 'current (time == 0)\n',
                        'point_independent': (
                            'all available, point-independent\n'
                        ),
                        'object_per_sweep': (
                            'all available, object-shared within each sweep\n'
                        ),
                        'track_shared': 'all available, track-shared\n',
                    }[args.temporal_mode]
                )
                f.write(f'Range Epsilon [m]: {args.epsilon_range}\n')
                f.write(
                    'Azimuth / Elevation Epsilon [deg]: '
                    f'{args.epsilon_azimuth_deg} / '
                    f'{args.epsilon_elevation_deg}\n'
                )
                f.write(
                    'Temporal parameter groups (all / active): '
                    f'{int(diagnostics.get("temporal_parameter_groups", 0.0))} / '
                    f'{int(diagnostics.get("temporal_active_parameter_groups", 0.0))}\n'
                )
        f.write(f'Total Samples: {results["total_samples"]}\n')
        f.write(f'Original Recall@0.5: {results["original_recall"]:.4f}\n')
        f.write(f'Attacked Recall@0.5: {results["attacked_recall"]:.4f}\n')
        outcomes = results['object_outcomes']
        counts = outcomes['counts']
        f.write(
            f'Object Failure ASR: {results["object_failure_asr"]:.4f}\n'
        )
        f.write(
            f'Pure Hiding ASR: {results["pure_hiding_asr"]:.4f} '
            f'({counts["pure_hiding"]}/{outcomes["eligible_clean_objects"]})\n'
        )
        f.write(
            'Object outcomes (correct/hiding/misclassification/localization): '
            f'{counts["still_correct"]}/{counts["pure_hiding"]}/'
            f'{counts["misclassification"]}/{counts["localization_failure"]}\n'
        )
        for class_name, class_result in outcomes['by_class'].items():
            class_counts = class_result['counts']
            f.write(
                f'{class_name}: eligible={class_result["eligible_clean_objects"]}, '
                f'hiding={class_counts["pure_hiding"]} '
                f'({class_result["rates"]["pure_hiding_asr"]:.4f}), '
                f'misclassification={class_counts["misclassification"]}, '
                f'localization={class_counts["localization_failure"]}, '
                f'correct={class_counts["still_correct"]}\n'
            )
        endpoint = results['object_endpoint_metrics']
        f.write(
            'Mean max-IoU drop: '
            f'{endpoint["max_iou_drop"]["mean"] or 0.0:.6f}\n'
        )
        f.write(
            'IoU-decreased object fraction: '
            f'{endpoint["max_iou_drop"]["positive_fraction"] or 0.0:.6f}\n'
        )
        f.write(
            'Mean matched-score drop: '
            f'{endpoint["match_score_drop"]["mean"] or 0.0:.6f}\n'
        )
        f.write(
            'Score-decreased object fraction: '
            f'{endpoint["match_score_drop"]["positive_fraction"] or 0.0:.6f}\n'
        )
        if endpoint['object_evidence_drop']['count']:
            f.write(
                'Mean object-evidence drop: '
                f'{endpoint["object_evidence_drop"]["mean"]:.6f}\n'
            )
            f.write(
                'Evidence-decreased object fraction: '
                f'{endpoint["object_evidence_drop"]["positive_fraction"]:.6f}\n'
            )
        f.write(f'Object endpoint CSV: {results["object_endpoint_file"]}\n')
        f.write(f'Recall Drop: {results["recall_drop"]:.4f}\n')
        if args.attack_domain == 'point':
            f.write(f'Max |delta|: {results["max_abs_perturbation"]:.6f}\n')
            f.write(f'Mean |delta|: {results["mean_abs_perturbation"]:.6f}\n')
            diagnostics = results.get('attack_diagnostics', {})
            if args.attack_space == 'radar_measurement':
                f.write(
                    'Measurement max delta range [m]: '
                    f'{diagnostics.get("measurement_max_abs_delta_range", 0.0):.6f}\n'
                )
                f.write(
                    'Measurement max delta azimuth / elevation [deg]: '
                    f'{math.degrees(diagnostics.get("measurement_max_abs_delta_azimuth_rad", 0.0)):.6f} / '
                    f'{math.degrees(diagnostics.get("measurement_max_abs_delta_elevation_rad", 0.0)):.6f}\n'
                )
                f.write(
                    'Measurement Cartesian max / mean L2 [m]: '
                    f'{diagnostics.get("measurement_max_xyz_l2", 0.0):.6f} / '
                    f'{diagnostics.get("measurement_mean_xyz_l2", 0.0):.6f}\n'
                )
                f.write(
                    'Measurement historical / non-target / non-geometry '
                    'modifications: '
                    f'{int(diagnostics.get("measurement_historical_modification_count", 0.0))} / '
                    f'{int(diagnostics.get("measurement_non_target_modification_count", 0.0))} / '
                    f'{int(diagnostics.get("measurement_non_geometry_modification_count", 0.0))}\n'
                )
            if 'object_evidence_targets' in diagnostics:
                f.write(
                    'Clean-detected attack targets: '
                    f'{int(diagnostics["object_evidence_targets"])}\n'
                )
                f.write(
                    'Mean candidates per target: '
                    f'{diagnostics["object_evidence_mean_candidates_per_target"]:.2f}\n'
                )
            if args.attack_loss == 'object_iou_s':
                f.write(
                    'Radar Object IoU-S targets: '
                    f'{int(diagnostics.get("object_iou_s_targets", 0.0))}\n'
                )
                f.write(
                    'Radar Object IoU-S mean candidates per target: '
                    f'{diagnostics.get("object_iou_s_mean_candidates_per_target", 0.0):.2f}\n'
                )
                f.write(
                    'Radar Object IoU-S score / IoU weights: '
                    f'{args.iou_s_score_weight} / {args.iou_s_iou_weight}\n'
                )
        if 'vod_official' in results:
            for area in ('entire_area', 'roi'):
                clean_map = results['vod_official']['clean'][area]['3d']['mAP']
                adversarial_map = (
                    results['vod_official']['adversarial'][area]['3d']['mAP']
                )
                map_drop = (
                    results['vod_official']['absolute_drop'][area]['3d']['mAP']
                )
                f.write(
                    f'VoD {area} 3D mAP clean: {clean_map:.4f}\n'
                )
                f.write(
                    f'VoD {area} 3D mAP adversarial: '
                    f'{adversarial_map:.4f}\n'
                )
                f.write(
                    f'VoD {area} 3D mAP drop: {map_drop:.4f}\n'
                )

    logger.info('Attack evaluation finished. Results saved to %s' % output_dir)


if __name__ == '__main__':
    main()
