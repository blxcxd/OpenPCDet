"""OpenPCDet experiment runner for 4D-radar adversarial attacks."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import datetime
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
from radar_attack.attacks.objective import ObjectEvidenceObjective
from radar_attack.attacks.voxel import voxel_attack
from radar_attack.evaluation import (
    AdversarialPointCloudWriter,
    DetectionAttackMetrics,
    compare_target_object_endpoints,
    evaluate_vod_pair,
    prepare_prediction_directory,
    resolve_vod_label_dir,
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
    parser.add_argument('--attack_type', type=str, default='fgsm',
                        choices=['fgsm', 'pgd', 'iadv'],
                        help='attack type: fgsm, pgd, or I-ADV-RCS')
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
    parser.add_argument('--random_start', action='store_true',
                        help='use a random PGD start for point attack')
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
        choices=['training', 'object_evidence', 'object_hybrid'],
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
    if args.attack_loss in {'object_evidence', 'object_hybrid'}:
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

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def evaluate_attack(model, dataloader, args, logger, output_dir):
    model.eval()
    if (args.point_target_selection == 'clean_detected'
            and model.dense_head.__class__.__name__ != 'AnchorHeadSingle'):
        raise NotImplementedError(
            'clean-detected target selection and object evidence currently '
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
                    'epsilon': args.epsilon,
                    'epsilon_xyz': args.epsilon_xyz,
                    'epsilon_rcs': args.epsilon_rcs,
                    'epsilon_doppler': args.epsilon_doppler,
                    'epsilon_time': args.epsilon_time,
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
            else:
                point_mask = None
                if args.point_target_selection == 'clean_detected':
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
                    )
                    point_mask = objective.point_mask(original_points)
                    clean_object_evidence = objective.evidence_by_target(model)
                elif args.point_scope == 'gt_boxes':
                    point_mask = build_iadv_attack_mask(
                        original_points,
                        batch_dict,
                        scope='gt_boxes',
                        target_class_ids=target_class_ids,
                        box_margin=args.point_box_margin,
                    )
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
                    loss_fn=(
                        objective
                        if args.attack_loss in {
                            'object_evidence', 'object_hybrid'
                        }
                        else None
                    ),
                    loss_stats=objective.stats if objective is not None else None,
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
    logger.info(f'Default Epsilon: {args.epsilon}')
    logger.info(f'Attack Feature: {args.attack_feature}')
    if args.attack_domain == 'point' and args.attack_type != 'iadv':
        logger.info(f'Attack Loss: {args.attack_loss}')
        logger.info(f'Point Target Selection: {args.point_target_selection}')
    if args.attack_type == 'pgd':
        logger.info(f'PGD Steps: {args.pgd_steps}')
    elif args.attack_type == 'iadv':
        logger.info(f'I-ADV Steps: {args.iadv_steps}')
        logger.info(f'I-ADV Scope: {args.iadv_scope}')
        logger.info(f'I-ADV Neighbor Scope: {args.iadv_neighbor_scope}')
        logger.info('I-ADV Target Classes: %s', ', '.join(target_class_names))
    if args.attack_domain == 'point':
        for group in ('xyz', 'rcs', 'doppler', 'time'):
            value = getattr(args, f'epsilon_{group}')
            if value is not None:
                logger.info('Epsilon %s: %s', group, value)
        logger.info(f'Voxel Mode: {args.voxel_mode}')
        logger.info(f'Max |delta|: {results["max_abs_perturbation"]:.6f}')
        logger.info(f'Mean |delta|: {results["mean_abs_perturbation"]:.6f}')
        if writer is not None:
            logger.info(f'Saved adversarial point clouds: {writer.saved_samples}')
            logger.info(f'Adversarial manifest: {writer.manifest_path}')
        diagnostics = results.get('attack_diagnostics', {})
        if 'object_evidence_targets' in diagnostics:
            logger.info(
                'Clean-detected attack targets: %d; mean candidates/target: %.1f',
                int(diagnostics['object_evidence_targets']),
                diagnostics['object_evidence_mean_candidates_per_target'],
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

    with open(output_dir / 'attack_results.txt', 'w') as f:
        f.write('4D Radar Attack Results\n')
        f.write('=' * 40 + '\n')
        f.write(f'Attack Domain: {args.attack_domain}\n')
        f.write(f'Attack Type: {args.attack_type}\n')
        f.write(f'Default Epsilon: {args.epsilon}\n')
        f.write(f'Attack Feature: {args.attack_feature}\n')
        if args.attack_domain == 'point' and args.attack_type != 'iadv':
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
        if args.attack_domain == 'point':
            for group in ('xyz', 'rcs', 'doppler', 'time'):
                value = getattr(args, f'epsilon_{group}')
                if value is not None:
                    f.write(f'Epsilon {group}: {value}\n')
            f.write(f'Voxel Mode: {args.voxel_mode}\n')
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
            if 'object_evidence_targets' in diagnostics:
                f.write(
                    'Clean-detected attack targets: '
                    f'{int(diagnostics["object_evidence_targets"])}\n'
                )
                f.write(
                    'Mean candidates per target: '
                    f'{diagnostics["object_evidence_mean_candidates_per_target"]:.2f}\n'
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
