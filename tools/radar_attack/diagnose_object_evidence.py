"""Diagnose object-level losses and feature sensitivities on 4D radar."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) in sys.path:
    sys.path.remove(str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils

from radar_attack.adapters.openpcdet import (
    PointCloudVoxelizer,
    get_feature_names,
    get_voxel_settings,
)
from radar_attack.analysis import (
    candidate_anchor_indices,
    domain_sensitivities,
    feature_scales_from_statistics,
    first_order_evidence_drop,
    flatten_anchors,
    hybrid_gradient_relationships,
    normalized_probe_budget,
    object_evidence,
    reshape_anchor_cls_logits,
    sensitivity_allocations,
    summarize_records,
    write_diagnostic_report,
)
from radar_attack.attacks.base import restore_attack_modes, set_attack_mode
from radar_attack.attacks.gradient import project_points
from radar_attack.attacks.iadv import assign_points_to_oriented_boxes
from radar_attack.attacks.objective import ObjectEvidenceObjective
from radar_attack.runner import build_subset_dataloader, select_sample_indices


DEFAULT_IOU_THRESHOLDS = {
    'Car': 0.5,
    'Pedestrian': 0.25,
    'Cyclist': 0.25,
}


def _parse_iou_thresholds(values):
    thresholds = dict(DEFAULT_IOU_THRESHOLDS)
    for value in values or []:
        if '=' not in value:
            raise argparse.ArgumentTypeError(
                f'IoU threshold {value!r} must use CLASS=VALUE'
            )
        name, threshold = value.split('=', 1)
        threshold = float(threshold)
        if not 0 <= threshold <= 1:
            raise argparse.ArgumentTypeError('IoU thresholds must be in [0, 1]')
        thresholds[name] = threshold
    return thresholds


def parse_config():
    parser = argparse.ArgumentParser(
        description=(
            'Compare detector training-loss and object-evidence gradients, '
            'then report IQR-normalized radar feature sensitivities'
        )
    )
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--feature_stats', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument(
        '--sample_strategy', choices=['first', 'uniform', 'random'], default='uniform'
    )
    parser.add_argument('--target_classes', nargs='+', default=None)
    parser.add_argument(
        '--iou_thresholds', nargs='*', default=None, metavar='CLASS=VALUE'
    )
    parser.add_argument('--candidate_box_margin', type=float, default=1.0)
    parser.add_argument('--candidate_topk', type=int, default=32)
    parser.add_argument(
        '--localization_topk', type=int, default=32,
        help='clean fixed decoded-IoU candidates for localization gradients',
    )
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--probe_fraction', type=float, default=0.01)
    parser.add_argument(
        '--probe_voxel_mode',
        choices=['fixed', 'revoxelize'],
        default='fixed',
        help=(
            'fixed keeps the gradient topology for the local loss comparison; '
            'revoxelize additionally includes discrete pillar changes'
        ),
    )
    parser.add_argument(
        '--probe_geometry_cap', type=float, default=0.02,
        help='maximum per-coordinate XYZ diagnostic step in metres',
    )
    parser.add_argument(
        '--probe_doppler_cap', type=float, default=0.1,
        help='maximum per-feature Doppler diagnostic step in m/s',
    )
    parser.add_argument(
        '--probe_rcs_cap', type=float, default=0.2,
        help='maximum RCS diagnostic step',
    )
    parser.add_argument('--output_dir', default=None)
    parser.add_argument(
        '--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    if args.workers < 0:
        parser.error('--workers must be non-negative')
    if args.batch_size != 1:
        parser.error('the first object diagnostic requires --batch_size 1')
    if args.seed < 0:
        parser.error('--seed must be non-negative')
    if args.num_samples <= 0:
        parser.error('--num_samples must be positive')
    if args.candidate_box_margin < 0:
        parser.error('--candidate_box_margin must be non-negative')
    if args.candidate_topk <= 0:
        parser.error('--candidate_topk must be positive')
    if args.localization_topk <= 0:
        parser.error('--localization_topk must be positive')
    if args.temperature <= 0:
        parser.error('--temperature must be positive')
    if args.probe_fraction <= 0:
        parser.error('--probe_fraction must be positive')
    for name in ('probe_geometry_cap', 'probe_doppler_cap', 'probe_rcs_cap'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    try:
        args.iou_threshold_map = _parse_iou_thresholds(args.iou_thresholds)
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args, cfg


def _resolve_repo_path(value, must_exist=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cfg.ROOT_DIR / path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _active_point_mask(topology, point_count):
    mask = torch.zeros(
        point_count, dtype=torch.bool, device=topology.point_indices.device
    )
    mask[topology.point_indices[topology.point_mask]] = True
    return mask


def _valid_target_rows(gt_boxes, target_class_ids):
    valid = (gt_boxes[:, 3:6] > 0).all(dim=1)
    valid &= torch.isin(
        gt_boxes[:, -1].long(),
        torch.as_tensor(
            sorted(target_class_ids), device=gt_boxes.device, dtype=torch.long
        ),
    )
    return torch.nonzero(valid, as_tuple=False).flatten()


def _clean_detected_targets(
    prediction,
    gt_boxes,
    target_rows,
    class_names,
    thresholds,
):
    detected = []
    for gt_row in target_rows.tolist():
        class_id = int(gt_boxes[gt_row, -1].item())
        class_name = class_names[class_id - 1]
        prediction_mask = prediction['pred_labels'].long() == class_id
        predicted_boxes = prediction['pred_boxes'][prediction_mask, :7]
        predicted_scores = prediction['pred_scores'][prediction_mask]
        if predicted_boxes.shape[0] == 0:
            continue
        ious = iou3d_nms_utils.boxes_iou3d_gpu(
            predicted_boxes, gt_boxes[gt_row:gt_row + 1, :7]
        ).squeeze(1)
        best_iou, best_index = ious.max(dim=0)
        if float(best_iou.item()) < float(thresholds[class_name]):
            continue
        detected.append(
            {
                'gt_row': int(gt_row),
                'class_id': class_id,
                'class_name': class_name,
                'clean_detection_iou': float(best_iou.item()),
                'clean_detection_score': float(
                    predicted_scores[best_index].item()
                ),
            }
        )
    return detected


def _mean_feature(points, mask, feature_names, name, absolute=False):
    if name not in feature_names or not mask.any():
        return float('nan')
    column = feature_names.index(name) + 1
    values = points[mask, column]
    if absolute:
        values = values.abs()
    return float(values.mean().item())


def _forward_train_logits(model, batch_dict, points, voxelizer, topology):
    model_inputs = voxelizer.materialize(points, topology)
    model_batch = dict(batch_dict)
    model_batch['points'] = points
    model_batch.update(model_inputs)
    ret_dict, _, _ = model(model_batch)
    raw_logits = model.dense_head.forward_ret_dict['cls_preds']
    cls_logits = reshape_anchor_cls_logits(raw_logits, model.num_class)
    return ret_dict['loss'].mean(), cls_logits


def _probe_metrics(
    model,
    batch_dict,
    points,
    voxelizer,
    candidate_indices,
    class_index,
    temperature,
    topology=None,
):
    topology = voxelizer.topology(points) if topology is None else topology
    with torch.no_grad():
        _, cls_logits = _forward_train_logits(
            model, batch_dict, points, voxelizer, topology
        )
        target_logits = cls_logits[0, candidate_indices, class_index]
        evidence = object_evidence(
            cls_logits[0], candidate_indices, class_index, temperature
        )
        maximum_score = torch.sigmoid(target_logits).max()
    return float(evidence.item()), float(maximum_score.item())


def _probe_points(
    original,
    ascent_gradient,
    point_mask,
    probe_budget,
    voxelizer,
    voxel_mode,
):
    masked_budget = probe_budget.expand_as(original) * point_mask.to(
        original.dtype
    ).unsqueeze(1)
    candidate = original + ascent_gradient.sign() * masked_budget
    return project_points(
        candidate,
        original,
        masked_budget,
        voxelizer.point_cloud_range,
        voxelizer.voxel_size,
        voxel_mode=voxel_mode,
    ).detach()


def diagnose_frame(
    model,
    batch_dict,
    voxelizer,
    feature_names,
    feature_scales,
    target_class_ids,
    class_names,
    thresholds,
    anchors,
    args,
):
    if int(batch_dict['batch_size']) != 1:
        raise ValueError('object diagnosis currently requires batch size 1')
    original = batch_dict['points'].detach().clone()
    fixed_topology = voxelizer.topology(original)
    model_active = _active_point_mask(fixed_topology, original.shape[0])
    gt_boxes = batch_dict['gt_boxes'][0]
    target_rows = _valid_target_rows(gt_boxes, target_class_ids)

    model.eval()
    with torch.no_grad():
        clean_predictions, _ = model(dict(batch_dict))
    clean_targets = _clean_detected_targets(
        clean_predictions[0],
        gt_boxes,
        target_rows,
        class_names,
        thresholds,
    )
    if not clean_targets:
        return [], {
            'target_gt': int(target_rows.numel()),
            'clean_detected': 0,
            'attackable_clean_detected': 0,
        }

    selected_boxes = gt_boxes[target_rows, :7]
    local_assignments = assign_points_to_oriented_boxes(
        original[:, 1:4], selected_boxes
    )
    gt_row_to_object_id = {
        int(gt_row): local_id
        for local_id, gt_row in enumerate(target_rows.tolist())
    }
    attackable_targets = []
    for target in clean_targets:
        object_id = gt_row_to_object_id[target['gt_row']]
        point_mask = (local_assignments == object_id) & model_active
        if not point_mask.any():
            continue
        gt_box = gt_boxes[target['gt_row'], :7]
        candidates = candidate_anchor_indices(
            anchors,
            gt_box,
            margin=args.candidate_box_margin,
            fallback_topk=args.candidate_topk,
        )
        attackable_targets.append(
            {
                **target,
                'object_id': object_id,
                'point_mask': point_mask,
                'candidate_indices': candidates,
            }
        )
    counts = {
        'target_gt': int(target_rows.numel()),
        'clean_detected': len(clean_targets),
        'attackable_clean_detected': len(attackable_targets),
    }
    if not attackable_targets:
        return [], counts

    states = set_attack_mode(model)
    records = []
    try:
        differentiable_points = original.detach().clone().requires_grad_(True)
        training_loss, cls_logits = _forward_train_logits(
            model,
            batch_dict,
            differentiable_points,
            voxelizer,
            fixed_topology,
        )
        if cls_logits.shape[1] != anchors.shape[0]:
            raise RuntimeError(
                f'{cls_logits.shape[1]} logits do not match '
                f'{anchors.shape[0]} anchors'
            )

        hybrid_components = {}
        for class_id in sorted(target_class_ids):
            class_name = class_names[class_id - 1]
            class_objective = ObjectEvidenceObjective.from_clean_predictions(
                model=model,
                batch_dict=batch_dict,
                clean_predictions=clean_predictions,
                target_class_ids=[class_id],
                iou_threshold=thresholds[class_name],
                candidate_margin=args.candidate_box_margin,
                candidate_topk=args.candidate_topk,
                temperature=args.temperature,
                localization_weight=1.0,
                localization_topk=args.localization_topk,
            )
            for component in class_objective.component_objectives(model):
                hybrid_target, classification, localization = component
                hybrid_components[hybrid_target.gt_index] = (
                    hybrid_target,
                    classification,
                    localization,
                )

        for target in attackable_targets:
            if target['gt_row'] not in hybrid_components:
                raise RuntimeError(
                    'clean-detected target is missing from hybrid objective: '
                    f'GT row {target["gt_row"]}'
                )
            hybrid_target, classification, localization = hybrid_components[
                target['gt_row']
            ]
            target['hybrid_target'] = hybrid_target
            target['classification_objective'] = classification
            target['localization_objective'] = localization
            target['evidence_tensor'] = -classification
            target_logits = cls_logits[
                0, target['candidate_indices'], target['class_id'] - 1
            ]
            target['clean_max_candidate_score'] = float(
                torch.sigmoid(target_logits).max().detach().item()
            )

        training_gradient = torch.autograd.grad(
            training_loss, differentiable_points, retain_graph=True
        )[0].detach()
        probe_budget = normalized_probe_budget(
            original,
            feature_names,
            feature_scales,
            args.probe_fraction,
            domains=('geometry', 'doppler', 'rcs'),
            domain_caps={
                'geometry': args.probe_geometry_cap,
                'doppler': args.probe_doppler_cap,
                'rcs': args.probe_rcs_cap,
            },
        )
        budget_scales = {
            name: float(probe_budget[0, column + 1].item())
            for column, name in enumerate(feature_names)
        }

        for target_index, target in enumerate(attackable_targets):
            evidence = target['evidence_tensor']
            object_gradient = torch.autograd.grad(
                target['classification_objective'],
                differentiable_points,
                retain_graph=True,
            )[0].detach()
            localization_gradient = torch.autograd.grad(
                target['localization_objective'],
                differentiable_points,
                retain_graph=target_index < len(attackable_targets) - 1,
            )[0].detach()
            evidence_gradient = -object_gradient
            point_mask = target['point_mask']
            hybrid_relationships = hybrid_gradient_relationships(
                object_gradient,
                localization_gradient,
                point_mask,
                feature_names,
            )
            iqr_sensitivity = domain_sensitivities(
                object_gradient,
                point_mask,
                feature_names,
                feature_scales,
            )
            budget_sensitivity = domain_sensitivities(
                object_gradient,
                point_mask,
                feature_names,
                budget_scales,
            )
            training_sensitivity = domain_sensitivities(
                training_gradient,
                point_mask,
                feature_names,
                feature_scales,
            )
            iqr_allocations = sensitivity_allocations(iqr_sensitivity)
            budget_allocations = sensitivity_allocations(budget_sensitivity)

            training_predicted_drop = first_order_evidence_drop(
                evidence_gradient,
                training_gradient,
                point_mask,
                probe_budget,
            )
            object_predicted_drop = first_order_evidence_drop(
                evidence_gradient,
                object_gradient,
                point_mask,
                probe_budget,
            )
            training_probe = _probe_points(
                original,
                training_gradient,
                point_mask,
                probe_budget,
                voxelizer,
                args.probe_voxel_mode,
            )
            object_probe = _probe_points(
                original,
                object_gradient,
                point_mask,
                probe_budget,
                voxelizer,
                args.probe_voxel_mode,
            )
            training_evidence, training_max_score = _probe_metrics(
                model,
                batch_dict,
                training_probe,
                voxelizer,
                target['candidate_indices'],
                target['class_id'] - 1,
                args.temperature,
                fixed_topology if args.probe_voxel_mode == 'fixed' else None,
            )
            object_probe_evidence, object_max_score = _probe_metrics(
                model,
                batch_dict,
                object_probe,
                voxelizer,
                target['candidate_indices'],
                target['class_id'] - 1,
                args.temperature,
                fixed_topology if args.probe_voxel_mode == 'fixed' else None,
            )

            gt_box = gt_boxes[target['gt_row']]
            frame_id = str(batch_dict['frame_id'][0])
            clean_evidence = float(evidence.detach().item())
            row = {
                'frame_id': frame_id,
                'object_id': int(target['object_id']),
                'gt_row': int(target['gt_row']),
                'class_name': target['class_name'],
                'class_id': int(target['class_id']),
                'distance_xy': float(torch.linalg.vector_norm(gt_box[:2]).item()),
                'active_point_count': int(point_mask.sum().item()),
                'mean_rcs': _mean_feature(
                    original, point_mask, feature_names, 'rcs'
                ),
                'mean_abs_v_r': _mean_feature(
                    original, point_mask, feature_names, 'v_r', absolute=True
                ),
                'mean_abs_v_r_comp': _mean_feature(
                    original,
                    point_mask,
                    feature_names,
                    'v_r_comp',
                    absolute=True,
                ),
                'candidate_count': int(target['candidate_indices'].numel()),
                'clean_detection_iou': target['clean_detection_iou'],
                'clean_detection_score': target['clean_detection_score'],
                'clean_object_evidence': clean_evidence,
                'clean_max_candidate_score': target[
                    'clean_max_candidate_score'
                ],
                'classification_objective': float(
                    target['classification_objective'].detach().item()
                ),
                'localization_objective': float(
                    target['localization_objective'].detach().item()
                ),
                'localization_candidate_count': int(
                    target['hybrid_target'].localization_indices.numel()
                ),
                'training_loss_evidence_drop': clean_evidence - training_evidence,
                'object_loss_evidence_drop': (
                    clean_evidence - object_probe_evidence
                ),
                'training_loss_max_score_drop': (
                    target['clean_max_candidate_score'] - training_max_score
                ),
                'object_loss_max_score_drop': (
                    target['clean_max_candidate_score'] - object_max_score
                ),
                'training_loss_predicted_drop': training_predicted_drop,
                'object_loss_predicted_drop': object_predicted_drop,
                # Unprefixed fields preserve the original IQR-normalized
                # diagnostic schema for result traceability.
                **iqr_sensitivity,
                **{
                    f'training_{key}': value
                    for key, value in training_sensitivity.items()
                },
                **iqr_allocations,
                **{f'iqr_{key}': value for key, value in iqr_sensitivity.items()},
                **{f'iqr_{key}': value for key, value in iqr_allocations.items()},
                **{
                    f'budget_{key}': value
                    for key, value in budget_sensitivity.items()
                },
                **{
                    f'budget_{key}': value
                    for key, value in budget_allocations.items()
                },
                **hybrid_relationships,
            }
            if not all(
                math_is_finite(value)
                for value in row.values()
                if isinstance(value, float)
            ):
                raise RuntimeError(
                    f'non-finite diagnostic value for frame {frame_id} '
                    f'object {target["object_id"]}'
                )
            records.append(row)
    finally:
        restore_attack_modes(states)
        model.zero_grad(set_to_none=True)
    return records, counts


def math_is_finite(value):
    return bool(np.isfinite(value))


def main():
    os.chdir(TOOLS_DIR)
    args, dataset_cfg = parse_config()
    feature_statistics = _resolve_repo_path(args.feature_stats, must_exist=True)
    checkpoint = _resolve_repo_path(args.ckpt, must_exist=True)
    output_dir = (
        _resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else (
            cfg.ROOT_DIR
            / 'output'
            / 'radar_attack'
            / cfg.TAG
            / f'object_evidence_{args.num_samples}'
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(
        output_dir / 'diagnose_object_evidence.log', rank=0
    )
    log_config_to_file(cfg, logger=logger)

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=dataset_cfg.DATA_CONFIG,
        class_names=dataset_cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    target_classes = args.target_classes or list(dataset.class_names)
    unknown_classes = sorted(set(target_classes) - set(dataset.class_names))
    if unknown_classes:
        raise ValueError(f'unknown target classes: {unknown_classes}')
    missing_thresholds = sorted(
        set(target_classes) - set(args.iou_threshold_map)
    )
    if missing_thresholds:
        raise ValueError(
            f'missing IoU thresholds for classes: {missing_thresholds}; use '
            '--iou_thresholds CLASS=VALUE'
        )
    target_class_ids = {
        dataset.class_names.index(name) + 1 for name in target_classes
    }
    selected_indices = select_sample_indices(
        len(dataset), args.num_samples, args.sample_strategy, args.seed
    )
    dataloader = build_subset_dataloader(dataset, args, selected_indices)

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    if model.dense_head.__class__.__name__ != 'AnchorHeadSingle':
        raise NotImplementedError(
            'the first object diagnostic supports AnchorHeadSingle only; got '
            f'{model.dense_head.__class__.__name__}'
        )
    model.load_params_from_file(filename=str(checkpoint), logger=logger)
    model.cuda()
    model.eval()

    feature_names = get_feature_names(cfg.DATA_CONFIG)
    feature_scales = feature_scales_from_statistics(
        feature_statistics, feature_names
    )
    voxel_size, max_points, max_voxels = get_voxel_settings(cfg.DATA_CONFIG)
    voxelizer = PointCloudVoxelizer(
        cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        voxel_size,
        max_points,
        max_voxels,
        batch_size=1,
    )
    anchors = flatten_anchors(model.dense_head)

    records = []
    counts = {
        'frames': 0,
        'target_gt': 0,
        'clean_detected': 0,
        'attackable_clean_detected': 0,
    }
    progress = tqdm.tqdm(
        dataloader,
        total=len(selected_indices),
        desc='Object gradient diagnosis',
        dynamic_ncols=True,
    )
    for batch_dict in progress:
        load_data_to_gpu(batch_dict)
        frame_records, frame_counts = diagnose_frame(
            model=model,
            batch_dict=batch_dict,
            voxelizer=voxelizer,
            feature_names=feature_names,
            feature_scales=feature_scales,
            target_class_ids=target_class_ids,
            class_names=list(dataset.class_names),
            thresholds=args.iou_threshold_map,
            anchors=anchors,
            args=args,
        )
        records.extend(frame_records)
        counts['frames'] += 1
        for key in ('target_gt', 'clean_detected', 'attackable_clean_detected'):
            counts[key] += frame_counts[key]
        progress.set_postfix(objects=len(records))

    probe_budget = normalized_probe_budget(
        torch.zeros((1, len(feature_names) + 1), device='cuda'),
        feature_names,
        feature_scales,
        args.probe_fraction,
        domains=('geometry', 'doppler', 'rcs'),
        domain_caps={
            'geometry': args.probe_geometry_cap,
            'doppler': args.probe_doppler_cap,
            'rcs': args.probe_rcs_cap,
        },
    )
    summary = summarize_records(records)
    summary.update(
        {
            **counts,
            'unattackable_clean_detected': (
                counts['clean_detected']
                - counts['attackable_clean_detected']
            ),
            'config': str(Path(args.cfg_file).resolve()),
            'checkpoint': str(checkpoint),
            'feature_statistics': str(feature_statistics),
            'sample_strategy': args.sample_strategy,
            'sample_indices': selected_indices.tolist(),
            'target_classes': target_classes,
            'iou_thresholds': {
                name: args.iou_threshold_map[name] for name in target_classes
            },
            'candidate_box_margin': args.candidate_box_margin,
            'candidate_topk': args.candidate_topk,
            'localization_topk': args.localization_topk,
            'temperature': args.temperature,
            'probe_fraction': args.probe_fraction,
            'probe_voxel_mode': args.probe_voxel_mode,
            'probe_budgets': {
                name: float(probe_budget[0, column + 1].item())
                for column, name in enumerate(feature_names)
            },
            'sensitivity_scales': {
                'iqr': 'per-feature dataset IQR',
                'budget': 'exact capped local probe budget per feature',
            },
            'allocation_primary': (
                'budget_* fields use the exact capped local probe budget; '
                'unprefixed and iqr_* fields preserve the IQR-scale diagnostic'
            ),
            'geometry_semantics': (
                'Cartesian input-domain diagnostic; not a physical radar '
                'range/angle perturbation'
            ),
            'doppler_semantics': (
                'v_r and v_r_comp are probed as model input features; '
                'coupled sensitivity is reported separately'
            ),
        }
    )
    write_diagnostic_report(records, summary, output_dir)
    with (output_dir / 'run_config.json').open(
        'w', encoding='utf-8'
    ) as output_file:
        json.dump(vars(args), output_file, indent=2, ensure_ascii=False)

    logger.info('Diagnosed frames: %d', counts['frames'])
    logger.info('Target GT objects: %d', counts['target_gt'])
    logger.info('Clean detected targets: %d', counts['clean_detected'])
    logger.info(
        'Attackable clean detected targets: %d',
        counts['attackable_clean_detected'],
    )
    logger.info('Per-object CSV: %s', output_dir / 'per_object.csv')
    logger.info('Summary JSON: %s', output_dir / 'summary.json')


if __name__ == '__main__':
    main()
