"""Paper-faithful IoU-S point-perturbation attack for OpenPCDet.

This module ports the ``iou_per`` branch released by the IoU-S authors.  It
keeps the original attack structure deliberately separate from the
Radar-adapted object IoU-S objective:

* every scene point is eligible and only XYZ is optimized;
* predictions and their IoUs are recomputed after every revoxelized forward;
* Adam minimizes the score-plus-IoU loss;
* bidirectional Chamfer distance and a global XYZ L2 term penalize distortion;
* there is no per-point epsilon projection.

OpenPCDet hard voxel assignment remains discrete.  Gradients pass through the
selected point gather while the topology is rebuilt on each step, matching the
BPDA convention used by the other raw-point attacks in this repository.
"""

from __future__ import annotations

from typing import Dict

import torch

from .base import AttackOutput, restore_attack_modes
from .objective import differentiable_oriented_iou3d


def directed_chamfer_squared(
    source: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Mean squared nearest-neighbour distance from ``source`` to ``target``."""
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('source must have shape [N, 3]')
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError('target must have shape [M, 3]')
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError('Chamfer distance requires non-empty point sets')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')

    nearest = []
    for start in range(0, source.shape[0], chunk_size):
        distances = torch.cdist(
            source[start:start + chunk_size], target, p=2
        ).square()
        nearest.append(distances.min(dim=1).values)
    return torch.cat(nearest).mean()


def symmetric_chamfer_squared(
    first: torch.Tensor,
    second: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """The two directed Chamfer terms used by the official IoU-S code."""
    return (
        directed_chamfer_squared(first, second, chunk_size)
        + directed_chamfer_squared(second, first, chunk_size)
    )


def original_iou_s_detection_loss(
    predictions: Dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    log_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Return the original IoU-S minimization loss for one scene.

    The released implementation pairs every valid GT with every current
    post-NMS prediction, sorts the pairs by IoU, and sums
    ``-log(1-score) - log(1-IoU)``.  Sorting does not change that sum, so the
    equivalent Cartesian product is used here.  Predicted class labels are not
    used, matching the released ``iou_per`` branch.
    """
    if not 0 < log_epsilon < 1:
        raise ValueError('log_epsilon must be in (0, 1)')
    required = {'pred_boxes', 'pred_scores', 'pred_labels'}
    missing = required - set(predictions)
    if missing:
        raise KeyError(f'predictions lack required keys: {sorted(missing)}')

    valid_gt = gt_boxes[(gt_boxes[:, 3:6] > 0).all(dim=1), :7]
    predicted_boxes = predictions['pred_boxes'][:, :7]
    predicted_scores = predictions['pred_scores']
    if predicted_boxes.shape[0] != predicted_scores.numel():
        raise ValueError('prediction boxes and scores have different lengths')
    pair_count = int(valid_gt.shape[0] * predicted_boxes.shape[0])
    if pair_count == 0:
        return predicted_boxes.sum() * 0.0 + predicted_scores.sum() * 0.0, 0

    gt_count = valid_gt.shape[0]
    prediction_count = predicted_boxes.shape[0]
    paired_predictions = predicted_boxes.unsqueeze(0).expand(
        gt_count, prediction_count, 7
    ).reshape(-1, 7)
    paired_targets = valid_gt.unsqueeze(1).expand(
        gt_count, prediction_count, 7
    ).reshape(-1, 7)
    paired_scores = predicted_scores.unsqueeze(0).expand(
        gt_count, prediction_count
    ).reshape(-1)
    ious = differentiable_oriented_iou3d(
        paired_predictions, paired_targets
    )
    loss = -(
        torch.log(1.0 - paired_scores + float(log_epsilon))
        + torch.log(1.0 - ious + float(log_epsilon))
    ).sum()
    return loss, pair_count


def original_iou_s_point_attack(
    model: torch.nn.Module,
    batch_dict: Dict,
    voxelizer,
    steps: int = 500,
    learning_rate: float = 0.01,
    initial_noise: float = 0.01,
    distance_weight: float = 1.0,
    log_epsilon: float = 1e-8,
    chamfer_chunk_size: int = 1024,
) -> AttackOutput:
    """Port the official IoU-S full-scene XYZ perturbation attack.

    The reference implementation attacks one validation scene at a time, so
    this port intentionally requires ``batch_size == 1``.  This avoids changing
    the paper's sum reduction or coupling independent scenes through one Adam
    objective.
    """
    if int(batch_dict.get('batch_size', 0)) != 1:
        raise ValueError('original IoU-S requires batch_size == 1')
    if steps <= 0:
        raise ValueError('steps must be positive')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    if initial_noise < 0:
        raise ValueError('initial_noise must be non-negative')
    if distance_weight < 0:
        raise ValueError('distance_weight must be non-negative')
    if not 0 < log_epsilon < 1:
        raise ValueError('log_epsilon must be in (0, 1)')
    if chamfer_chunk_size <= 0:
        raise ValueError('chamfer_chunk_size must be positive')

    original = batch_dict['points'].detach().clone()
    if original.ndim != 2 or original.shape[1] < 4:
        raise ValueError('points must have batch index and XYZ columns')
    if original.shape[0] == 0:
        raise ValueError('original IoU-S requires a non-empty point cloud')
    if not torch.equal(original[:, 0], torch.zeros_like(original[:, 0])):
        raise ValueError('batch_size=1 points must all have batch index zero')

    xyz = original[:, 1:4].detach().clone()
    if initial_noise > 0:
        # The official implementation uses a positive uniform 1e-2 start.
        xyz.add_(torch.rand_like(xyz) * float(initial_noise))
    xyz.requires_grad_(True)
    optimizer = torch.optim.Adam([xyz], lr=float(learning_rate), weight_decay=0.0)

    model_states = {module: module.training for module in model.modules()}
    model.eval()
    best_points = original.clone()
    best_distance = float('inf')
    best_total = float('inf')
    best_attack = float('inf')
    last_distance = 0.0
    last_attack = 0.0
    prediction_total = 0
    pair_total = 0
    nonfinite_steps = 0

    try:
        for _ in range(steps):
            adversarial = original.clone()
            adversarial[:, 1:4] = xyz
            topology = voxelizer.topology(adversarial)
            voxel_data = voxelizer.materialize(adversarial, topology)
            attack_batch = dict(batch_dict)
            attack_batch['points'] = adversarial
            attack_batch.update(voxel_data)

            predictions, _ = model(attack_batch)
            if len(predictions) != 1:
                raise RuntimeError('original IoU-S expected one prediction scene')
            attack_loss, pair_count = original_iou_s_detection_loss(
                predictions[0], batch_dict['gt_boxes'][0], log_epsilon
            )
            chamfer = symmetric_chamfer_squared(
                xyz, original[:, 1:4], chunk_size=chamfer_chunk_size
            )
            global_l2 = torch.sqrt(
                (xyz - original[:, 1:4]).square().sum() + 1e-4
            )
            distance_loss = chamfer + global_l2
            total_loss = attack_loss + float(distance_weight) * distance_loss
            gradient = torch.autograd.grad(total_loss, xyz)[0]
            if not torch.isfinite(total_loss) or not torch.isfinite(gradient).all():
                nonfinite_steps += 1
                raise RuntimeError('non-finite gradient during original IoU-S attack')

            optimizer.zero_grad(set_to_none=True)
            xyz.grad = gradient
            optimizer.step()

            distance_value = float(distance_loss.detach().item())
            attack_value = float(attack_loss.detach().item())
            total_value = float(total_loss.detach().item())
            prediction_total += int(predictions[0]['pred_boxes'].shape[0])
            pair_total += pair_count
            last_distance = distance_value
            last_attack = attack_value
            # Preserve the reference implementation's joint improvement rule.
            if distance_value < best_distance and total_value < best_total:
                best_distance = distance_value
                best_total = total_value
                best_attack = attack_value
                best_points = original.clone()
                best_points[:, 1:4] = xyz.detach()
    finally:
        restore_attack_modes(model_states)
        model.zero_grad(set_to_none=True)

    adversarial = best_points.detach()
    final_topology = voxelizer.topology(adversarial)
    voxel_data = voxelizer.materialize(adversarial, final_topology)
    delta_xyz = adversarial[:, 1:4] - original[:, 1:4]
    changed_points = (delta_xyz != 0).any(dim=1)
    non_xyz_changes = (adversarial[:, 4:] != original[:, 4:]).sum()
    batch_index_changes = (adversarial[:, 0] != original[:, 0]).sum()
    if non_xyz_changes or batch_index_changes:
        raise RuntimeError('original IoU-S modified a non-XYZ feature')
    xyz_l2 = torch.linalg.vector_norm(delta_xyz, dim=1)
    absolute_delta = delta_xyz.abs()
    stats = {
        'max_abs_perturbation': float(absolute_delta.max().item()),
        'mean_abs_perturbation': float(absolute_delta.mean().item()),
        'sum_abs_perturbation': float(absolute_delta.sum().item()),
        'perturbation_values': float(absolute_delta.numel()),
        'point_attack_mask_points': float(original.shape[0]),
        'point_modified_points': float(changed_points.sum().item()),
        'point_non_mask_modification_count': 0.0,
        'point_non_selected_feature_modification_count': 0.0,
        'point_max_xyz_l2': float(xyz_l2.max().item()),
        'point_xyz_l2_sum': float(xyz_l2.sum().item()),
        'point_xyz_l2_count': float(original.shape[0]),
        'iou_s_original_steps': float(steps),
        'iou_s_original_prediction_total': float(prediction_total),
        'iou_s_original_pair_total': float(pair_total),
        'iou_s_original_mean_predictions_per_step': prediction_total / steps,
        'iou_s_original_mean_pairs_per_step': pair_total / steps,
        'iou_s_original_best_attack_loss': best_attack,
        'iou_s_original_best_distance_loss': best_distance,
        'iou_s_original_best_total_loss': best_total,
        'iou_s_original_last_attack_loss': last_attack,
        'iou_s_original_last_distance_loss': last_distance,
        'iou_s_original_nonfinite_gradient_steps': float(nonfinite_steps),
        'iou_s_original_hard_epsilon_projection': 0.0,
    }
    return AttackOutput(
        adv_points=adversarial,
        model_inputs=voxel_data,
        stats=stats,
    )
