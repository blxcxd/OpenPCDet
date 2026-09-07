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


def prediction_set_retention(
    previous: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """Fraction of the larger post-NMS set retained across two iterations.

    A prediction is retained when it has a same-class box in the other set
    with oriented 3D IoU at least ``iou_threshold``.  Dividing the number of
    one-to-one greedy matches by the larger set size makes both disappearing
    and newly appearing boxes reduce the score.  This is a detached diagnostic
    only and never contributes to the attack gradient.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError('iou_threshold must be in [0, 1]')
    previous_boxes = previous['pred_boxes'][:, :7].detach()
    current_boxes = current['pred_boxes'][:, :7].detach()
    previous_labels = previous['pred_labels'].detach().long()
    current_labels = current['pred_labels'].detach().long()
    previous_count = int(previous_boxes.shape[0])
    current_count = int(current_boxes.shape[0])
    denominator = max(previous_count, current_count)
    if denominator == 0:
        return 1.0
    if previous_count == 0 or current_count == 0:
        return 0.0

    paired_previous = previous_boxes[:, None, :].expand(
        previous_count, current_count, 7
    ).reshape(-1, 7)
    paired_current = current_boxes[None, :, :].expand(
        previous_count, current_count, 7
    ).reshape(-1, 7)
    pair_ious = differentiable_oriented_iou3d(
        paired_previous, paired_current
    ).reshape(previous_count, current_count)
    same_class = previous_labels[:, None] == current_labels[None, :]
    pair_ious = torch.where(same_class, pair_ious, pair_ious.new_zeros(()))

    matches = 0
    available_previous = torch.ones(
        previous_count, dtype=torch.bool, device=pair_ious.device
    )
    available_current = torch.ones(
        current_count, dtype=torch.bool, device=pair_ious.device
    )
    # The post-NMS sets are small.  A detached greedy assignment is enough for
    # a stable diagnostic and avoids adding a SciPy dependency to the attack.
    for _ in range(min(previous_count, current_count)):
        available = available_previous[:, None] & available_current[None, :]
        candidates = torch.where(
            available, pair_ious, pair_ious.new_full((), -1.0)
        )
        best_value, flat_index = candidates.reshape(-1).max(dim=0)
        if float(best_value.item()) < float(iou_threshold):
            break
        previous_index = int(flat_index.item()) // current_count
        current_index = int(flat_index.item()) % current_count
        available_previous[previous_index] = False
        available_current[current_index] = False
        matches += 1
    return matches / denominator


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
    return_policy: str = 'joint_best',
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
    if return_policy not in {'joint_best', 'last'}:
        raise ValueError('return_policy must be joint_best or last')

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
    best_step = 0
    best_updates = 0
    first_distance = 0.0
    first_attack = 0.0
    first_total = 0.0
    prediction_total = 0
    pair_total = 0
    nonfinite_steps = 0
    zero_prediction_steps = 0
    transition_count = 0
    retention_sum = 0.0
    switch_count = 0
    prediction_count_change_count = 0
    previous_predictions = None

    try:
        for step_index in range(steps):
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
            current_predictions = {
                'pred_boxes': predictions[0]['pred_boxes'].detach(),
                'pred_labels': predictions[0]['pred_labels'].detach(),
            }
            if previous_predictions is not None:
                retention = prediction_set_retention(
                    previous_predictions, current_predictions
                )
                retention_sum += retention
                transition_count += 1
                if retention < 1.0:
                    switch_count += 1
                if (previous_predictions['pred_boxes'].shape[0]
                        != current_predictions['pred_boxes'].shape[0]):
                    prediction_count_change_count += 1
            previous_predictions = current_predictions
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
            if predictions[0]['pred_boxes'].shape[0] == 0:
                zero_prediction_steps += 1
            if step_index == 0:
                first_distance = distance_value
                first_attack = attack_value
                first_total = total_value
            # Preserve the reference implementation's joint improvement rule.
            if distance_value < best_distance and total_value < best_total:
                best_distance = distance_value
                best_total = total_value
                best_attack = attack_value
                best_step = step_index + 1
                best_updates += 1
                best_points = original.clone()
                best_points[:, 1:4] = xyz.detach()

        last_points = original.clone()
        last_points[:, 1:4] = xyz.detach()

        def endpoint_losses(points: torch.Tensor) -> tuple[float, float, float, int]:
            endpoint_topology = voxelizer.topology(points)
            endpoint_voxels = voxelizer.materialize(points, endpoint_topology)
            endpoint_batch = dict(batch_dict)
            endpoint_batch['points'] = points
            endpoint_batch.update(endpoint_voxels)
            with torch.no_grad():
                endpoint_predictions, _ = model(endpoint_batch)
                endpoint_attack, endpoint_pairs = original_iou_s_detection_loss(
                    endpoint_predictions[0], batch_dict['gt_boxes'][0], log_epsilon
                )
                endpoint_chamfer = symmetric_chamfer_squared(
                    points[:, 1:4], original[:, 1:4],
                    chunk_size=chamfer_chunk_size,
                )
                endpoint_global_l2 = torch.sqrt(
                    (points[:, 1:4] - original[:, 1:4]).square().sum()
                    + 1e-4
                )
                endpoint_distance = endpoint_chamfer + endpoint_global_l2
                endpoint_total = (
                    endpoint_attack
                    + float(distance_weight) * endpoint_distance
                )
            return (
                float(endpoint_attack.item()),
                float(endpoint_distance.item()),
                float(endpoint_total.item()),
                int(endpoint_pairs),
            )

        joint_best_endpoint = endpoint_losses(best_points)
        last_endpoint = endpoint_losses(last_points)
    finally:
        restore_attack_modes(model_states)
        model.zero_grad(set_to_none=True)

    selected_endpoint = (
        joint_best_endpoint if return_policy == 'joint_best' else last_endpoint
    )
    adversarial = (
        best_points if return_policy == 'joint_best' else last_points
    ).detach()
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
        'iou_s_original_best_step': float(best_step),
        'iou_s_original_best_step_fraction': best_step / steps,
        'iou_s_original_best_updates': float(best_updates),
        'iou_s_original_best_step_le_10': float(best_step <= 10),
        'iou_s_original_best_step_le_100': float(best_step <= 100),
        'iou_s_original_first_attack_loss': first_attack,
        'iou_s_original_first_distance_loss': first_distance,
        'iou_s_original_first_total_loss': first_total,
        'iou_s_original_joint_best_endpoint_attack_loss': joint_best_endpoint[0],
        'iou_s_original_joint_best_endpoint_distance_loss': joint_best_endpoint[1],
        'iou_s_original_joint_best_endpoint_total_loss': joint_best_endpoint[2],
        'iou_s_original_last_attack_loss': last_endpoint[0],
        'iou_s_original_last_distance_loss': last_endpoint[1],
        'iou_s_original_last_total_loss': last_endpoint[2],
        'iou_s_original_selected_attack_loss': selected_endpoint[0],
        'iou_s_original_selected_distance_loss': selected_endpoint[1],
        'iou_s_original_selected_total_loss': selected_endpoint[2],
        'iou_s_original_joint_best_endpoint_attack_loss_per_pair': (
            joint_best_endpoint[0] / max(joint_best_endpoint[3], 1)
        ),
        'iou_s_original_last_attack_loss_per_pair': (
            last_endpoint[0] / max(last_endpoint[3], 1)
        ),
        'iou_s_original_mean_prediction_retention': (
            retention_sum / max(transition_count, 1)
        ),
        'iou_s_original_prediction_switch_fraction': (
            switch_count / max(transition_count, 1)
        ),
        'iou_s_original_prediction_count_change_fraction': (
            prediction_count_change_count / max(transition_count, 1)
        ),
        'iou_s_original_zero_prediction_step_fraction': (
            zero_prediction_steps / steps
        ),
        'iou_s_original_return_last': float(return_policy == 'last'),
        'iou_s_original_nonfinite_gradient_steps': float(nonfinite_steps),
        'iou_s_original_hard_epsilon_projection': 0.0,
    }
    return AttackOutput(
        adv_points=adversarial,
        model_inputs=voxel_data,
        stats=stats,
    )
