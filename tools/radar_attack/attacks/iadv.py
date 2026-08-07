"""I-ADV intensity attack adapted to raw 4D-radar RCS points.

The implementation follows Algorithm 1 of I-ADV where the paper is explicit.
Parameters omitted by the paper (neighbour count, PCA fallback, gradient norm,
and target-point extraction) are exposed by the runner and recorded in output
metadata instead of being hidden as implementation details.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from .base import AttackOutput, restore_attack_modes, set_attack_mode


RCS_FEATURE_NAMES = {'rcs', 'intensity', 'power'}


def rcs_column(feature_names: Sequence[str]) -> int:
    """Return the RCS column in batched points, including batch_idx offset."""
    matches = [
        index + 1
        for index, name in enumerate(feature_names)
        if name.lower() in RCS_FEATURE_NAMES
    ]
    if len(matches) != 1:
        raise ValueError(
            'I-ADV requires exactly one RCS/intensity feature; got '
            f'{list(feature_names)}'
        )
    return matches[0]


def assign_points_to_oriented_boxes(
    points_xyz: torch.Tensor,
    boxes: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Assign each point to one yaw-oriented 3D box, or ``-1``.

    Boxes use OpenPCDet's ``[x, y, z, dx, dy, dz, heading, ...]`` layout.
    A point inside overlapping boxes is assigned to the box with the smallest
    squared center distance after normalizing each local axis by that box's
    (margin-expanded) half size.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError('points_xyz must have shape [N, 3]')
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError('boxes must have shape [M, >=7]')
    if margin < 0:
        raise ValueError('box margin must be non-negative')
    if boxes.shape[0] == 0:
        return torch.full(
            (points_xyz.shape[0],),
            -1,
            dtype=torch.long,
            device=points_xyz.device,
        )

    relative = points_xyz[:, None, :] - boxes[None, :, :3]
    heading = boxes[:, 6]
    cosine = torch.cos(heading)
    sine = torch.sin(heading)
    local_x = relative[..., 0] * cosine + relative[..., 1] * sine
    local_y = -relative[..., 0] * sine + relative[..., 1] * cosine
    local_z = relative[..., 2]
    half_size = boxes[:, 3:6] * 0.5 + margin

    inside = (
        (local_x.abs() <= half_size[None, :, 0])
        & (local_y.abs() <= half_size[None, :, 1])
        & (local_z.abs() <= half_size[None, :, 2])
    )
    normalized_distance = (
        (local_x / half_size[None, :, 0].clamp_min(1e-12)).square()
        + (local_y / half_size[None, :, 1].clamp_min(1e-12)).square()
        + (local_z / half_size[None, :, 2].clamp_min(1e-12)).square()
    )
    normalized_distance = normalized_distance.masked_fill(
        ~inside, torch.inf
    )
    best_distance, assignments = normalized_distance.min(dim=1)
    assignments = assignments.long()
    assignments[~torch.isfinite(best_distance)] = -1
    return assignments


def points_in_oriented_boxes(
    points_xyz: torch.Tensor,
    boxes: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Return the union mask of points inside yaw-oriented 3D boxes."""
    return assign_points_to_oriented_boxes(points_xyz, boxes, margin) >= 0


def build_iadv_object_ids(
    points: torch.Tensor,
    batch_dict: Dict,
    scope: str,
    target_class_id: Optional[int] = None,
    target_class_ids: Optional[Sequence[int]] = None,
    box_margin: float = 0.0,
) -> torch.Tensor:
    """Return batch-local target object ids for every point.

    ``gt_boxes`` assigns target points to individual boxes. ``scene`` assigns
    every point to object zero within its batch sample so full-scene grouping
    remains sample-local.
    """
    if scope not in {'scene', 'gt_boxes'}:
        raise ValueError('I-ADV scope must be "scene" or "gt_boxes"')
    batch_indices = points[:, 0].long()
    if scope == 'scene':
        return torch.zeros(
            points.shape[0], dtype=torch.long, device=points.device
        )
    if 'gt_boxes' not in batch_dict:
        raise ValueError('I-ADV gt_boxes scope requires batch_dict["gt_boxes"]')

    gt_boxes = batch_dict['gt_boxes']
    if gt_boxes.ndim != 3 or gt_boxes.shape[0] != int(batch_dict['batch_size']):
        raise ValueError('gt_boxes must have shape [batch_size, M, box_features]')

    object_ids = torch.full(
        (points.shape[0],), -1, dtype=torch.long, device=points.device
    )
    for batch_index in range(int(batch_dict['batch_size'])):
        sample_points = torch.nonzero(
            batch_indices == batch_index, as_tuple=False
        ).flatten()
        boxes = gt_boxes[batch_index]
        valid_boxes = (boxes[:, 3:6] > 0).all(dim=1)
        selected_class_ids = target_class_ids
        if selected_class_ids is None and target_class_id is not None:
            selected_class_ids = [target_class_id]
        if selected_class_ids is not None:
            if boxes.shape[1] < 8:
                raise ValueError(
                    'target-class filtering requires class id in the last '
                    'gt_boxes column'
                )
            class_ids = torch.as_tensor(
                sorted({int(value) for value in selected_class_ids}),
                device=boxes.device,
                dtype=torch.long,
            )
            if class_ids.numel() == 0:
                raise ValueError('target_class_ids must not be empty')
            valid_boxes &= torch.isin(boxes[:, -1].long(), class_ids)
        boxes = boxes[valid_boxes]
        if sample_points.numel() and boxes.numel():
            object_ids[sample_points] = assign_points_to_oriented_boxes(
                points[sample_points, 1:4], boxes, margin=box_margin
            )
    return object_ids


def build_iadv_attack_mask(
    points: torch.Tensor,
    batch_dict: Dict,
    scope: str,
    target_class_id: Optional[int] = None,
    target_class_ids: Optional[Sequence[int]] = None,
    box_margin: float = 0.0,
) -> torch.Tensor:
    """Select scene points or the union of target-class GT boxes."""
    return build_iadv_object_ids(
        points,
        batch_dict,
        scope=scope,
        target_class_id=target_class_id,
        target_class_ids=target_class_ids,
        box_margin=box_margin,
    ) >= 0


def build_iadv_groups(
    points: torch.Tensor,
    attack_mask: torch.Tensor,
    voxel_size: float,
    object_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """Group attacked points into sensor-anchored equal cubes.

    When ``object_ids`` is provided it becomes part of the group key, which
    prevents two target objects from sharing a fusion group even when their
    quantized cube coordinates coincide.
    """
    if voxel_size <= 0:
        raise ValueError('I-ADV attack voxel size must be positive')
    if attack_mask.shape != (points.shape[0],):
        raise ValueError('attack_mask must have shape [N]')
    if object_ids is not None:
        if object_ids.shape != (points.shape[0],):
            raise ValueError('object_ids must have shape [N]')
        if (object_ids[attack_mask] < 0).any():
            raise ValueError('attacked points must have non-negative object ids')

    group_ids = torch.full(
        (points.shape[0],), -1, dtype=torch.long, device=points.device
    )
    attacked_indices = torch.nonzero(attack_mask, as_tuple=False).flatten()
    if attacked_indices.numel() == 0:
        return group_ids, 0

    cube_coords = torch.floor(
        points[attacked_indices, 1:4] / float(voxel_size)
    ).long()
    key_parts = [points[attacked_indices, :1].long()]
    if object_ids is not None:
        key_parts.append(object_ids[attacked_indices, None].long())
    key_parts.append(cube_coords)
    keys = torch.cat(key_parts, dim=1)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    group_ids[attacked_indices] = inverse
    return group_ids, int(inverse.max().item()) + 1


def _query_neighbours(
    candidate_xyz: np.ndarray,
    query_xyz: np.ndarray,
    k_neighbors: int,
    radius: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    query_k = min(k_neighbors, candidate_xyz.shape[0])
    tree = cKDTree(candidate_xyz)
    distance_upper_bound = np.inf if radius is None else radius
    distances, indices = tree.query(
        query_xyz,
        k=query_k,
        distance_upper_bound=distance_upper_bound,
        workers=1,
    )
    if query_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return distances, indices


def compute_reflectivity_features(
    points: torch.Tensor,
    attack_mask: torch.Tensor,
    k_neighbors: int = 16,
    min_neighbors: int = 3,
    neighbor_radius: Optional[float] = None,
    d_max: float = 75.0,
    neighbor_scope: str = 'attack_union',
    object_ids: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute I-ADV's ``sin(phi) * sin(d/d_max*pi/2)`` feature.

    The query point is included among its k nearest neighbours. When fewer
    than ``min_neighbors`` valid points are available, the angular factor is
    set to one, which is a documented neutral (distance-only) Radar fallback.
    """
    if k_neighbors <= 0:
        raise ValueError('k_neighbors must be positive')
    if min_neighbors < 3:
        raise ValueError('min_neighbors must be at least 3 for plane PCA')
    if min_neighbors > k_neighbors:
        raise ValueError('min_neighbors must not exceed k_neighbors')
    if neighbor_radius is not None and neighbor_radius <= 0:
        raise ValueError('neighbor_radius must be positive')
    if d_max <= 0:
        raise ValueError('d_max must be positive')
    if neighbor_scope not in {'object', 'attack_union', 'scene'}:
        raise ValueError(
            'I-ADV neighbor scope must be object, attack_union, or scene'
        )
    if object_ids is None:
        object_ids = torch.full(
            (points.shape[0],), -1, dtype=torch.long, device=points.device
        )
    elif object_ids.shape != (points.shape[0],):
        raise ValueError('object_ids must have shape [N]')
    if neighbor_scope == 'object' and (object_ids[attack_mask] < 0).any():
        raise ValueError('object neighbor scope requires assigned object ids')
    if candidate_mask is None:
        candidate_mask = torch.ones(
            points.shape[0], dtype=torch.bool, device=points.device
        )
    elif candidate_mask.shape != (points.shape[0],):
        raise ValueError('candidate_mask must have shape [N]')

    features = points.new_zeros(points.shape[0])
    valid_normals = 0
    fallback_normals = 0
    neighbor_count_sum = 0
    cross_target_neighbors = 0
    attacked_indices = torch.nonzero(attack_mask, as_tuple=False).flatten()
    batch_indices = points[:, 0].long()

    for batch_index in torch.unique(batch_indices[attacked_indices]).tolist():
        sample_query_indices = torch.nonzero(
            attack_mask & (batch_indices == batch_index), as_tuple=False
        ).flatten()
        if neighbor_scope == 'object':
            pools = [
                (
                    sample_query_indices[
                        object_ids[sample_query_indices] == object_id
                    ],
                    sample_query_indices[
                        object_ids[sample_query_indices] == object_id
                    ],
                )
                for object_id in torch.unique(
                    object_ids[sample_query_indices]
                ).tolist()
            ]
        else:
            sample_candidates = torch.nonzero(
                (batch_indices == batch_index)
                & (
                    attack_mask
                    if neighbor_scope == 'attack_union'
                    else candidate_mask
                ),
                as_tuple=False,
            ).flatten()
            pools = [(sample_query_indices, sample_candidates)]

        for query_indices, candidate_indices in pools:
            query_xyz = points[query_indices, 1:4]
            candidate_xyz = points[candidate_indices, 1:4]
            if candidate_indices.numel() == 0:
                counts = torch.zeros(
                    query_indices.numel(), dtype=torch.long, device=points.device
                )
                reliable = torch.zeros_like(counts, dtype=torch.bool)
                angular_factor = query_xyz.new_ones(query_indices.numel())
            else:
                distances_np, neighbours_np = _query_neighbours(
                    candidate_xyz.detach().cpu().numpy(),
                    query_xyz.detach().cpu().numpy(),
                    k_neighbors,
                    neighbor_radius,
                )
                valid_np = np.isfinite(distances_np) & (
                    neighbours_np < candidate_xyz.shape[0]
                )
                counts = torch.as_tensor(
                    valid_np.sum(axis=1),
                    device=points.device,
                    dtype=torch.long,
                )
                neighbour_indices = torch.as_tensor(
                    neighbours_np, device=points.device, dtype=torch.long
                )
                valid = torch.as_tensor(valid_np, device=points.device)
                neighbour_indices = torch.where(
                    valid, neighbour_indices, torch.zeros_like(neighbour_indices)
                )

                neighbour_xyz = candidate_xyz[neighbour_indices]
                weights = valid.unsqueeze(-1).to(query_xyz.dtype)
                safe_counts = counts.clamp_min(1).to(query_xyz.dtype).unsqueeze(1)
                means = (neighbour_xyz * weights).sum(dim=1) / safe_counts
                centered = (neighbour_xyz - means[:, None, :]) * weights
                covariance = torch.matmul(centered.transpose(1, 2), centered)
                covariance = covariance / safe_counts.unsqueeze(2)
                _, eigenvectors = torch.linalg.eigh(covariance)
                normals = eigenvectors[:, :, 0]

                point_distance = torch.linalg.vector_norm(query_xyz, dim=1)
                safe_distance = point_distance.clamp_min(1e-12)
                beam_direction = query_xyz / safe_distance.unsqueeze(1)
                cosine = (
                    (normals * beam_direction).sum(dim=1).abs().clamp(0.0, 1.0)
                )
                sin_phi = torch.sqrt(
                    (1.0 - cosine.square()).clamp_min(0.0)
                )
                reliable = (
                    (counts >= min_neighbors) & (point_distance > 1e-12)
                )
                angular_factor = torch.where(
                    reliable, sin_phi, torch.ones_like(sin_phi)
                )

                query_objects = object_ids[query_indices, None]
                candidate_objects = object_ids[candidate_indices][
                    neighbour_indices
                ]
                cross_target_neighbors += int(
                    (
                        valid
                        & (query_objects >= 0)
                        & (candidate_objects >= 0)
                        & (query_objects != candidate_objects)
                    ).sum().item()
                )

            point_distance = torch.linalg.vector_norm(query_xyz, dim=1)
            distance_factor = torch.sin(
                point_distance / float(d_max) * torch.pi / 2
            )
            features[query_indices] = angular_factor * distance_factor

            valid_normals += int(reliable.sum().item())
            fallback_normals += int((~reliable).sum().item())
            neighbor_count_sum += int(counts.sum().item())

    attacked_count = int(attacked_indices.numel())
    stats = {
        'reflectivity_valid_normal_points': float(valid_normals),
        'reflectivity_fallback_points': float(fallback_normals),
        'reflectivity_mean_neighbor_count': (
            float(neighbor_count_sum) / attacked_count if attacked_count else 0.0
        ),
        'reflectivity_pca_fallback_rate': (
            float(fallback_normals) / attacked_count if attacked_count else 0.0
        ),
        'reflectivity_cross_target_neighbors': float(cross_target_neighbors),
        'reflectivity_min': (
            float(features[attacked_indices].min().item())
            if attacked_count else 0.0
        ),
        'reflectivity_max': (
            float(features[attacked_indices].max().item())
            if attacked_count else 0.0
        ),
    }
    return features, stats


def normalize_rcs_gradient(
    gradient: torch.Tensor,
    points: torch.Tensor,
    attack_mask: torch.Tensor,
    norm: str,
) -> torch.Tensor:
    """Normalize one RCS gradient vector per batch sample."""
    if norm not in {'l1', 'l2'}:
        raise ValueError('I-ADV gradient norm must be "l1" or "l2"')
    normalized = torch.zeros_like(gradient)
    batch_indices = points[:, 0].long()
    attacked_indices = torch.nonzero(attack_mask, as_tuple=False).flatten()
    for batch_index in torch.unique(batch_indices[attacked_indices]).tolist():
        sample_mask = attack_mask & (batch_indices == batch_index)
        values = gradient[sample_mask]
        denominator = (
            values.abs().sum()
            if norm == 'l1'
            else torch.linalg.vector_norm(values)
        )
        if denominator > 0:
            normalized[sample_mask] = values / denominator
    return normalized


def extremum_fusion(
    enhanced_gradient: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Return one I-ADV extremum-based sign for every point."""
    directions = torch.zeros_like(enhanced_gradient)
    if num_groups == 0:
        return directions
    attack_mask = group_ids >= 0
    active_groups = group_ids[attack_mask]
    active_gradient = enhanced_gradient[attack_mask]
    maxima = torch.full(
        (num_groups,), -torch.inf,
        device=enhanced_gradient.device,
        dtype=enhanced_gradient.dtype,
    )
    minima = torch.full_like(maxima, torch.inf)
    maxima.scatter_reduce_(0, active_groups, active_gradient, reduce='amax')
    minima.scatter_reduce_(0, active_groups, active_gradient, reduce='amin')
    group_directions = torch.where(
        maxima.abs() >= minima.abs(),
        torch.ones_like(maxima),
        -torch.ones_like(maxima),
    )
    group_directions = torch.where(
        (maxima == 0) & (minima == 0),
        torch.zeros_like(group_directions),
        group_directions,
    )
    directions[attack_mask] = group_directions[active_groups]
    return directions


def _project_rcs(
    candidate_rcs: torch.Tensor,
    original_rcs: torch.Tensor,
    epsilon_rcs: float,
    rcs_min: Optional[float],
    rcs_max: Optional[float],
) -> torch.Tensor:
    epsilon = original_rcs.new_tensor(float(epsilon_rcs))
    if epsilon_rcs > 0:
        epsilon = torch.nextafter(epsilon, torch.zeros_like(epsilon))
    upper_bound = original_rcs + epsilon
    lower_bound = original_rcs - epsilon
    if epsilon_rcs > 0:
        # Float32 addition can make ``(original + epsilon) - original`` a few
        # ULPs larger than epsilon. Move both bounds one representable value
        # toward clean RCS so reported perturbations remain budget-conservative.
        upper_bound = torch.nextafter(upper_bound, original_rcs)
        lower_bound = torch.nextafter(lower_bound, original_rcs)
    projected = torch.maximum(
        torch.minimum(candidate_rcs, upper_bound),
        lower_bound,
    )
    if rcs_min is not None:
        projected = projected.clamp_min(rcs_min)
    if rcs_max is not None:
        projected = projected.clamp_max(rcs_max)
    if epsilon_rcs > 0:
        for _ in range(8):
            exceeds_budget = (projected - original_rcs).abs() > epsilon
            if not exceeds_budget.any():
                break
            projected = torch.where(
                exceeds_budget,
                torch.nextafter(projected, original_rcs),
                projected,
            )
        if ((projected - original_rcs).abs() > epsilon).any():
            raise RuntimeError('could not represent an RCS value within budget')
    return projected


def iadv_rcs_attack(
    model: nn.Module,
    batch_dict: Dict,
    voxelizer,
    feature_names: Sequence[str],
    epsilon_rcs: float,
    steps: int = 10,
    step_size: Optional[float] = None,
    attack_voxel_size: float = 0.1,
    momentum_decay: float = 1.0,
    gradient_enhancement: float = 1000.0,
    d_max: float = 75.0,
    k_neighbors: int = 16,
    min_neighbors: int = 3,
    neighbor_radius: Optional[float] = None,
    gradient_norm: str = 'l1',
    scope: str = 'gt_boxes',
    neighbor_scope: str = 'object',
    target_class_id: Optional[int] = None,
    target_class_ids: Optional[Sequence[int]] = None,
    box_margin: float = 0.0,
    rcs_min: Optional[float] = None,
    rcs_max: Optional[float] = None,
) -> AttackOutput:
    """Generate an I-ADV-style adversarial raw Radar point cloud."""
    if epsilon_rcs < 0:
        raise ValueError('epsilon_rcs must be non-negative')
    if steps <= 0:
        raise ValueError('I-ADV steps must be positive')
    if step_size is not None and step_size <= 0:
        raise ValueError('I-ADV step_size must be positive')
    if momentum_decay < 0:
        raise ValueError('I-ADV momentum decay must be non-negative')
    if gradient_enhancement < 0:
        raise ValueError('I-ADV gradient enhancement must be non-negative')
    if rcs_min is not None and rcs_max is not None and rcs_min > rcs_max:
        raise ValueError('rcs_min must not exceed rcs_max')
    if neighbor_scope not in {'object', 'attack_union', 'scene'}:
        raise ValueError(
            'I-ADV neighbor scope must be object, attack_union, or scene'
        )
    if neighbor_scope == 'object' and scope != 'gt_boxes':
        raise ValueError(
            'object neighbor scope requires I-ADV gt_boxes attack scope'
        )

    original = batch_dict['points'].detach().clone()
    rcs_index = rcs_column(feature_names)
    fixed_topology = voxelizer.topology(original)
    model_active_mask = torch.zeros(
        original.shape[0], dtype=torch.bool, device=original.device
    )
    model_active_mask[
        fixed_topology.point_indices[fixed_topology.point_mask]
    ] = True
    object_ids = build_iadv_object_ids(
        original,
        batch_dict,
        scope=scope,
        target_class_id=target_class_id,
        target_class_ids=target_class_ids,
        box_margin=box_margin,
    )
    attack_mask = object_ids >= 0
    attack_mask &= model_active_mask
    isolated_object_ids = object_ids if neighbor_scope == 'object' else None
    group_ids, num_groups = build_iadv_groups(
        original,
        attack_mask,
        attack_voxel_size,
        object_ids=isolated_object_ids,
    )

    active_object_keys = torch.stack(
        (original[:, 0].long(), object_ids), dim=1
    )[attack_mask]
    valid_targets = (
        int(torch.unique(active_object_keys, dim=0).shape[0])
        if active_object_keys.numel()
        else 0
    )

    if not attack_mask.any():
        model_inputs = voxelizer.materialize(original, fixed_topology)
        return AttackOutput(
            adv_points=original,
            model_inputs=model_inputs,
            stats={
                'max_abs_perturbation': 0.0,
                'mean_abs_perturbation': 0.0,
                'sum_abs_perturbation': 0.0,
                'perturbation_values': float(original.shape[0]),
                'iadv_attacked_points': 0.0,
                'iadv_changed_points': 0.0,
                'iadv_groups': 0.0,
                'iadv_valid_targets': 0.0,
                'iadv_mean_points_per_target': 0.0,
                'iadv_singleton_group_ratio': 0.0,
                'iadv_mean_groups_per_target': 0.0,
                'iadv_mean_points_per_group': 0.0,
                'iadv_cross_target_neighbors': 0.0,
                'iadv_pca_fallback_rate': 0.0,
                'iadv_nonfinite_gradient_steps': 0.0,
            },
        )

    reflectivity, reflectivity_stats = compute_reflectivity_features(
        original,
        attack_mask,
        k_neighbors=k_neighbors,
        min_neighbors=min_neighbors,
        neighbor_radius=neighbor_radius,
        d_max=d_max,
        neighbor_scope=neighbor_scope,
        object_ids=object_ids,
        candidate_mask=model_active_mask,
    )
    if (
        neighbor_scope == 'object'
        and reflectivity_stats['reflectivity_cross_target_neighbors'] != 0
    ):
        raise RuntimeError('object-isolated I-ADV found a cross-target neighbour')
    step = 0.2 * epsilon_rcs if step_size is None else float(step_size)
    adversarial = original.clone()
    momentum = original.new_zeros(original.shape[0])

    states = set_attack_mode(model)
    try:
        for _ in range(steps):
            adversarial = adversarial.detach().requires_grad_(True)
            voxel_data = voxelizer.materialize(adversarial, fixed_topology)
            attack_batch = dict(batch_dict)
            attack_batch['points'] = adversarial
            attack_batch.update(voxel_data)

            ret_dict, _, _ = model(attack_batch)
            loss = ret_dict['loss'].mean()
            point_gradient = torch.autograd.grad(loss, adversarial)[0]
            rcs_gradient = point_gradient[:, rcs_index]
            if not torch.isfinite(rcs_gradient).all():
                raise RuntimeError('non-finite RCS gradient during I-ADV attack')

            normalized = normalize_rcs_gradient(
                rcs_gradient,
                adversarial,
                attack_mask,
                gradient_norm,
            )
            momentum = momentum_decay * momentum + normalized
            momentum = gradient_enhancement * reflectivity * momentum
            directions = extremum_fusion(momentum, group_ids, num_groups)

            updated = adversarial.detach().clone()
            candidate_rcs = updated[:, rcs_index] + step * directions
            projected_rcs = _project_rcs(
                candidate_rcs,
                original[:, rcs_index],
                epsilon_rcs,
                rcs_min,
                rcs_max,
            )
            updated[:, rcs_index] = torch.where(
                attack_mask, projected_rcs, original[:, rcs_index]
            )
            adversarial = updated
    finally:
        restore_attack_modes(states)
        model.zero_grad(set_to_none=True)

    adversarial = adversarial.detach()
    unchanged_columns = torch.ones(
        original.shape[1], dtype=torch.bool, device=original.device
    )
    unchanged_columns[rcs_index] = False
    if not torch.equal(
        adversarial[:, unchanged_columns], original[:, unchanged_columns]
    ):
        raise RuntimeError('I-ADV modified a non-RCS point feature')
    model_inputs = voxelizer.materialize(adversarial, fixed_topology)
    absolute_delta = (adversarial[:, rcs_index] - original[:, rcs_index]).abs()
    attacked_delta = absolute_delta[attack_mask]
    group_counts = torch.bincount(
        group_ids[attack_mask], minlength=num_groups
    )
    singleton_groups = int((group_counts == 1).sum().item())
    stats = {
        'max_abs_perturbation': float(absolute_delta.max().item()),
        'mean_abs_perturbation': float(absolute_delta.mean().item()),
        'sum_abs_perturbation': float(absolute_delta.sum().item()),
        'perturbation_values': float(absolute_delta.numel()),
        'iadv_attacked_points': float(attack_mask.sum().item()),
        'iadv_changed_points': float((attacked_delta > 0).sum().item()),
        'iadv_groups': float(num_groups),
        'iadv_singleton_groups': float(singleton_groups),
        'iadv_valid_targets': float(valid_targets),
        'iadv_mean_points_per_target': (
            float(attack_mask.sum().item()) / valid_targets
            if valid_targets else 0.0
        ),
        'iadv_singleton_group_ratio': (
            float(singleton_groups) / num_groups if num_groups else 0.0
        ),
        'iadv_mean_groups_per_target': (
            float(num_groups) / valid_targets if valid_targets else 0.0
        ),
        'iadv_mean_points_per_group': (
            float(attack_mask.sum().item()) / num_groups if num_groups else 0.0
        ),
        'iadv_cross_target_neighbors': reflectivity_stats[
            'reflectivity_cross_target_neighbors'
        ],
        'iadv_pca_fallback_rate': reflectivity_stats[
            'reflectivity_pca_fallback_rate'
        ],
        'iadv_nonfinite_gradient_steps': 0.0,
        'iadv_mean_abs_perturbation_attacked': float(
            attacked_delta.mean().item()
        ),
    }
    stats.update(reflectivity_stats)
    return AttackOutput(
        adv_points=adversarial,
        model_inputs=model_inputs,
        stats=stats,
    )
