"""Radar-measurement-parameterized attacks on raw 4D-radar geometry."""

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from ..temporal import TemporalBatchData
from .base import AttackOutput, restore_attack_modes, set_attack_mode


MEASUREMENT_NAMES = ('range', 'azimuth', 'elevation')
DISTANCE_BINS_METRES = (
    (0.0, 10.0, 'r0_10'),
    (10.0, 20.0, 'r10_20'),
    (20.0, 30.0, 'r20_30'),
    (30.0, 50.0, 'r30_50'),
    (50.0, float('inf'), 'r50_inf'),
)
TEMPORAL_PARAMETER_MODES = (
    'point_independent',
    'object_per_sweep',
    'track_shared',
)


def cartesian_to_radar_measurement(xyz: torch.Tensor) -> torch.Tensor:
    """Convert x-forward, y-left, z-up Cartesian points to [r, az, el]."""
    if xyz.shape[-1] != 3:
        raise ValueError('xyz must have a final dimension of size 3')
    horizontal_range = torch.linalg.vector_norm(xyz[..., :2], dim=-1)
    distance = torch.linalg.vector_norm(xyz, dim=-1)
    azimuth = torch.atan2(xyz[..., 1], xyz[..., 0])
    elevation = torch.atan2(xyz[..., 2], horizontal_range)
    return torch.stack((distance, azimuth, elevation), dim=-1)


def radar_measurement_to_cartesian(measurement: torch.Tensor) -> torch.Tensor:
    """Convert [range, azimuth, elevation] to x-forward/y-left/z-up XYZ."""
    if measurement.shape[-1] != 3:
        raise ValueError('measurement must have a final dimension of size 3')
    distance, azimuth, elevation = measurement.unbind(dim=-1)
    horizontal_range = distance * torch.cos(elevation)
    x = horizontal_range * torch.cos(azimuth)
    y = horizontal_range * torch.sin(azimuth)
    z = distance * torch.sin(elevation)
    return torch.stack((x, y, z), dim=-1)


def _feature_column(feature_names: Sequence[str], name: str) -> int:
    try:
        return list(feature_names).index(name) + 1
    except ValueError as error:
        raise ValueError(
            f'measurement attack requires feature {name!r}; got '
            f'{list(feature_names)}'
        ) from error


def current_sweep_mask(
    points: torch.Tensor,
    feature_names: Sequence[str],
    atol: float = 1e-6,
) -> torch.Tensor:
    """Select VoD current-sweep points whose discrete sweep index is zero."""
    time_column = _feature_column(feature_names, 'time')
    return points[:, time_column].abs() <= float(atol)


def model_active_point_mask(points: torch.Tensor, voxelizer) -> torch.Tensor:
    """Return points retained by clean hard voxelization and its capacity caps."""
    topology = voxelizer.topology(points)
    active = torch.zeros(
        points.shape[0], dtype=torch.bool, device=points.device
    )
    active[topology.point_indices[topology.point_mask]] = True
    return active


def build_measurement_attack_mask(
    points: torch.Tensor,
    feature_names: Sequence[str],
    target_mask: torch.Tensor,
    voxelizer,
    min_range: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Intersect target, current-sweep, valid-range and clean-active masks."""
    if target_mask.shape != (points.shape[0],):
        raise ValueError('target_mask must have shape [N]')
    if list(feature_names[:3]) != ['x', 'y', 'z']:
        raise ValueError(
            'measurement attack requires x, y, z as the first point features'
        )
    target_mask = target_mask.to(device=points.device, dtype=torch.bool)
    sweep_mask = current_sweep_mask(points, feature_names)
    active_mask = model_active_point_mask(points, voxelizer)
    clean_range = torch.linalg.vector_norm(points[:, 1:4], dim=1)
    valid_range_mask = clean_range > float(min_range)
    attack_mask = target_mask & sweep_mask & active_mask & valid_range_mask
    stats = {
        'measurement_target_points': float(target_mask.sum().item()),
        'measurement_current_target_points': float(
            (target_mask & sweep_mask).sum().item()
        ),
        'measurement_historical_target_points': float(
            (target_mask & ~sweep_mask).sum().item()
        ),
        'measurement_clean_active_current_target_points': float(
            attack_mask.sum().item()
        ),
        'measurement_zero_range_target_points': float(
            (target_mask & sweep_mask & active_mask & ~valid_range_mask)
            .sum().item()
        ),
    }
    return attack_mask, stats


def build_temporal_measurement_attack_mask(
    points: torch.Tensor,
    temporal_data: TemporalBatchData,
    voxelizer,
    point_scope: str = 'gt_boxes',
    min_range: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Select clean-active points across all available source sweeps."""
    if point_scope not in {'scene', 'gt_boxes'}:
        raise ValueError('point_scope must be "scene" or "gt_boxes"')
    if len(temporal_data.group_ids) != points.shape[0]:
        raise ValueError('temporal metadata does not align with points')
    group_ids = torch.as_tensor(
        temporal_data.group_ids, dtype=torch.long, device=points.device
    )
    sweep_ids = torch.as_tensor(
        temporal_data.sweep_ids, dtype=torch.long, device=points.device
    )
    source_xyz = torch.as_tensor(
        temporal_data.source_xyz, dtype=points.dtype, device=points.device
    )
    tracked_target_mask = group_ids >= 0
    target_mask = (
        torch.ones_like(tracked_target_mask)
        if point_scope == 'scene'
        else tracked_target_mask
    )
    active_mask = model_active_point_mask(points, voxelizer)
    valid_range = torch.linalg.vector_norm(source_xyz, dim=1) > min_range
    attack_mask = target_mask & active_mask & valid_range
    current = sweep_ids == 0
    attacked_tracked_points = attack_mask & tracked_target_mask
    active_groups = (
        torch.unique(group_ids[attacked_tracked_points]).numel()
        if attacked_tracked_points.any()
        else 0
    )
    stats = {
        **{
            key: float(value)
            for key, value in temporal_data.diagnostics.items()
        },
        'measurement_target_points': float(target_mask.sum().item()),
        'measurement_current_target_points': float(
            (target_mask & current).sum().item()
        ),
        'measurement_historical_target_points': float(
            (target_mask & ~current).sum().item()
        ),
        'measurement_clean_active_target_points': float(
            attack_mask.sum().item()
        ),
        'measurement_clean_active_current_target_points': float(
            (attack_mask & current).sum().item()
        ),
        'measurement_clean_active_historical_target_points': float(
            (attack_mask & ~current).sum().item()
        ),
        'measurement_zero_range_target_points': float(
            (target_mask & active_mask & ~valid_range).sum().item()
        ),
        'temporal_active_shared_groups': float(active_groups),
        'measurement_scene_scope': float(point_scope == 'scene'),
        'measurement_gt_boxes_scope': float(point_scope == 'gt_boxes'),
    }
    return attack_mask, stats


def build_temporal_parameter_groups(
    temporal_data: TemporalBatchData,
    mode: str,
    point_scope: str = 'gt_boxes',
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """Build compact measurement-parameter groups for a temporal batch.

    GT-box membership comes from the track-aware temporal data layer. Scene
    scope is defined only for point-independent parameters because background
    points do not have object or track identities. Non-target points retain -1.
    """
    if mode not in TEMPORAL_PARAMETER_MODES:
        raise ValueError(
            f'temporal parameter mode must be one of '
            f'{TEMPORAL_PARAMETER_MODES}; got {mode!r}'
        )
    if point_scope not in {'scene', 'gt_boxes'}:
        raise ValueError('point_scope must be "scene" or "gt_boxes"')
    if point_scope == 'scene' and mode != 'point_independent':
        raise ValueError(
            'scene temporal measurement attack requires point_independent mode'
        )
    track_groups = torch.as_tensor(
        temporal_data.group_ids, dtype=torch.long, device=device
    )
    sweep_ids = torch.as_tensor(
        temporal_data.sweep_ids, dtype=torch.long, device=device
    )
    if track_groups.ndim != 1 or sweep_ids.shape != track_groups.shape:
        raise ValueError('temporal group and sweep IDs must be aligned vectors')
    target = (
        torch.ones_like(track_groups, dtype=torch.bool)
        if point_scope == 'scene'
        else track_groups >= 0
    )
    parameter_groups = torch.full_like(track_groups, -1)
    if not target.any():
        return parameter_groups, 0

    if mode == 'track_shared':
        _, inverse = torch.unique(
            track_groups[target], sorted=True, return_inverse=True
        )
    elif mode == 'object_per_sweep':
        keys = torch.stack((track_groups[target], sweep_ids[target]), dim=1)
        _, inverse = torch.unique(
            keys, sorted=True, return_inverse=True, dim=0
        )
    else:
        inverse = torch.arange(
            int(target.sum().item()), dtype=torch.long,
            device=track_groups.device,
        )
    parameter_groups[target] = inverse
    return parameter_groups, int(inverse.max().item() + 1)


def _inside_point_cloud_range(
    xyz: torch.Tensor, point_cloud_range: Sequence[float]
) -> torch.Tensor:
    range_min = xyz.new_tensor(point_cloud_range[:3])
    range_max = xyz.new_tensor(point_cloud_range[3:])
    return ((xyz >= range_min) & (xyz < range_max)).all(dim=1)


def project_radar_measurements(
    candidate: torch.Tensor,
    original: torch.Tensor,
    budget: torch.Tensor,
    attack_mask: torch.Tensor,
    point_cloud_range: Sequence[float],
    min_range: float = 1e-6,
    max_backtracks: int = 24,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Project in measurement space and backtrack updates leaving model range.

    Cartesian coordinates are never independently clipped. A measurement update
    that reconstructs outside the detector range is halved until it is valid,
    or reset to the clean measurement if no representable valid update remains.
    """
    if candidate.shape != original.shape or candidate.shape[-1] != 3:
        raise ValueError('candidate and original must have shape [N, 3]')
    if budget.shape not in {(1, 3), original.shape}:
        raise ValueError('budget must have shape [1, 3] or [N, 3]')
    if attack_mask.shape != (original.shape[0],):
        raise ValueError('attack_mask must have shape [N]')
    if max_backtracks <= 0:
        raise ValueError('max_backtracks must be positive')

    effective_budget = budget.expand_as(original)
    projected = torch.maximum(
        torch.minimum(candidate, original + effective_budget),
        original - effective_budget,
    )
    projected[:, 0] = projected[:, 0].clamp_min(float(min_range))
    half_pi = torch.pi / 2
    projected[:, 2] = projected[:, 2].clamp(-half_pi, half_pi)
    projected = torch.where(attack_mask.unsqueeze(1), projected, original)

    delta = projected - original
    reconstructed = radar_measurement_to_cartesian(projected)
    valid = _inside_point_cloud_range(reconstructed, point_cloud_range)
    invalid = attack_mask & ~valid
    backtracked_points = int(invalid.sum().item())
    scale = torch.ones(
        (original.shape[0], 1), dtype=original.dtype, device=original.device
    )
    for _ in range(max_backtracks):
        if not invalid.any():
            break
        scale = torch.where(invalid.unsqueeze(1), scale * 0.5, scale)
        projected = original + delta * scale
        reconstructed = radar_measurement_to_cartesian(projected)
        valid = _inside_point_cloud_range(reconstructed, point_cloud_range)
        invalid = attack_mask & ~valid
    rejected_points = int(invalid.sum().item())
    if rejected_points:
        projected = torch.where(invalid.unsqueeze(1), original, projected)

    # Floating-point addition/subtraction can make a value at the exact bound
    # exceed epsilon by one ULP. Move only those values toward clean, matching
    # the existing Cartesian projector's representable-budget guarantee.
    for _ in range(8):
        absolute_delta = (projected - original).abs()
        exceeds_budget = absolute_delta > effective_budget
        if not exceeds_budget.any():
            break
        projected = torch.where(
            exceeds_budget,
            torch.nextafter(projected, original),
            projected,
        )
    absolute_delta = (projected - original).abs()
    if (absolute_delta > effective_budget).any():
        raise RuntimeError('measurement projection exceeded its epsilon budget')
    return projected, {
        'measurement_out_of_range_backtracks': float(backtracked_points),
        'measurement_out_of_range_rejections': float(rejected_points),
    }


def _measurement_budget(
    original: torch.Tensor,
    epsilon_range: float,
    epsilon_azimuth: float,
    epsilon_elevation: float,
) -> torch.Tensor:
    values = original.new_tensor(
        [[epsilon_range, epsilon_azimuth, epsilon_elevation]]
    )
    return torch.where(
        values > 0,
        torch.nextafter(values, torch.zeros_like(values)),
        values,
    )


def _measurement_step(
    budget: torch.Tensor,
    steps: int,
    attack_type: str,
    step_size_range: Optional[float],
    step_size_azimuth: Optional[float],
    step_size_elevation: Optional[float],
) -> torch.Tensor:
    if attack_type == 'fgsm':
        return budget.clone()
    default = budget * (2.0 / steps)
    overrides = (step_size_range, step_size_azimuth, step_size_elevation)
    values = [
        default[0, index] if value is None else default.new_tensor(value)
        for index, value in enumerate(overrides)
    ]
    step = torch.stack(values).reshape(1, 3)
    return torch.minimum(step, budget)


def _points_from_measurements(
    original_points: torch.Tensor,
    measurements: torch.Tensor,
    attack_mask: torch.Tensor,
) -> torch.Tensor:
    reconstructed = radar_measurement_to_cartesian(measurements)
    xyz = torch.where(
        attack_mask.unsqueeze(1), reconstructed, original_points[:, 1:4]
    )
    return torch.cat(
        (original_points[:, :1], xyz, original_points[:, 4:]), dim=1
    )


def _empty_attack_output(
    original: torch.Tensor,
    voxelizer,
    stats: Dict[str, float],
    loss_stats: Optional[Dict[str, float]],
) -> AttackOutput:
    topology = voxelizer.topology(original)
    voxel_data = voxelizer.materialize(original, topology)
    stats.update({
        'max_abs_perturbation': 0.0,
        'mean_abs_perturbation': 0.0,
        'sum_abs_perturbation': 0.0,
        'perturbation_values': 0.0,
        'measurement_max_abs_delta_range': 0.0,
        'measurement_max_abs_delta_azimuth_rad': 0.0,
        'measurement_max_abs_delta_elevation_rad': 0.0,
        'measurement_max_xyz_l2': 0.0,
        'measurement_xyz_l2_sum': 0.0,
        'measurement_modified_current_points': 0.0,
        'measurement_historical_modification_count': 0.0,
        'measurement_non_target_modification_count': 0.0,
        'measurement_non_geometry_modification_count': 0.0,
        'measurement_nonfinite_gradient_steps': 0.0,
        'measurement_zero_gradient_steps': 0.0,
    })
    if loss_stats is not None:
        stats.update({key: float(value) for key, value in loss_stats.items()})
    return AttackOutput(original, voxel_data, stats)


def radar_measurement_geometry_attack(
    model: torch.nn.Module,
    batch_dict: Dict,
    voxelizer,
    feature_names: Sequence[str],
    attack_type: str,
    epsilon_range: float,
    epsilon_azimuth: float,
    epsilon_elevation: float,
    pgd_steps: int = 10,
    step_size_range: Optional[float] = None,
    step_size_azimuth: Optional[float] = None,
    step_size_elevation: Optional[float] = None,
    random_start: bool = False,
    target_mask: Optional[torch.Tensor] = None,
    attack_mask: Optional[torch.Tensor] = None,
    attack_mask_stats: Optional[Dict[str, float]] = None,
    loss_fn: Optional[Callable[[torch.nn.Module], torch.Tensor]] = None,
    loss_stats: Optional[Dict[str, float]] = None,
) -> AttackOutput:
    """Attack current-sweep target geometry in Radar measurement space."""
    if attack_type not in {'fgsm', 'pgd'}:
        raise ValueError('measurement attack_type must be "fgsm" or "pgd"')
    if pgd_steps <= 0:
        raise ValueError('pgd_steps must be positive')
    named_values = {
        'epsilon_range': epsilon_range,
        'epsilon_azimuth': epsilon_azimuth,
        'epsilon_elevation': epsilon_elevation,
        'step_size_range': step_size_range,
        'step_size_azimuth': step_size_azimuth,
        'step_size_elevation': step_size_elevation,
    }
    for name, value in named_values.items():
        if value is None:
            continue
        if name.startswith('step_size_') and value <= 0:
            raise ValueError(f'{name} must be positive')
        if name.startswith('epsilon_') and value < 0:
            raise ValueError(f'{name} must be non-negative')
    if target_mask is None:
        raise ValueError('measurement attack requires an object target_mask')

    original = batch_dict['points'].detach().clone()
    computed_attack_mask, stats = build_measurement_attack_mask(
        original, feature_names, target_mask, voxelizer
    )
    if attack_mask is None:
        attack_mask = computed_attack_mask
    else:
        if attack_mask.shape != (original.shape[0],):
            raise ValueError('attack_mask must have shape [N]')
        attack_mask = attack_mask.to(device=original.device, dtype=torch.bool)
        if not torch.equal(attack_mask, computed_attack_mask):
            raise ValueError(
                'precomputed measurement attack_mask differs from the '
                'target/current/clean-active mask'
            )
    if attack_mask_stats is not None:
        stats.update({key: float(value) for key, value in attack_mask_stats.items()})
    stats['measurement_batches'] = float(batch_dict['batch_size'])
    batch_indices = original[:, 0].long()
    attacked_batches = torch.unique(batch_indices[attack_mask]).numel()
    stats['measurement_batches_with_attack_points'] = float(attacked_batches)

    original_measurements = cartesian_to_radar_measurement(original[:, 1:4])
    budget = _measurement_budget(
        original_measurements,
        epsilon_range,
        epsilon_azimuth,
        epsilon_elevation,
    )
    if not attack_mask.any() or not (budget > 0).any():
        return _empty_attack_output(
            original, voxelizer, stats, loss_stats
        )
    steps = 1 if attack_type == 'fgsm' else pgd_steps
    step = _measurement_step(
        budget,
        steps,
        attack_type,
        step_size_range,
        step_size_azimuth,
        step_size_elevation,
    )
    adversarial_measurements = original_measurements.clone()
    backtracks = 0.0
    rejections = 0.0
    zero_gradient_steps = 0.0

    if attack_type == 'pgd' and random_start:
        random_delta = (
            torch.rand_like(original_measurements) * 2.0 - 1.0
        ) * budget
        adversarial_measurements, projection_stats = (
            project_radar_measurements(
                adversarial_measurements + random_delta,
                original_measurements,
                budget,
                attack_mask,
                voxelizer.point_cloud_range,
            )
        )
        backtracks += projection_stats[
            'measurement_out_of_range_backtracks'
        ]
        rejections += projection_stats[
            'measurement_out_of_range_rejections'
        ]

    states = set_attack_mode(model)
    try:
        for _ in range(steps):
            adversarial_measurements = (
                adversarial_measurements.detach().requires_grad_(True)
            )
            adversarial_points = _points_from_measurements(
                original, adversarial_measurements, attack_mask
            )
            topology = voxelizer.topology(adversarial_points)
            voxel_data = voxelizer.materialize(
                adversarial_points, topology
            )
            attack_batch = dict(batch_dict)
            attack_batch['points'] = adversarial_points
            attack_batch.update(voxel_data)
            ret_dict, _, _ = model(attack_batch)
            loss = (
                ret_dict['loss'].mean()
                if loss_fn is None
                else loss_fn(model).mean()
            )
            gradient = torch.autograd.grad(
                loss, adversarial_measurements
            )[0]
            attacked_gradient = gradient[attack_mask]
            if not torch.isfinite(attacked_gradient).all():
                raise RuntimeError(
                    'non-finite Radar measurement gradient during attack'
                )
            if not attacked_gradient.abs().any():
                zero_gradient_steps += 1.0
            candidate = (
                adversarial_measurements
                + step * gradient.sign() * attack_mask.unsqueeze(1)
            )
            adversarial_measurements, projection_stats = (
                project_radar_measurements(
                    candidate,
                    original_measurements,
                    budget,
                    attack_mask,
                    voxelizer.point_cloud_range,
                )
            )
            backtracks += projection_stats[
                'measurement_out_of_range_backtracks'
            ]
            rejections += projection_stats[
                'measurement_out_of_range_rejections'
            ]
    finally:
        restore_attack_modes(states)
        model.zero_grad(set_to_none=True)

    adversarial_measurements = adversarial_measurements.detach()
    measurement_delta = adversarial_measurements - original_measurements
    changed_measurement = measurement_delta.abs().any(dim=1) & attack_mask
    reconstructed_xyz = radar_measurement_to_cartesian(
        adversarial_measurements
    )
    final_xyz = torch.where(
        changed_measurement.unsqueeze(1),
        reconstructed_xyz,
        original[:, 1:4],
    )
    adversarial = torch.cat(
        (original[:, :1], final_xyz, original[:, 4:]), dim=1
    ).detach()
    final_topology = voxelizer.topology(adversarial)
    voxel_data = voxelizer.materialize(adversarial, final_topology)

    xyz_delta = adversarial[:, 1:4] - original[:, 1:4]
    xyz_l2 = torch.linalg.vector_norm(xyz_delta, dim=1)
    modified = xyz_delta.abs().any(dim=1)
    sweep_mask = current_sweep_mask(original, feature_names)
    historical_modifications = modified & ~sweep_mask
    non_target_modifications = modified & ~target_mask.to(
        device=original.device, dtype=torch.bool
    )
    non_geometry_modifications = (
        adversarial[:, 4:] != original[:, 4:]
    ).any(dim=1)
    if historical_modifications.any():
        raise RuntimeError('measurement attack modified a historical point')
    if non_target_modifications.any():
        raise RuntimeError('measurement attack modified a non-target point')
    if non_geometry_modifications.any():
        raise RuntimeError('measurement attack modified a non-geometry feature')

    absolute_measurement_delta = measurement_delta.abs()[attack_mask]
    xyz_components = xyz_delta.abs()[attack_mask]
    stats.update({
        'max_abs_perturbation': (
            xyz_components.max().item() if xyz_components.numel() else 0.0
        ),
        'mean_abs_perturbation': (
            xyz_components.mean().item() if xyz_components.numel() else 0.0
        ),
        'sum_abs_perturbation': (
            xyz_components.sum().item() if xyz_components.numel() else 0.0
        ),
        'perturbation_values': float(xyz_components.numel()),
        'measurement_max_abs_delta_range': float(
            absolute_measurement_delta[:, 0].max().item()
        ),
        'measurement_max_abs_delta_azimuth_rad': float(
            absolute_measurement_delta[:, 1].max().item()
        ),
        'measurement_max_abs_delta_elevation_rad': float(
            absolute_measurement_delta[:, 2].max().item()
        ),
        'measurement_max_xyz_l2': float(xyz_l2[attack_mask].max().item()),
        'measurement_xyz_l2_sum': float(xyz_l2[attack_mask].sum().item()),
        'measurement_modified_current_points': float(modified.sum().item()),
        'measurement_historical_modification_count': float(
            historical_modifications.sum().item()
        ),
        'measurement_non_target_modification_count': float(
            non_target_modifications.sum().item()
        ),
        'measurement_non_geometry_modification_count': float(
            non_geometry_modifications.sum().item()
        ),
        'measurement_out_of_range_backtracks': backtracks,
        'measurement_out_of_range_rejections': rejections,
        'measurement_nonfinite_gradient_steps': 0.0,
        'measurement_zero_gradient_steps': zero_gradient_steps,
    })
    clean_range = original_measurements[:, 0]
    for lower, upper, suffix in DISTANCE_BINS_METRES:
        bin_mask = attack_mask & (clean_range >= lower)
        if upper != float('inf'):
            bin_mask &= clean_range < upper
        stats[f'measurement_xyz_l2_count_{suffix}'] = float(
            bin_mask.sum().item()
        )
        stats[f'measurement_xyz_l2_sum_{suffix}'] = float(
            xyz_l2[bin_mask].sum().item()
        )
    if loss_stats is not None:
        stats.update({key: float(value) for key, value in loss_stats.items()})
    return AttackOutput(adversarial, voxel_data, stats)


def _temporal_reference_xyz(
    original_reference_xyz: torch.Tensor,
    original_measurements: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    group_delta: torch.Tensor,
    group_ids: torch.Tensor,
    attack_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct source measurements and move their displacement to reference."""
    point_delta = torch.zeros_like(original_measurements)
    point_delta[attack_mask] = group_delta[group_ids[attack_mask]]
    adversarial_measurements = original_measurements + point_delta
    clean_source = radar_measurement_to_cartesian(original_measurements)
    adversarial_source = radar_measurement_to_cartesian(
        adversarial_measurements
    )
    clean_reference_fit = torch.bmm(
        rotations, clean_source.unsqueeze(-1)
    ).squeeze(-1) + translations
    adversarial_reference_fit = torch.bmm(
        rotations, adversarial_source.unsqueeze(-1)
    ).squeeze(-1) + translations
    # Cancelling the fitted clean reference makes delta=0 bit-exact while
    # retaining a differentiable source-measurement-to-reference path.
    displacement = adversarial_reference_fit - clean_reference_fit
    reference_xyz = original_reference_xyz + displacement
    return reference_xyz, adversarial_measurements


def project_temporal_group_delta(
    candidate: torch.Tensor,
    original_reference_xyz: torch.Tensor,
    original_measurements: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    group_ids: torch.Tensor,
    attack_mask: torch.Tensor,
    active_groups: torch.Tensor,
    budget: torch.Tensor,
    point_cloud_range: Sequence[float],
    min_range: float = 1e-6,
    max_backtracks: int = 24,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Project shared deltas without breaking equality within a track group."""
    if candidate.ndim != 2 or candidate.shape[1] != 3:
        raise ValueError('candidate must have shape [num_groups, 3]')
    if active_groups.shape != (candidate.shape[0],):
        raise ValueError('active_groups must have shape [num_groups]')
    if budget.shape != (1, 3):
        raise ValueError('budget must have shape [1, 3]')
    if max_backtracks <= 0:
        raise ValueError('max_backtracks must be positive')
    projected = torch.maximum(
        torch.minimum(candidate, budget), -budget
    )
    projected = torch.where(
        active_groups.unsqueeze(1), projected, torch.zeros_like(projected)
    )
    range_min = original_reference_xyz.new_tensor(point_cloud_range[:3])
    range_max = original_reference_xyz.new_tensor(point_cloud_range[3:])
    initial_invalid_points = 0
    initial_invalid_groups = 0
    invalid_groups = projected.new_zeros(
        candidate.shape[0], dtype=torch.bool
    )
    for iteration in range(max_backtracks + 1):
        reference_xyz, adversarial_measurements = _temporal_reference_xyz(
            original_reference_xyz,
            original_measurements,
            rotations,
            translations,
            projected,
            group_ids,
            attack_mask,
        )
        valid_source = adversarial_measurements[:, 0] > float(min_range)
        valid_source &= adversarial_measurements[:, 2].abs() <= torch.pi / 2
        valid_reference = (
            (reference_xyz >= range_min) & (reference_xyz < range_max)
        ).all(dim=1)
        invalid_points = attack_mask & ~(valid_source & valid_reference)
        invalid_groups.zero_()
        if invalid_points.any():
            invalid_groups[
                torch.unique(group_ids[invalid_points])
            ] = True
        if iteration == 0:
            initial_invalid_points = int(invalid_points.sum().item())
            initial_invalid_groups = int(invalid_groups.sum().item())
        if not invalid_groups.any():
            break
        if iteration == max_backtracks:
            projected = torch.where(
                invalid_groups.unsqueeze(1),
                torch.zeros_like(projected),
                projected,
            )
            break
        projected = torch.where(
            invalid_groups.unsqueeze(1), projected * 0.5, projected
        )

    for _ in range(8):
        exceeds = projected.abs() > budget
        if not exceeds.any():
            break
        projected = torch.where(
            exceeds,
            torch.nextafter(projected, torch.zeros_like(projected)),
            projected,
        )
    if (projected.abs() > budget).any():
        raise RuntimeError('temporal shared delta exceeded its budget')
    return projected, {
        'temporal_out_of_range_backtrack_points': float(
            initial_invalid_points
        ),
        'temporal_out_of_range_backtrack_groups': float(
            initial_invalid_groups
        ),
        'temporal_out_of_range_rejected_groups': float(
            invalid_groups.sum().item()
        ),
    }


def radar_temporal_measurement_geometry_attack(
    model: torch.nn.Module,
    batch_dict: Dict,
    voxelizer,
    temporal_data: TemporalBatchData,
    attack_type: str,
    epsilon_range: float,
    epsilon_azimuth: float,
    epsilon_elevation: float,
    temporal_mode: str = 'track_shared',
    point_scope: str = 'gt_boxes',
    pgd_steps: int = 10,
    step_size_range: Optional[float] = None,
    step_size_azimuth: Optional[float] = None,
    step_size_elevation: Optional[float] = None,
    random_start: bool = False,
    attack_mask: Optional[torch.Tensor] = None,
    attack_mask_stats: Optional[Dict[str, float]] = None,
    loss_fn: Optional[Callable[[torch.nn.Module], torch.Tensor]] = None,
    loss_stats: Optional[Dict[str, float]] = None,
) -> AttackOutput:
    """Attack all source sweeps with configurable parameter sharing."""
    if attack_type not in {'fgsm', 'pgd'}:
        raise ValueError('temporal measurement attack supports FGSM/PGD only')
    if pgd_steps <= 0:
        raise ValueError('pgd_steps must be positive')
    for name, value in {
        'epsilon_range': epsilon_range,
        'epsilon_azimuth': epsilon_azimuth,
        'epsilon_elevation': epsilon_elevation,
    }.items():
        if value < 0:
            raise ValueError(f'{name} must be non-negative')
    if temporal_mode not in TEMPORAL_PARAMETER_MODES:
        raise ValueError(
            f'temporal_mode must be one of {TEMPORAL_PARAMETER_MODES}'
        )

    original = batch_dict['points'].detach().clone()
    computed_mask, stats = build_temporal_measurement_attack_mask(
        original, temporal_data, voxelizer, point_scope=point_scope
    )
    if attack_mask is None:
        attack_mask = computed_mask
    else:
        attack_mask = attack_mask.to(
            device=original.device, dtype=torch.bool
        )
        if not torch.equal(attack_mask, computed_mask):
            raise ValueError('precomputed temporal attack mask differs')
    if attack_mask_stats is not None:
        stats.update({
            key: float(value) for key, value in attack_mask_stats.items()
        })
    if loss_stats is not None:
        stats.update({key: float(value) for key, value in loss_stats.items()})
    stats['measurement_batches'] = float(batch_dict['batch_size'])
    stats['measurement_batches_with_attack_points'] = float(
        torch.unique(original[attack_mask, 0].long()).numel()
    )

    source_xyz = torch.as_tensor(
        temporal_data.source_xyz,
        dtype=original.dtype,
        device=original.device,
    )
    rotations = torch.as_tensor(
        temporal_data.rotations,
        dtype=original.dtype,
        device=original.device,
    )
    translations = torch.as_tensor(
        temporal_data.translations,
        dtype=original.dtype,
        device=original.device,
    )
    group_ids, group_count = build_temporal_parameter_groups(
        temporal_data,
        temporal_mode,
        point_scope=point_scope,
        device=original.device,
    )
    sweep_ids = torch.as_tensor(
        temporal_data.sweep_ids,
        dtype=torch.long,
        device=original.device,
    )
    active_groups = torch.zeros(
        group_count, dtype=torch.bool, device=original.device
    )
    if attack_mask.any():
        active_groups[torch.unique(group_ids[attack_mask])] = True
    stats.update({
        'temporal_parameter_groups': float(group_count),
        'temporal_active_parameter_groups': float(active_groups.sum().item()),
        'temporal_point_independent_mode': float(
            temporal_mode == 'point_independent'
        ),
        'temporal_object_per_sweep_mode': float(
            temporal_mode == 'object_per_sweep'
        ),
        'temporal_track_shared_mode': float(
            temporal_mode == 'track_shared'
        ),
    })
    original_measurements = cartesian_to_radar_measurement(source_xyz)
    budget = _measurement_budget(
        original_measurements,
        epsilon_range,
        epsilon_azimuth,
        epsilon_elevation,
    )
    if not attack_mask.any() or group_count == 0 or not (budget > 0).any():
        return _empty_attack_output(original, voxelizer, stats, loss_stats)
    steps = 1 if attack_type == 'fgsm' else pgd_steps
    step = _measurement_step(
        budget,
        steps,
        attack_type,
        step_size_range,
        step_size_azimuth,
        step_size_elevation,
    )
    group_delta = original.new_zeros((group_count, 3))
    backtrack_points = 0.0
    backtrack_groups = 0.0
    rejected_groups = 0.0
    zero_gradient_steps = 0.0
    if attack_type == 'pgd' and random_start:
        candidate = (
            torch.rand_like(group_delta) * 2.0 - 1.0
        ) * budget
        group_delta, projection_stats = project_temporal_group_delta(
            candidate,
            original[:, 1:4],
            original_measurements,
            rotations,
            translations,
            group_ids,
            attack_mask,
            active_groups,
            budget,
            voxelizer.point_cloud_range,
        )
        backtrack_points += projection_stats[
            'temporal_out_of_range_backtrack_points'
        ]
        backtrack_groups += projection_stats[
            'temporal_out_of_range_backtrack_groups'
        ]
        rejected_groups += projection_stats[
            'temporal_out_of_range_rejected_groups'
        ]

    states = set_attack_mode(model)
    try:
        for _ in range(steps):
            group_delta = group_delta.detach().requires_grad_(True)
            reference_xyz, _ = _temporal_reference_xyz(
                original[:, 1:4],
                original_measurements,
                rotations,
                translations,
                group_delta,
                group_ids,
                attack_mask,
            )
            adversarial_points = torch.cat((
                original[:, :1], reference_xyz, original[:, 4:]
            ), dim=1)
            topology = voxelizer.topology(adversarial_points)
            voxel_data = voxelizer.materialize(adversarial_points, topology)
            attack_batch = dict(batch_dict)
            attack_batch['points'] = adversarial_points
            attack_batch.update(voxel_data)
            ret_dict, _, _ = model(attack_batch)
            loss = (
                ret_dict['loss'].mean()
                if loss_fn is None
                else loss_fn(model).mean()
            )
            gradient = torch.autograd.grad(loss, group_delta)[0]
            active_gradient = gradient[active_groups]
            if not torch.isfinite(active_gradient).all():
                raise RuntimeError(
                    'non-finite temporal measurement gradient'
                )
            if not active_gradient.abs().any():
                zero_gradient_steps += 1.0
            candidate = group_delta + step * gradient.sign()
            group_delta, projection_stats = project_temporal_group_delta(
                candidate,
                original[:, 1:4],
                original_measurements,
                rotations,
                translations,
                group_ids,
                attack_mask,
                active_groups,
                budget,
                voxelizer.point_cloud_range,
            )
            backtrack_points += projection_stats[
                'temporal_out_of_range_backtrack_points'
            ]
            backtrack_groups += projection_stats[
                'temporal_out_of_range_backtrack_groups'
            ]
            rejected_groups += projection_stats[
                'temporal_out_of_range_rejected_groups'
            ]
    finally:
        restore_attack_modes(states)
        model.zero_grad(set_to_none=True)

    group_delta = group_delta.detach()
    final_xyz, adversarial_measurements = _temporal_reference_xyz(
        original[:, 1:4],
        original_measurements,
        rotations,
        translations,
        group_delta,
        group_ids,
        attack_mask,
    )
    adversarial = torch.cat(
        (original[:, :1], final_xyz, original[:, 4:]), dim=1
    ).detach()
    final_topology = voxelizer.topology(adversarial)
    voxel_data = voxelizer.materialize(adversarial, final_topology)

    xyz_delta = adversarial[:, 1:4] - original[:, 1:4]
    xyz_l2 = torch.linalg.vector_norm(xyz_delta, dim=1)
    modified = xyz_delta.abs().any(dim=1)
    temporal_target = group_ids >= 0
    non_target_modifications = modified & ~temporal_target
    non_geometry_modifications = (
        adversarial[:, 4:] != original[:, 4:]
    ).any(dim=1)
    if non_target_modifications.any():
        raise RuntimeError('temporal attack modified a non-target point')
    if non_geometry_modifications.any():
        raise RuntimeError('temporal attack modified a non-geometry feature')
    point_measurement_delta = adversarial_measurements - original_measurements
    absolute_measurement_delta = point_measurement_delta.abs()[attack_mask]
    xyz_components = xyz_delta.abs()[attack_mask]
    current = sweep_ids == 0
    stats.update({
        'max_abs_perturbation': (
            xyz_components.max().item() if xyz_components.numel() else 0.0
        ),
        'mean_abs_perturbation': (
            xyz_components.mean().item() if xyz_components.numel() else 0.0
        ),
        'sum_abs_perturbation': float(xyz_components.sum().item()),
        'perturbation_values': float(xyz_components.numel()),
        'measurement_max_abs_delta_range': float(
            absolute_measurement_delta[:, 0].max().item()
        ),
        'measurement_max_abs_delta_azimuth_rad': float(
            absolute_measurement_delta[:, 1].max().item()
        ),
        'measurement_max_abs_delta_elevation_rad': float(
            absolute_measurement_delta[:, 2].max().item()
        ),
        'measurement_max_xyz_l2': float(xyz_l2[attack_mask].max().item()),
        'measurement_xyz_l2_sum': float(xyz_l2[attack_mask].sum().item()),
        'measurement_modified_current_points': float(
            (modified & current).sum().item()
        ),
        'measurement_historical_modification_count': float(
            (modified & ~current).sum().item()
        ),
        'measurement_non_target_modification_count': float(
            non_target_modifications.sum().item()
        ),
        'measurement_non_geometry_modification_count': float(
            non_geometry_modifications.sum().item()
        ),
        'measurement_out_of_range_backtracks': backtrack_points,
        'measurement_out_of_range_rejections': rejected_groups,
        'measurement_nonfinite_gradient_steps': 0.0,
        'measurement_zero_gradient_steps': zero_gradient_steps,
        'temporal_out_of_range_backtrack_groups': backtrack_groups,
        'temporal_out_of_range_rejected_groups': rejected_groups,
        'temporal_shared_delta_max_abs': float(
            group_delta[active_groups].abs().max().item()
        ),
        'temporal_parameter_delta_max_abs': float(
            group_delta[active_groups].abs().max().item()
        ),
    })
    clean_range = original_measurements[:, 0]
    for lower, upper, suffix in DISTANCE_BINS_METRES:
        bin_mask = attack_mask & (clean_range >= lower)
        if upper != float('inf'):
            bin_mask &= clean_range < upper
        stats[f'measurement_xyz_l2_count_{suffix}'] = float(
            bin_mask.sum().item()
        )
        stats[f'measurement_xyz_l2_sum_{suffix}'] = float(
            xyz_l2[bin_mask].sum().item()
        )
    return AttackOutput(adversarial, voxel_data, stats)
