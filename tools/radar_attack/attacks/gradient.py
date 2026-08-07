"""FGSM/PGD-style attacks on raw 4D-radar points."""

from typing import Callable, Dict, Optional, Sequence

import torch

from .base import AttackOutput, restore_attack_modes, set_attack_mode


def _feature_group(name: str) -> str:
    name = name.lower()
    if name in {'x', 'y', 'z'}:
        return 'xyz'
    if name in {'rcs', 'intensity', 'power'}:
        return 'rcs'
    if name in {
        'v_r',
        'v_r_comp',
        'velocity',
        'velocity_comp',
        'doppler',
        'doppler_comp',
    }:
        return 'doppler'
    if name in {'time', 'timestamp'}:
        return 'time'
    return 'other'


def build_budget(
    points: torch.Tensor,
    feature_names: Sequence[str],
    attack_feature: str,
    epsilon: float,
    epsilon_overrides: Optional[Dict[str, Optional[float]]] = None,
) -> torch.Tensor:
    if points.shape[1] != len(feature_names) + 1:
        raise ValueError(
            f'points contain {points.shape[1] - 1} features but config declares '
            f'{len(feature_names)}'
        )

    selected_group = 'rcs' if attack_feature == 'intensity' else attack_feature
    overrides = epsilon_overrides or {}
    budget = points.new_zeros((1, points.shape[1]))
    selected = 0

    for column, name in enumerate(feature_names, start=1):
        group = _feature_group(name)
        if selected_group == 'all' or group == selected_group:
            override = overrides.get(group)
            budget[0, column] = epsilon if override is None else override
            selected += 1

    if selected == 0:
        raise ValueError(
            f'feature group "{attack_feature}" is absent from {list(feature_names)}'
        )
    budget = torch.where(
        budget > 0,
        torch.nextafter(budget, torch.zeros_like(budget)),
        budget,
    )
    return budget


def project_points(
    candidate: torch.Tensor,
    original: torch.Tensor,
    budget: torch.Tensor,
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
    voxel_mode: str,
) -> torch.Tensor:
    projected = torch.maximum(
        torch.minimum(candidate, original + budget), original - budget
    )
    projected[:, 0] = original[:, 0]

    spatial_min = projected.new_tensor(point_cloud_range[:3])
    spatial_max = projected.new_tensor(point_cloud_range[3:])
    margin = max(min(float(x) for x in voxel_size) * 1e-4, 1e-6)
    original_in_range = (
        (original[:, 1:4] >= spatial_min)
        & (original[:, 1:4] < spatial_max)
    ).all(dim=1)
    xyz_budget_active = (budget[:, 1:4] > 0).expand_as(original[:, 1:4])
    bounded_xyz = torch.maximum(
        torch.minimum(projected[:, 1:4], spatial_max - margin), spatial_min
    )
    projected[:, 1:4] = torch.where(
        original_in_range.unsqueeze(1) & xyz_budget_active,
        bounded_xyz,
        original[:, 1:4],
    )

    if voxel_mode == 'fixed':
        size = projected.new_tensor(voxel_size)
        original_cell = torch.floor((original[:, 1:4] - spatial_min) / size)
        cell_min = spatial_min + original_cell * size
        fixed_xyz = torch.maximum(
            torch.minimum(projected[:, 1:4], cell_min + size - margin),
            cell_min,
        )
        projected[:, 1:4] = torch.where(
            original_in_range.unsqueeze(1) & xyz_budget_active,
            fixed_xyz,
            original[:, 1:4],
        )
    for _ in range(8):
        exceeds_budget = (projected - original).abs() > budget
        if not exceeds_budget.any():
            break
        projected = torch.where(
            exceeds_budget,
            torch.nextafter(projected, original),
            projected,
        )
    if ((projected - original).abs() > budget).any():
        raise RuntimeError('could not represent an adversarial point within budget')
    return projected


def point_cloud_attack(
    model: torch.nn.Module,
    batch_dict: Dict,
    voxelizer,
    feature_names: Sequence[str],
    attack_type: str,
    attack_feature: str,
    epsilon: float,
    epsilon_overrides: Optional[Dict[str, Optional[float]]] = None,
    pgd_steps: int = 10,
    step_size: Optional[float] = None,
    random_start: bool = False,
    voxel_mode: str = 'fixed',
    point_mask: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.nn.Module], torch.Tensor]] = None,
    loss_stats: Optional[Dict[str, float]] = None,
) -> AttackOutput:
    """Generate an untargeted adversarial raw radar point cloud."""
    if attack_type not in {'fgsm', 'pgd'}:
        raise ValueError('attack_type must be "fgsm" or "pgd"')
    if voxel_mode not in {'fixed', 'revoxelize'}:
        raise ValueError('voxel_mode must be "fixed" or "revoxelize"')
    if epsilon < 0:
        raise ValueError('epsilon must be non-negative')
    if pgd_steps <= 0:
        raise ValueError('pgd_steps must be positive')
    if step_size is not None and step_size <= 0:
        raise ValueError('step_size must be positive')
    for name, value in (epsilon_overrides or {}).items():
        if value is not None and value < 0:
            raise ValueError(f'{name} epsilon must be non-negative')

    original = batch_dict['points'].detach().clone()
    budget = build_budget(
        original, feature_names, attack_feature, epsilon, epsilon_overrides
    )
    if point_mask is not None:
        if point_mask.shape != (original.shape[0],):
            raise ValueError('point_mask must have shape [N]')
        budget = budget * point_mask.to(
            device=original.device, dtype=original.dtype
        ).unsqueeze(1)
    steps = 1 if attack_type == 'fgsm' else pgd_steps
    step = budget if attack_type == 'fgsm' else budget * (2.0 / steps)
    if step_size is not None:
        step = (budget > 0).to(budget.dtype) * step_size

    fixed_topology = (
        voxelizer.topology(original) if voxel_mode == 'fixed' else None
    )
    adversarial = original.clone()

    if attack_type == 'pgd' and random_start:
        random_delta = (torch.rand_like(original) * 2.0 - 1.0) * budget
        if fixed_topology is not None:
            active_points = torch.zeros(
                original.shape[0], dtype=torch.bool, device=original.device
            )
            active_points[
                fixed_topology.point_indices[fixed_topology.point_mask]
            ] = True
            random_delta *= active_points.unsqueeze(1)
        adversarial = project_points(
            adversarial + random_delta,
            original,
            budget,
            voxelizer.point_cloud_range,
            voxelizer.voxel_size,
            voxel_mode,
        )

    states = set_attack_mode(model)
    try:
        for _ in range(steps):
            adversarial = adversarial.detach().requires_grad_(True)
            topology = (
                fixed_topology
                if fixed_topology is not None
                else voxelizer.topology(adversarial)
            )
            voxel_data = voxelizer.materialize(adversarial, topology)
            attack_batch = dict(batch_dict)
            attack_batch['points'] = adversarial
            attack_batch.update(voxel_data)

            ret_dict, _, _ = model(attack_batch)
            loss = (
                ret_dict['loss'].mean()
                if loss_fn is None
                else loss_fn(model).mean()
            )
            gradient = torch.autograd.grad(loss, adversarial)[0]
            if not torch.isfinite(gradient).all():
                raise RuntimeError('non-finite point gradient during radar attack')

            adversarial = project_points(
                adversarial + step * gradient.sign(),
                original,
                budget,
                voxelizer.point_cloud_range,
                voxelizer.voxel_size,
                voxel_mode,
            )
    finally:
        restore_attack_modes(states)
        model.zero_grad(set_to_none=True)

    adversarial = adversarial.detach()
    final_topology = (
        fixed_topology
        if fixed_topology is not None
        else voxelizer.topology(adversarial)
    )
    voxel_data = voxelizer.materialize(adversarial, final_topology)

    attacked_values = budget > 0
    if attacked_values.shape[0] == 1:
        attacked_values = attacked_values.expand_as(original)
    delta = (adversarial - original).abs()[attacked_values]
    stats = {
        'max_abs_perturbation': delta.max().item() if delta.numel() else 0.0,
        'mean_abs_perturbation': delta.mean().item() if delta.numel() else 0.0,
        'sum_abs_perturbation': delta.sum().item() if delta.numel() else 0.0,
        'perturbation_values': float(delta.numel()),
    }
    if loss_stats is not None:
        stats.update({key: float(value) for key, value in loss_stats.items()})
    return AttackOutput(
        adv_points=adversarial,
        model_inputs=voxel_data,
        stats=stats,
    )
