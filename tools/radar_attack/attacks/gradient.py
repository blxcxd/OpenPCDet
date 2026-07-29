"""FGSM/PGD-style attacks on raw 4D-radar points."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .base import AttackOutput


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
    projected[:, 1:4] = torch.maximum(
        torch.minimum(projected[:, 1:4], spatial_max - margin), spatial_min
    )

    if voxel_mode == 'fixed':
        size = projected.new_tensor(voxel_size)
        original_cell = torch.floor((original[:, 1:4] - spatial_min) / size)
        cell_min = spatial_min + original_cell * size
        projected[:, 1:4] = torch.maximum(
            torch.minimum(projected[:, 1:4], cell_min + size - margin),
            cell_min,
        )
    return projected


def _set_attack_mode(model: nn.Module) -> Dict[nn.Module, bool]:
    states = {module: module.training for module in model.modules()}
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return states


def _restore_modes(states: Dict[nn.Module, bool]) -> None:
    for module, training in states.items():
        module.training = training


def point_cloud_attack(
    model: nn.Module,
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

    states = _set_attack_mode(model)
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
            loss = ret_dict['loss'].mean()
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
        _restore_modes(states)
        model.zero_grad(set_to_none=True)

    adversarial = adversarial.detach()
    final_topology = (
        fixed_topology
        if fixed_topology is not None
        else voxelizer.topology(adversarial)
    )
    voxel_data = voxelizer.materialize(adversarial, final_topology)

    attacked_columns = budget.squeeze(0) > 0
    delta = (adversarial - original).abs()[:, attacked_columns]
    stats = {
        'max_abs_perturbation': delta.max().item() if delta.numel() else 0.0,
        'mean_abs_perturbation': delta.mean().item() if delta.numel() else 0.0,
        'sum_abs_perturbation': delta.sum().item() if delta.numel() else 0.0,
        'perturbation_values': float(delta.numel()),
    }
    return AttackOutput(
        adv_points=adversarial,
        model_inputs=voxel_data,
        stats=stats,
    )
