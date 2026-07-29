"""Point-level attacks for hard-voxelized 4D-radar PointPillars models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class VoxelTopology:
    point_indices: torch.Tensor
    point_mask: torch.Tensor
    voxel_coords: torch.Tensor
    voxel_num_points: torch.Tensor


class PointCloudVoxelizer:
    """Hard voxelization with differentiable point-feature gathering.

    Voxel assignment is discrete. During backward, gradients pass through the
    gather to the selected raw radar points, which is a BPDA approximation.
    """

    def __init__(
        self,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        max_points_per_voxel: int,
        max_voxels: int,
        batch_size: int,
    ):
        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.voxel_size = tuple(float(x) for x in voxel_size)
        self.max_points_per_voxel = int(max_points_per_voxel)
        self.max_voxels = int(max_voxels)
        self.batch_size = int(batch_size)

    def topology(self, points: torch.Tensor) -> VoxelTopology:
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError('points must have shape [N, batch_idx + point features]')

        device = points.device
        range_min = points.new_tensor(self.point_cloud_range[:3])
        range_max = points.new_tensor(self.point_cloud_range[3:])
        voxel_size = points.new_tensor(self.voxel_size)

        with torch.no_grad():
            xyz = points[:, 1:4]
            coords_xyz = torch.floor((xyz - range_min) / voxel_size).long()
            batch_indices = points[:, 0].long()
            valid = ((xyz >= range_min) & (xyz < range_max)).all(dim=1)
            valid &= (batch_indices >= 0) & (batch_indices < self.batch_size)
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()

            coords_cpu = coords_xyz[valid_indices].cpu().tolist()
            batches_cpu = batch_indices[valid_indices].cpu().tolist()
            indices_cpu = valid_indices.cpu().tolist()

        voxel_maps: List[Dict[Tuple[int, int, int], int]] = [
            {} for _ in range(self.batch_size)
        ]
        voxel_points: List[List[int]] = []
        voxel_coords: List[Tuple[int, int, int, int]] = []

        for point_index, batch_index, coord_xyz in zip(
            indices_cpu, batches_cpu, coords_cpu
        ):
            x_index, y_index, z_index = coord_xyz
            key = (z_index, y_index, x_index)
            sample_map = voxel_maps[batch_index]
            voxel_index = sample_map.get(key)

            if voxel_index is None:
                if len(sample_map) >= self.max_voxels:
                    continue
                voxel_index = len(voxel_points)
                sample_map[key] = voxel_index
                voxel_points.append([])
                voxel_coords.append((batch_index, z_index, y_index, x_index))

            if len(voxel_points[voxel_index]) < self.max_points_per_voxel:
                voxel_points[voxel_index].append(point_index)

        if not voxel_points:
            raise RuntimeError('point attack produced no voxels inside POINT_CLOUD_RANGE')

        num_voxels = len(voxel_points)
        point_indices = torch.zeros(
            (num_voxels, self.max_points_per_voxel),
            dtype=torch.long,
            device=device,
        )
        point_mask = torch.zeros_like(point_indices, dtype=torch.bool)
        voxel_num_points = torch.empty(
            num_voxels, dtype=torch.int32, device=device
        )

        for voxel_index, indices in enumerate(voxel_points):
            count = len(indices)
            point_indices[voxel_index, :count] = torch.as_tensor(
                indices, dtype=torch.long, device=device
            )
            point_mask[voxel_index, :count] = True
            voxel_num_points[voxel_index] = count

        return VoxelTopology(
            point_indices=point_indices,
            point_mask=point_mask,
            voxel_coords=torch.as_tensor(
                voxel_coords, dtype=torch.int32, device=device
            ),
            voxel_num_points=voxel_num_points,
        )

    @staticmethod
    def materialize(
        points: torch.Tensor, topology: VoxelTopology
    ) -> Dict[str, torch.Tensor]:
        voxels = points[topology.point_indices, 1:]
        voxels = voxels * topology.point_mask.unsqueeze(-1).to(voxels.dtype)
        return {
            'voxels': voxels.contiguous(),
            'voxel_coords': topology.voxel_coords,
            'voxel_num_points': topology.voxel_num_points,
        }


def get_voxel_settings(dataset_cfg):
    for processor in dataset_cfg.DATA_PROCESSOR:
        if processor.NAME == 'transform_points_to_voxels':
            max_voxels = processor.MAX_NUMBER_OF_VOXELS
            max_voxels = (
                max_voxels['test'] if isinstance(max_voxels, dict)
                else max_voxels.test
            )
            return (
                processor.VOXEL_SIZE,
                int(processor.MAX_POINTS_PER_VOXEL),
                int(max_voxels),
            )
    raise ValueError('point attack requires transform_points_to_voxels')


def get_feature_names(dataset_cfg) -> List[str]:
    names = list(dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)
    if names[:3] != ['x', 'y', 'z']:
        raise ValueError(f'point features must begin with x, y, z; got {names}')
    return names


def _feature_group(name: str) -> str:
    name = name.lower()
    if name in {'x', 'y', 'z'}:
        return 'xyz'
    if name in {'rcs', 'intensity', 'power'}:
        return 'rcs'
    if name in {
        'v_r', 'v_r_comp', 'velocity', 'velocity_comp',
        'doppler', 'doppler_comp',
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
    voxelizer: PointCloudVoxelizer,
    feature_names: Sequence[str],
    attack_type: str,
    attack_feature: str,
    epsilon: float,
    epsilon_overrides: Optional[Dict[str, Optional[float]]] = None,
    pgd_steps: int = 10,
    step_size: Optional[float] = None,
    random_start: bool = False,
    voxel_mode: str = 'fixed',
):
    """Return adversarial raw points, corresponding voxels, and statistics."""
    original = batch_dict['points'].detach().clone()
    budget = build_budget(
        original, feature_names, attack_feature, epsilon, epsilon_overrides
    )
    steps = 1 if attack_type == 'fgsm' else pgd_steps
    step = (
        budget if attack_type == 'fgsm'
        else budget * (2.0 / steps)
    )
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
    }
    return adversarial, voxel_data, stats
