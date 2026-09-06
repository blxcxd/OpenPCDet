"""OpenPCDet adapter for differentiable hard voxelization."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class VoxelTopology:
    point_indices: torch.Tensor
    point_mask: torch.Tensor
    voxel_coords: torch.Tensor
    voxel_num_points: torch.Tensor


class PointCloudVoxelizer:
    """Hard voxelization with differentiable point-feature gathering.

    Voxel assignment is discrete. During backward, gradients pass through the
    feature gather to selected raw radar points as a BPDA approximation.
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

        if len(self.point_cloud_range) != 6 or len(self.voxel_size) != 3:
            raise ValueError(
                'point_cloud_range must contain 6 values and voxel_size 3 values'
            )
        if self.max_points_per_voxel <= 0 or self.max_voxels <= 0:
            raise ValueError('voxel capacity values must be positive')
        self.grid_size = tuple(
            int(round(
                (self.point_cloud_range[index + 3]
                 - self.point_cloud_range[index])
                / self.voxel_size[index]
            ))
            for index in range(3)
        )
        if any(size <= 0 for size in self.grid_size):
            raise ValueError('voxel grid dimensions must be positive')

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
            # A float32 coordinate just below range_max can round to the
            # first out-of-grid integer after division (for example,
            # 25.599998 / 0.16 -> 320). Match the detector grid explicitly
            # so PointPillarScatter never receives an invalid coordinate.
            grid_size = torch.as_tensor(
                self.grid_size, dtype=torch.long, device=device
            )
            valid &= (
                (coords_xyz >= 0) & (coords_xyz < grid_size)
            ).all(dim=1)
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
                max_voxels['test']
                if isinstance(max_voxels, dict)
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
