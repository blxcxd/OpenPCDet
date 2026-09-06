import torch

from .pillar_vfe import PFNLayer
from .vfe_template import VFETemplate


class RadarPillarsVFE(VFETemplate):
    """RadarPillars pillar encoder for VoD's seven-channel radar points.

    The input feature order is ``[x, y, z, rcs, v_r, v_r_comp, time]``.
    RadarPillars decomposes the ego-motion-compensated radial velocity into
    Cartesian x/y components before applying the PointPillars PFN.

    This is a separate VFE instead of a modification to ``Radar7PillarVFE`` so
    existing PointPillars-radar experiments retain their original behaviour.
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = model_cfg.USE_NORM
        self.with_distance = model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = model_cfg.get('USE_ABSOLUTE_XYZ', True)
        self.use_velocity_decomposition = model_cfg.get('USE_VELOCITY_DECOMPOSITION', True)
        self.velocity_comp_index = model_cfg.get('VELOCITY_COMP_INDEX', 5)

        if num_point_features <= self.velocity_comp_index:
            raise ValueError(
                f'VELOCITY_COMP_INDEX={self.velocity_comp_index} is invalid for '
                f'{num_point_features} input features'
            )

        pfn_input_features = num_point_features
        if self.use_velocity_decomposition:
            pfn_input_features += 2
        pfn_input_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            pfn_input_features += 1

        self.num_filters = model_cfg.NUM_FILTERS
        if not self.num_filters:
            raise ValueError('RadarPillarsVFE requires at least one NUM_FILTERS value')

        num_filters = [pfn_input_features] + list(self.num_filters)
        self.pfn_layers = torch.nn.ModuleList([
            PFNLayer(
                num_filters[i], num_filters[i + 1], self.use_norm,
                last_layer=(i == len(num_filters) - 2)
            )
            for i in range(len(num_filters) - 1)
        ])

        self.voxel_x, self.voxel_y, self.voxel_z = voxel_size
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        return actual_num.int() > max_num

    def forward(self, batch_dict, **kwargs):
        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']

        xyz = voxel_features[:, :, :3]
        points_mean = xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(
            voxel_features
        ).view(-1, 1, 1).clamp(min=1)
        f_cluster = xyz - points_mean

        f_center = torch.zeros_like(xyz)
        f_center[:, :, 0] = xyz[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        )
        f_center[:, :, 1] = xyz[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        )
        f_center[:, :, 2] = xyz[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        )

        point_features = voxel_features
        if self.use_velocity_decomposition:
            azimuth = torch.atan2(xyz[:, :, 1], xyz[:, :, 0] + 1e-6)
            radial_velocity = voxel_features[:, :, self.velocity_comp_index]
            cartesian_velocity = torch.stack([
                radial_velocity * torch.cos(azimuth),
                radial_velocity * torch.sin(azimuth),
            ], dim=-1)
            point_features = torch.cat([point_features, cartesian_velocity], dim=-1)

        if self.use_absolute_xyz:
            features = [point_features, f_cluster, f_center]
        else:
            features = [point_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            features.append(torch.norm(xyz, p=2, dim=2, keepdim=True))
        features = torch.cat(features, dim=-1)

        mask = self.get_paddings_indicator(voxel_num_points, features.shape[1])
        features *= mask.unsqueeze(-1).type_as(features)
        for pfn in self.pfn_layers:
            features = pfn(features)

        batch_dict['pillar_features'] = features.squeeze(dim=1)
        return batch_dict
