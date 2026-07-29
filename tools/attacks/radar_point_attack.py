"""Backward-compatible imports for the modular 4D-radar attack package.

New code should import from ``radar_attack``.  The tuple return value is kept
here so existing experiments that imported this module continue to work.
"""

try:
    from radar_attack.adapters.openpcdet import (
        PointCloudVoxelizer,
        VoxelTopology,
        get_feature_names,
        get_voxel_settings,
    )
    from radar_attack.attacks.gradient import (
        build_budget,
        point_cloud_attack as _point_cloud_attack,
        project_points,
    )
except ModuleNotFoundError:
    from tools.radar_attack.adapters.openpcdet import (
        PointCloudVoxelizer,
        VoxelTopology,
        get_feature_names,
        get_voxel_settings,
    )
    from tools.radar_attack.attacks.gradient import (
        build_budget,
        point_cloud_attack as _point_cloud_attack,
        project_points,
    )


def point_cloud_attack(*args, **kwargs):
    output = _point_cloud_attack(*args, **kwargs)
    return output.adv_points, output.model_inputs, output.stats


__all__ = [
    'PointCloudVoxelizer',
    'VoxelTopology',
    'build_budget',
    'get_feature_names',
    'get_voxel_settings',
    'point_cloud_attack',
    'project_points',
]
