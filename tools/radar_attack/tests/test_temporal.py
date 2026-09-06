import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tools.radar_attack.adapters.openpcdet import PointCloudVoxelizer
from tools.radar_attack.attacks.measurement import (
    build_temporal_measurement_attack_mask,
    build_temporal_parameter_groups,
    cartesian_to_radar_measurement,
    radar_temporal_measurement_geometry_attack,
)
from tools.radar_attack.temporal import (
    TemporalBatchData,
    TemporalSweepResolver,
    TrackedBox,
    apply_rigid_transform,
    assign_points_to_tracked_boxes,
    fit_rigid_transform,
)


class SumVoxelModel(nn.Module):
    def forward(self, batch_dict):
        return {'loss': batch_dict['voxels'].sum()}, {}, {}


class TemporalRadarTest(unittest.TestCase):
    @staticmethod
    def _transform(angle=0.2, translation=(1.0, -0.5, 0.2)):
        cosine, sine = np.cos(angle), np.sin(angle)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform[:3, 3] = translation
        return transform

    @staticmethod
    def _single_points(offset=0.0):
        return np.asarray([
            [1.0, 0.0, 0.0, 10.0 + offset, 1.0, 0.5, 0.0],
            [0.0, 2.0, 0.0, 11.0 + offset, 2.0, 0.6, 0.0],
            [0.5, 0.5, 1.0, 12.0 + offset, 3.0, 0.7, 0.0],
            [2.0, 1.0, -0.5, 13.0 + offset, 4.0, 0.8, 0.0],
        ], dtype=np.float32)

    def test_rigid_transform_recovery(self):
        source = self._single_points()[:, :3]
        expected = self._transform()
        target = apply_rigid_transform(source, expected)

        actual, residuals = fit_rigid_transform(source, target)

        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertLess(float(residuals.max()), 1e-12)
        self.assertAlmostEqual(np.linalg.det(actual[:3, :3]), 1.0)

    def test_resolver_cache_and_zero_delta_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accumulated_dir = root / 'radar_5frames/training/velodyne'
            single_dir = root / 'radar/training/velodyne'
            accumulated_dir.mkdir(parents=True)
            single_dir.mkdir(parents=True)
            previous = self._single_points()
            current = self._single_points(offset=20.0)
            history_transform = self._transform()
            accumulated_previous = previous.copy()
            accumulated_previous[:, :3] = apply_rigid_transform(
                previous[:, :3], history_transform
            )
            accumulated_previous[:, 6] = -1.0
            accumulated_current = current.copy()
            accumulated = np.concatenate(
                (accumulated_previous, accumulated_current), axis=0
            )
            previous.tofile(single_dir / '00000.bin')
            current.tofile(single_dir / '00001.bin')
            accumulated.tofile(accumulated_dir / '00001.bin')
            cache_dir = root / 'cache'
            resolver = TemporalSweepResolver(
                root, cache_dir=cache_dir, max_residual_m=1e-4
            )

            first = resolver.resolve('00001')
            diagnostics = first.zero_delta_diagnostics()

            self.assertFalse(first.cache_hit)
            self.assertEqual([item.sweep_id for item in first.sweeps], [-1, 0])
            self.assertTrue(bool(diagnostics['zero_mask_exact']))
            self.assertEqual(diagnostics['non_xyz_change_count'], 0.0)
            self.assertLess(diagnostics['measurement_roundtrip_max_m'], 1e-5)
            np.testing.assert_array_equal(
                first.reconstruct_points(
                    first.aligned_source_xyz(),
                    np.zeros(len(accumulated), dtype=np.bool_),
                ),
                accumulated,
            )
            subset_indices = np.asarray([1, 4, 7])
            np.testing.assert_array_equal(
                first.align_reference_subset(accumulated[subset_indices]),
                subset_indices,
            )
            second = resolver.resolve(1)
            self.assertTrue(second.cache_hit)
            np.testing.assert_allclose(
                second.sweeps[0].source_to_reference,
                history_transform,
                atol=1e-6,
            )

    def test_resolver_rejects_changed_source_feature_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accumulated_dir = root / 'radar_5frames/training/velodyne'
            single_dir = root / 'radar/training/velodyne'
            accumulated_dir.mkdir(parents=True)
            single_dir.mkdir(parents=True)
            source = self._single_points()
            accumulated = source.copy()
            accumulated[[0, 1]] = accumulated[[1, 0]]
            source.tofile(single_dir / '00000.bin')
            accumulated.tofile(accumulated_dir / '00000.bin')
            resolver = TemporalSweepResolver(root)

            with self.assertRaisesRegex(ValueError, 'point order'):
                resolver.resolve('00000')

    def test_track_assignment_uses_normalized_box_distance(self):
        boxes = (
            TrackedBox(
                frame_id='00001', track_id=10, class_name='Car', class_id=1,
                box=np.asarray([0, 0, 0, 4, 2, 2, 0], dtype=np.float32),
            ),
            TrackedBox(
                frame_id='00001', track_id=20, class_name='Pedestrian',
                class_id=2,
                box=np.asarray([1, 0, 0, 2, 2, 2, 0], dtype=np.float32),
            ),
        )
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])

        tracks, classes = assign_points_to_tracked_boxes(points, boxes)

        np.testing.assert_array_equal(tracks, np.asarray([10, 20, -1]))
        np.testing.assert_array_equal(classes, np.asarray([1, 2, -1]))

    def test_track_shared_attack_uses_one_delta_across_sweeps(self):
        points = torch.tensor([
            [0, 5.0, 0.0, 0.0, 1.0, 0.2, 0.1, -1.0],
            [0, 6.0, 0.0, 0.0, 2.0, 0.3, 0.2, 0.0],
            [0, 8.0, 1.0, 0.0, 3.0, 0.4, 0.3, 0.0],
            [0, 10.0, 2.0, 0.0, 4.0, 0.5, 0.4, -1.0],
        ], dtype=torch.float32)
        count = len(points)
        temporal = TemporalBatchData(
            source_xyz=points[:, 1:4].numpy().astype(np.float64),
            rotations=np.repeat(
                np.eye(3, dtype=np.float64)[None], count, axis=0
            ),
            translations=np.zeros((count, 3), dtype=np.float64),
            group_ids=np.asarray([0, 0, 1, -1]),
            track_ids=np.asarray([10, 10, 20, -1]),
            gt_rows=np.asarray([0, 0, 1, -1]),
            sweep_ids=np.asarray([-1, 0, 0, -1]),
            diagnostics={'temporal_shared_groups': 2.0},
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -10, -3, 20, 10, 3],
            voxel_size=[1, 1, 6],
            max_points_per_voxel=10,
            max_voxels=100,
            batch_size=1,
        )
        attack_mask, mask_stats = build_temporal_measurement_attack_mask(
            points, temporal, voxelizer
        )

        output = radar_temporal_measurement_geometry_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            temporal_data=temporal,
            attack_type='fgsm',
            epsilon_range=0.1,
            epsilon_azimuth=0.0,
            epsilon_elevation=0.0,
            attack_mask=attack_mask,
            attack_mask_stats=mask_stats,
        )

        clean_measurement = cartesian_to_radar_measurement(points[:, 1:4])
        adversarial_measurement = cartesian_to_radar_measurement(
            output.adv_points[:, 1:4]
        )
        delta = adversarial_measurement - clean_measurement
        self.assertAlmostEqual(delta[0, 0].item(), delta[1, 0].item(), places=6)
        self.assertAlmostEqual(delta[0, 0].item(), 0.1, places=5)
        self.assertTrue(torch.equal(output.adv_points[3], points[3]))
        self.assertTrue(torch.equal(output.adv_points[:, 4:], points[:, 4:]))
        self.assertEqual(
            output.stats['measurement_historical_modification_count'], 1.0
        )
        self.assertEqual(
            output.stats['measurement_non_target_modification_count'], 0.0
        )

    def test_temporal_parameter_grouping_modes(self):
        temporal = TemporalBatchData(
            source_xyz=np.zeros((6, 3), dtype=np.float64),
            rotations=np.repeat(np.eye(3)[None], 6, axis=0),
            translations=np.zeros((6, 3)),
            group_ids=np.asarray([0, 0, 0, 1, 1, -1]),
            track_ids=np.asarray([10, 10, 10, 20, 20, -1]),
            gt_rows=np.asarray([0, 0, 0, 1, 1, -1]),
            sweep_ids=np.asarray([-1, 0, 0, -1, -1, 0]),
            diagnostics={},
        )

        point_groups, point_count = build_temporal_parameter_groups(
            temporal, 'point_independent'
        )
        sweep_groups, sweep_count = build_temporal_parameter_groups(
            temporal, 'object_per_sweep'
        )
        track_groups, track_count = build_temporal_parameter_groups(
            temporal, 'track_shared'
        )
        scene_groups, scene_count = build_temporal_parameter_groups(
            temporal, 'point_independent', point_scope='scene'
        )

        self.assertEqual(point_count, 5)
        self.assertEqual(sweep_count, 3)
        self.assertEqual(track_count, 2)
        self.assertEqual(scene_count, 6)
        self.assertEqual(point_groups.tolist(), [0, 1, 2, 3, 4, -1])
        self.assertNotEqual(sweep_groups[0].item(), sweep_groups[1].item())
        self.assertEqual(sweep_groups[1].item(), sweep_groups[2].item())
        self.assertEqual(sweep_groups[3].item(), sweep_groups[4].item())
        self.assertEqual(track_groups.tolist(), [0, 0, 0, 1, 1, -1])
        self.assertEqual(scene_groups.tolist(), [0, 1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, 'point_independent'):
            build_temporal_parameter_groups(
                temporal, 'track_shared', point_scope='scene'
            )

    def test_scene_point_independent_attack_includes_untracked_points(self):
        points = torch.tensor([
            [0, 5.0, 0.0, 0.0, 1.0, 0.2, 0.1, -1.0],
            [0, 6.0, 0.0, 0.0, 2.0, 0.3, 0.2, 0.0],
            [0, 8.0, 0.0, 0.0, 3.0, 0.4, 0.3, 0.0],
        ], dtype=torch.float32)
        temporal = TemporalBatchData(
            source_xyz=points[:, 1:4].numpy().astype(np.float64),
            rotations=np.repeat(np.eye(3)[None], 3, axis=0),
            translations=np.zeros((3, 3)),
            group_ids=np.asarray([0, 0, -1]),
            track_ids=np.asarray([10, 10, -1]),
            gt_rows=np.asarray([0, 0, -1]),
            sweep_ids=np.asarray([-1, 0, 0]),
            diagnostics={'temporal_shared_groups': 1.0},
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -10, -3, 20, 10, 3],
            voxel_size=[1, 1, 6],
            max_points_per_voxel=10,
            max_voxels=100,
            batch_size=1,
        )
        attack_mask, mask_stats = build_temporal_measurement_attack_mask(
            points, temporal, voxelizer, point_scope='scene'
        )

        output = radar_temporal_measurement_geometry_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            temporal_data=temporal,
            temporal_mode='point_independent',
            point_scope='scene',
            attack_type='fgsm',
            epsilon_range=0.1,
            epsilon_azimuth=0.0,
            epsilon_elevation=0.0,
            attack_mask=attack_mask,
            attack_mask_stats=mask_stats,
        )

        self.assertEqual(attack_mask.tolist(), [True, True, True])
        self.assertEqual(output.stats['temporal_parameter_groups'], 3.0)
        self.assertEqual(output.stats['measurement_scene_scope'], 1.0)
        self.assertNotEqual(output.adv_points[2, 1].item(), points[2, 1].item())
        self.assertTrue(torch.equal(output.adv_points[:, 4:], points[:, 4:]))

    def test_object_per_sweep_attack_reports_parameter_groups(self):
        points = torch.tensor([
            [0, 5.0, 0.0, 0.0, 1.0, 0.2, 0.1, -1.0],
            [0, 6.0, 0.0, 0.0, 2.0, 0.3, 0.2, 0.0],
            [0, 7.0, 0.0, 0.0, 3.0, 0.4, 0.3, 0.0],
            [0, 9.0, 0.0, 0.0, 4.0, 0.5, 0.4, 0.0],
        ], dtype=torch.float32)
        temporal = TemporalBatchData(
            source_xyz=points[:, 1:4].numpy().astype(np.float64),
            rotations=np.repeat(np.eye(3)[None], 4, axis=0),
            translations=np.zeros((4, 3)),
            group_ids=np.asarray([0, 0, 0, -1]),
            track_ids=np.asarray([10, 10, 10, -1]),
            gt_rows=np.asarray([0, 0, 0, -1]),
            sweep_ids=np.asarray([-1, 0, 0, 0]),
            diagnostics={'temporal_shared_groups': 1.0},
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -10, -3, 20, 10, 3],
            voxel_size=[1, 1, 6],
            max_points_per_voxel=10,
            max_voxels=100,
            batch_size=1,
        )

        output = radar_temporal_measurement_geometry_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            temporal_data=temporal,
            temporal_mode='object_per_sweep',
            attack_type='fgsm',
            epsilon_range=0.1,
            epsilon_azimuth=0.0,
            epsilon_elevation=0.0,
        )

        self.assertEqual(output.stats['temporal_parameter_groups'], 2.0)
        self.assertEqual(
            output.stats['temporal_active_parameter_groups'], 2.0
        )
        self.assertEqual(output.stats['temporal_object_per_sweep_mode'], 1.0)
        self.assertTrue(torch.equal(output.adv_points[3], points[3]))
        self.assertTrue(torch.equal(output.adv_points[:, 4:], points[:, 4:]))

    def test_zero_budget_track_shared_attack_is_bit_exact(self):
        points = torch.tensor([
            [0, 5.0, 0.0, 0.0, 1.0, 0.2, 0.1, -1.0],
            [0, 6.0, 0.0, 0.0, 2.0, 0.3, 0.2, 0.0],
        ], dtype=torch.float32)
        temporal = TemporalBatchData(
            source_xyz=points[:, 1:4].numpy().astype(np.float64),
            rotations=np.repeat(np.eye(3)[None], 2, axis=0),
            translations=np.zeros((2, 3)),
            group_ids=np.asarray([0, 0]),
            track_ids=np.asarray([10, 10]),
            gt_rows=np.asarray([0, 0]),
            sweep_ids=np.asarray([-1, 0]),
            diagnostics={},
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -10, -3, 20, 10, 3],
            voxel_size=[1, 1, 6],
            max_points_per_voxel=10,
            max_voxels=100,
            batch_size=1,
        )

        output = radar_temporal_measurement_geometry_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            temporal_data=temporal,
            attack_type='pgd',
            epsilon_range=0.0,
            epsilon_azimuth=0.0,
            epsilon_elevation=0.0,
            pgd_steps=2,
        )

        self.assertTrue(torch.equal(output.adv_points, points))


if __name__ == '__main__':
    unittest.main()
