import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from tools.radar_attack.adapters.openpcdet import PointCloudVoxelizer
from tools.radar_attack.analysis import (
    StreamingFeatureStatistics,
    extract_voxelized_features,
)
from tools.radar_attack.attacks import (
    AttackOutput,
    assign_points_to_oriented_boxes,
    build_iadv_attack_mask,
    build_iadv_groups,
    build_iadv_object_ids,
    build_iou_s_geometry_reference_mask,
    build_measurement_attack_mask,
    build_feature_mask,
    cartesian_to_radar_measurement,
    compute_reflectivity_features,
    extremum_fusion,
    iadv_rcs_attack,
    original_iou_s_detection_loss,
    original_iou_s_point_attack,
    prediction_set_retention,
    radar_geometry_q95_hinge_loss,
    point_cloud_attack,
    points_in_oriented_boxes,
    project_radar_measurements,
    radar_measurement_geometry_attack,
    radar_measurement_to_cartesian,
    symmetric_chamfer_squared,
    voxel_attack,
)
from tools.radar_attack.attacks.gradient import project_points
from tools.radar_attack.evaluation import (
    AdversarialPointCloudWriter,
    DetectionAttackMetrics,
    MeasurementNaturalnessAccumulator,
    MeasurementQ95Reference,
    compare_target_object_endpoints,
    build_target_screening_context,
    merge_target_diagnostics,
    point_measurement_naturalness,
    prepare_prediction_directory,
    summarize_current_sweep,
    target_attack_diagnostics,
    target_correlations,
)
from tools.radar_attack.evaluation.vod import (
    _metric_difference,
    _relative_metric_difference,
    summarize_vod_results,
)
from tools.radar_attack.runner import select_sample_indices


class SumVoxelModel(nn.Module):
    def forward(self, batch_dict):
        self.last_voxels = batch_dict['voxels']
        return {'loss': batch_dict['voxels'].sum()}, {}, {}


class DynamicPredictionModel(nn.Module):
    """Small differentiable post-NMS-like model for original IoU-S tests."""

    def forward(self, batch_dict):
        xyz = batch_dict['points'][:, 1:4]
        center = xyz.mean(dim=0)
        box = torch.cat((center, xyz.new_tensor([1.0, 1.0, 1.0, 0.0])))
        score = torch.sigmoid(xyz.sum().reshape(1))
        return [{
            'pred_boxes': box.reshape(1, 7),
            'pred_scores': score,
            'pred_labels': torch.ones(1, dtype=torch.long, device=xyz.device),
        }], {'gt': 1, 'rcnn_0.5': 1}


class RadarAttackComponentTest(unittest.TestCase):
    @staticmethod
    def _axis_aligned_iou(boxes_a, boxes_b):
        if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
            return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
        minimum_a = boxes_a[:, None, :3] - boxes_a[:, None, 3:6] / 2
        maximum_a = boxes_a[:, None, :3] + boxes_a[:, None, 3:6] / 2
        minimum_b = boxes_b[None, :, :3] - boxes_b[None, :, 3:6] / 2
        maximum_b = boxes_b[None, :, :3] + boxes_b[None, :, 3:6] / 2
        overlap = (torch.minimum(maximum_a, maximum_b) - torch.maximum(minimum_a, minimum_b)).clamp_min(0)
        intersection = overlap.prod(dim=-1)
        volume_a = boxes_a[:, 3:6].prod(dim=-1)[:, None]
        volume_b = boxes_b[:, 3:6].prod(dim=-1)[None, :]
        return intersection / (volume_a + volume_b - intersection).clamp_min(1e-8)

    def test_validation_subset_selection_is_reproducible(self):
        uniform = select_sample_indices(10, 3, 'uniform', seed=7)
        random_first = select_sample_indices(20, 5, 'random', seed=7)
        random_second = select_sample_indices(20, 5, 'random', seed=7)

        np.testing.assert_array_equal(uniform, np.array([1, 5, 8]))
        np.testing.assert_array_equal(random_first, random_second)
        self.assertEqual(len(np.unique(random_first)), 5)

    def test_voxelizer_rejects_float32_rounded_upper_grid_index(self):
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
            voxel_size=[0.16, 0.16, 5],
            max_points_per_voxel=10,
            max_voxels=40000,
            batch_size=1,
        )
        points = torch.tensor([
            [0, 1.0, 0.0, 0.0, 1.0],
            [0, 1.0, 25.599998, 0.0, 2.0],
        ], dtype=torch.float32)

        topology = voxelizer.topology(points)

        self.assertEqual(topology.voxel_coords.shape[0], 1)
        self.assertLess(topology.voxel_coords[:, 2].max().item(), 320)
        self.assertEqual(
            topology.point_indices[topology.point_mask].tolist(), [0]
        )

    def test_extract_voxelized_features_excludes_zero_padding(self):
        voxels = np.array(
            [
                [[1.0, 10.0], [2.0, 20.0], [0.0, 0.0]],
                [[3.0, 30.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )

        values = extract_voxelized_features(voxels, np.array([2, 1]))

        np.testing.assert_array_equal(
            values,
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        )

    def test_streaming_feature_statistics_handles_batches_and_nonfinite_values(self):
        statistics = StreamingFeatureStatistics(
            feature_names=['x', 'rcs'],
            quantiles=[0.25, 0.5, 0.75],
            max_quantile_points=10,
            seed=7,
        )
        statistics.update(np.array([[1.0, 10.0], [2.0, 20.0]]))
        statistics.update(np.array([[3.0, np.inf]]))

        result = statistics.compute()

        self.assertEqual(result['total_points'], 3)
        self.assertEqual(result['quantile_sample_points'], 3)
        self.assertEqual(result['quantile_sampling'], 'exact')
        self.assertEqual(result['features']['x']['finite_count'], 3)
        self.assertAlmostEqual(result['features']['x']['mean'], 2.0)
        self.assertAlmostEqual(
            result['features']['x']['std'],
            np.sqrt(2.0 / 3.0),
        )
        self.assertEqual(result['features']['x']['quantiles']['0.5'], 2.0)
        self.assertEqual(result['features']['rcs']['finite_count'], 2)
        self.assertEqual(result['features']['rcs']['nonfinite_count'], 1)
        self.assertEqual(result['features']['rcs']['mean'], 15.0)
        self.assertEqual(result['features']['rcs']['unique_count'], 2)
        self.assertEqual(
            result['features']['rcs']['unique_values'],
            [10.0, 20.0],
        )

    def test_streaming_feature_statistics_caps_quantile_sample(self):
        statistics = StreamingFeatureStatistics(
            feature_names=['x'],
            quantiles=[0.5],
            max_quantile_points=3,
            seed=11,
        )
        statistics.update(np.arange(10, dtype=np.float64).reshape(-1, 1))

        result = statistics.compute()

        self.assertEqual(result['total_points'], 10)
        self.assertEqual(result['quantile_sample_points'], 3)
        self.assertEqual(
            result['quantile_sampling'],
            'uniform_without_replacement',
        )
        self.assertAlmostEqual(result['features']['x']['mean'], 4.5)

    def test_vod_summary_and_attack_drop(self):
        raw = {
            area: {
                f'{class_name}_{suffix}': float(index + metric_index + 1)
                for index, class_name in enumerate(
                    ('Car', 'Pedestrian', 'Cyclist')
                )
                for metric_index, suffix in enumerate(
                    ('3d_all', 'bev_all', 'aos_all')
                )
            }
            for area in ('entire_area', 'roi')
        }
        clean = summarize_vod_results(raw)
        adversarial = {
            area: {
                metric: {
                    key: value - 0.5
                    for key, value in metric_values.items()
                }
                for metric, metric_values in area_values.items()
            }
            for area, area_values in clean.items()
        }
        drop = _metric_difference(clean, adversarial)
        relative = _relative_metric_difference(clean, drop)

        self.assertEqual(clean['entire_area']['3d']['mAP'], 2.0)
        self.assertEqual(drop['roi']['bev']['mAP'], 0.5)
        self.assertEqual(relative['roi']['bev']['mAP'], 0.5 / 3.0)

    def test_prepare_prediction_directory_removes_only_stale_txt(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            (output_dir / 'old.txt').write_text('stale')
            (output_dir / 'keep.json').write_text('{}')

            prepared = prepare_prediction_directory(output_dir)

            self.assertEqual(prepared, output_dir)
            self.assertFalse((output_dir / 'old.txt').exists())
            self.assertTrue((output_dir / 'keep.json').exists())

    def test_projection_does_not_move_points_outside_model_z_range(self):
        original = torch.tensor(
            [[0.0, 1.0, 2.0, 6.0, 4.0, -2.0, -3.0, 0.01]]
        )
        budget = torch.tensor(
            [[0.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0]]
        )
        candidate = original + budget

        projected = project_points(
            candidate,
            original,
            budget,
            point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
            voxel_size=[0.16, 0.16, 5],
            voxel_mode='fixed',
        )

        self.assertTrue(torch.equal(projected[:, 1:4], original[:, 1:4]))

    def test_rcs_only_projection_does_not_clamp_xyz_at_pillar_edge(self):
        original = torch.tensor(
            [[0.0, 0.99999, 0.5, 0.5, 2.0]], dtype=torch.float32
        )
        candidate = original.clone()
        candidate[:, 4] += 0.2
        budget = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.2]])

        projected = project_points(
            candidate,
            original,
            budget,
            point_cloud_range=[0, 0, 0, 2, 2, 2],
            voxel_size=[1, 1, 1],
            voxel_mode='fixed',
        )

        self.assertTrue(torch.equal(projected[:, 1:4], original[:, 1:4]))
        self.assertLessEqual(
            (projected[:, 4] - original[:, 4]).abs().item(), 0.2
        )

    def test_voxel_doppler_attack_selects_both_velocity_channels(self):
        voxels = torch.tensor(
            [[[1.0, 2.0, 0.0, 4.0, -2.0, -3.0, 0.01]]]
        )
        model = SumVoxelModel()
        model.eval()

        adversarial = voxel_attack(
            model=model,
            batch_dict={'voxels': voxels},
            epsilon=0.1,
            attack_type='fgsm',
            attack_feature='doppler',
            point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        )

        expected = voxels.clone()
        expected[..., 4:6] += 0.1
        self.assertTrue(torch.allclose(adversarial, expected))
        self.assertFalse(model.training)
        mask = build_feature_mask(voxels, 'doppler')
        self.assertEqual(mask.flatten().tolist(), [0, 0, 0, 0, 1, 1, 0])

    def test_point_fgsm_returns_raw_adversarial_points(self):
        points = torch.tensor(
            [
                [0.0, 0.2, 0.3, 0.4, 2.0, 0.5, 0.6, 0.01],
                [0.0, 1.2, 1.3, 1.4, 3.0, 0.7, 0.8, 0.02],
            ]
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, 0, 0, 4, 4, 4],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=4,
            max_voxels=10,
            batch_size=1,
        )

        output = point_cloud_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            feature_names=[
                'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            ],
            attack_type='fgsm',
            attack_feature='xyz',
            epsilon=0.1,
        )

        self.assertIsInstance(output, AttackOutput)
        self.assertTrue(torch.equal(output.adv_points[:, 0], points[:, 0]))
        self.assertTrue(
            torch.allclose(output.adv_points[:, 1:4], points[:, 1:4] + 0.1)
        )
        self.assertTrue(torch.equal(output.adv_points[:, 4:], points[:, 4:]))
        self.assertFalse(output.model_inputs['voxels'].requires_grad)
        self.assertTrue(
            np.isclose(output.stats['max_abs_perturbation'], 0.1)
        )

        masked_output = point_cloud_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            feature_names=[
                'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            ],
            attack_type='fgsm',
            attack_feature='rcs',
            epsilon=0.2,
            point_mask=torch.tensor([True, False]),
        )
        self.assertAlmostEqual(
            masked_output.adv_points[0, 4].item(), 2.2, places=6
        )
        self.assertEqual(masked_output.adv_points[1, 4].item(), 3.0)

    def test_point_fgsm_accepts_a_custom_attack_objective(self):
        points = torch.tensor(
            [[0.0, 0.2, 0.3, 0.4, 2.0, 0.5, 0.6, 0.01]]
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, 0, 0, 4, 4, 4],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=4,
            max_voxels=10,
            batch_size=1,
        )

        output = point_cloud_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            feature_names=[
                'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            ],
            attack_type='fgsm',
            attack_feature='rcs',
            epsilon=0.2,
            loss_fn=lambda model: -model.last_voxels.sum(),
        )

        self.assertAlmostEqual(output.adv_points[0, 4].item(), 1.8, places=6)

    def test_original_iou_s_loss_uses_every_gt_prediction_pair(self):
        predictions = {
            'pred_boxes': torch.tensor([
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [4.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ], requires_grad=True),
            'pred_scores': torch.tensor([0.8, 0.4], requires_grad=True),
            'pred_labels': torch.tensor([1, 2]),
        }
        gt_boxes = torch.tensor([
            [0.5, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0],
            [4.5, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        loss, pair_count = original_iou_s_detection_loss(
            predictions, gt_boxes
        )
        loss.backward()

        self.assertEqual(pair_count, 4)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(predictions['pred_boxes'].grad)
        self.assertIsNotNone(predictions['pred_scores'].grad)

    def test_iou_s_geometry_reference_is_car_current_and_in_range(self):
        reference = MeasurementQ95Reference.load(
            Path(__file__).resolve().parents[1]
            / 'references/vod_stage2_gate1_point_range_q95.json'
        )
        points = torch.tensor([
            [0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            [0.0, 15.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 60.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        boxes = torch.tensor([
            [5.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 1.0],
            [15.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 2.0],
            [60.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 1.0],
        ])

        mask, q95 = build_iou_s_geometry_reference_mask(
            points=points,
            gt_boxes=boxes,
            feature_names=['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'],
            q95_reference=reference,
            reference_class_id=1,
        )

        self.assertEqual(mask.tolist(), [True, False, False, False, False])
        self.assertTrue((q95[mask] > 0).all())

    def test_iou_s_geometry_loss_uses_reference_count_denominator(self):
        clean = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        adversarial = clean.clone()
        adversarial[0, 0] = 12.0
        adversarial[1, 0] = 100.0
        adversarial.requires_grad_(True)
        mask = torch.tensor([True, False])
        q95 = torch.ones_like(clean)

        loss = radar_geometry_q95_hinge_loss(
            clean, adversarial, mask, q95
        )
        loss.backward()

        self.assertAlmostEqual(loss.item(), 1.0, places=5)
        self.assertNotEqual(adversarial.grad[0, 0].item(), 0.0)
        self.assertEqual(adversarial.grad[1].abs().sum().item(), 0.0)

    def test_symmetric_chamfer_matches_two_directed_terms(self):
        first = torch.tensor([[0.0, 0.0, 0.0]])
        second = torch.tensor([[1.0, 2.0, 2.0]])

        distance = symmetric_chamfer_squared(first, second, chunk_size=1)

        self.assertAlmostEqual(distance.item(), 18.0)

    def test_prediction_set_retention_detects_replacement(self):
        previous = {
            'pred_boxes': torch.tensor([
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [4.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]),
            'pred_labels': torch.tensor([1, 2]),
        }
        current = {
            'pred_boxes': torch.tensor([
                [0.1, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [8.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]),
            'pred_labels': torch.tensor([1, 2]),
        }

        retention = prediction_set_retention(previous, current)

        self.assertAlmostEqual(retention, 0.5)

    def test_original_iou_s_attack_is_full_scene_xyz_only(self):
        points = torch.tensor([
            [0.0, 0.5, 0.5, 0.5, 4.0, 0.2, 0.1, 0.0],
            [0.0, 1.0, 0.5, 0.5, 5.0, 0.3, 0.2, -1.0],
        ])
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, 0, 0, 4, 4, 4],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=4,
            max_voxels=10,
            batch_size=1,
        )
        model = DynamicPredictionModel()
        model.train()
        gt_boxes = torch.tensor([[[
            0.75, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0
        ]]])

        output = original_iou_s_point_attack(
            model=model,
            batch_dict={
                'points': points,
                'gt_boxes': gt_boxes,
                'batch_size': 1,
            },
            voxelizer=voxelizer,
            steps=2,
            learning_rate=0.01,
            initial_noise=0.001,
            chamfer_chunk_size=1,
        )

        self.assertIsInstance(output, AttackOutput)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(output.adv_points[:, 0], points[:, 0]))
        self.assertTrue(torch.equal(output.adv_points[:, 4:], points[:, 4:]))
        self.assertGreater(output.stats['point_modified_points'], 0)
        self.assertEqual(
            output.stats['iou_s_original_hard_epsilon_projection'], 0.0
        )
        self.assertEqual(output.stats['iou_s_original_steps'], 2.0)
        self.assertGreaterEqual(output.stats['iou_s_original_best_step'], 1.0)
        self.assertLessEqual(output.stats['iou_s_original_best_step'], 2.0)
        self.assertIn(
            'iou_s_original_mean_prediction_retention', output.stats
        )

        last_output = original_iou_s_point_attack(
            model=model,
            batch_dict={
                'points': points,
                'gt_boxes': gt_boxes,
                'batch_size': 1,
            },
            voxelizer=voxelizer,
            steps=2,
            learning_rate=0.01,
            initial_noise=0.001,
            distance_weight=0.0,
            chamfer_chunk_size=1,
            return_policy='last',
            geometry_weight=0.1,
            geometry_reference=MeasurementQ95Reference.load(
                Path(__file__).resolve().parents[1]
                / 'references/vod_stage2_gate1_point_range_q95.json'
            ),
            feature_names=[
                'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            ],
            geometry_reference_class_id=1,
        )
        self.assertEqual(last_output.stats['iou_s_original_return_last'], 1.0)
        self.assertTrue(torch.equal(last_output.adv_points[:, 4:], points[:, 4:]))
        self.assertEqual(
            last_output.stats['iou_s_original_geometry_reference_points'],
            1.0,
        )
        self.assertEqual(len(last_output.step_metrics), 2)
        self.assertIn('weighted_geometry_loss', last_output.step_metrics[0])
        self.assertEqual(
            last_output.stats[
                'iou_s_original_distance_regularizer_enabled'
            ],
            0.0,
        )
        self.assertTrue(all(
            row['distance_loss'] == 0.0
            for row in last_output.step_metrics
        ))

    def test_measurement_naturalness_uses_gate1_q95_and_wrapped_angles(self):
        reference = MeasurementQ95Reference.load(
            Path(__file__).resolve().parents[1]
            / 'references/vod_stage2_gate1_q95.json'
        )
        angle_clean = np.radians(179.0)
        angle_adversarial = np.radians(-179.0)
        elevation = 0.02
        clean = torch.tensor([
            [0.0, 5.0 * np.cos(angle_clean), 5.0 * np.sin(angle_clean), 0.0],
            [0.0, 20.0, 0.0, 0.0],
            [0.0, 60.0, 0.0, 0.0],
        ], dtype=torch.float64)
        adversarial = torch.tensor([
            [
                0.0,
                5.0 * np.cos(angle_adversarial),
                5.0 * np.sin(angle_adversarial),
                0.0,
            ],
            [
                0.0,
                20.0 * np.cos(elevation),
                0.0,
                20.0 * np.sin(elevation),
            ],
            [0.0, 60.1, 0.0, 0.0],
        ], dtype=torch.float64)

        result = point_measurement_naturalness(
            clean, adversarial, reference
        )

        self.assertEqual(reference.metadata['gate_m'], 1.0)
        self.assertAlmostEqual(
            result['x'][0, 1], 5.0 * np.radians(2.0), places=8
        )
        self.assertAlmostEqual(result['x'][1, 2], 20.0 * elevation, places=8)
        self.assertEqual(result['bin_index'].tolist(), [0, 2, -1])
        self.assertEqual(result['covered'].tolist(), [True, True, False])
        self.assertAlmostEqual(
            result['z'][1, 2],
            0.4 / 0.385273922220705,
            places=8,
        )

    def test_measurement_naturalness_rejects_non_one_metre_gate(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / 'references/vod_stage2_gate1_q95.json'
        )
        payload = json.loads(source_path.read_text(encoding='utf-8'))
        payload['gate_m'] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong_gate.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '1 m gate'):
                MeasurementQ95Reference.load(path)

    def test_measurement_naturalness_accumulator_keeps_pointwise_tuple(self):
        reference = MeasurementQ95Reference.load(
            Path(__file__).resolve().parents[1]
            / 'references/vod_stage2_gate1_q95.json'
        )
        clean = torch.tensor([
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 15.0, 0.0, 0.0],
        ])
        adversarial = clean.clone()
        adversarial[0, 1] += 0.3
        accumulator = MeasurementNaturalnessAccumulator(reference)
        accumulator.update(clean, adversarial, ['000001'])

        summary = accumulator.compute()

        self.assertEqual(summary['total_points'], 2)
        self.assertEqual(summary['covered_points'], 2)
        self.assertEqual(summary['modified_points'], 1)
        self.assertGreater(
            summary['modified_covered']['A_exceedance_fraction']['gt_1'], 0
        )
        self.assertEqual(summary['by_range_bin']['0-10']['all_covered']['count'], 1)
        self.assertEqual(summary['by_range_bin']['10-20']['all_covered']['count'], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'points.csv.gz'
            accumulator.write_csv(path)
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                header = stream.readline()
                rows = stream.readlines()
            self.assertIn('z_range,z_azimuth,z_elevation,A', header)
            self.assertEqual(len(rows), 2)

    def test_radar_measurement_cartesian_round_trip(self):
        xyz = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [5.0, 2.0, 1.0],
                [7.0, -3.0, -0.5],
                [0.0, 4.0, 2.0],
            ],
            dtype=torch.float64,
        )

        measurement = cartesian_to_radar_measurement(xyz)
        reconstructed = radar_measurement_to_cartesian(measurement)

        self.assertTrue(torch.allclose(reconstructed, xyz, atol=1e-12))
        self.assertAlmostEqual(measurement[0, 1].item(), 0.0)
        self.assertGreater(measurement[1, 1].item(), 0.0)
        self.assertLess(measurement[2, 1].item(), 0.0)

    def test_measurement_mask_keeps_only_current_clean_active_targets(self):
        points = torch.tensor(
            [
                [0.0, 1.1, 1.1, 1.1, 2.0, 0.5, 0.1, 0.0],
                [0.0, 1.2, 1.2, 1.2, 3.0, 0.6, 0.2, 0.0],
                [0.0, 2.1, 2.1, 2.1, 4.0, 0.7, 0.3, -1.0],
                [0.0, 3.1, 3.1, 3.1, 5.0, 0.8, 0.4, 0.0],
            ]
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, 0, 0, 5, 5, 5],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=1,
            max_voxels=10,
            batch_size=1,
        )
        target_mask = torch.tensor([True, True, True, False])

        attack_mask, stats = build_measurement_attack_mask(
            points,
            ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'],
            target_mask,
            voxelizer,
        )

        self.assertEqual(attack_mask.tolist(), [True, False, False, False])
        self.assertEqual(stats['measurement_target_points'], 3.0)
        self.assertEqual(stats['measurement_current_target_points'], 2.0)
        self.assertEqual(stats['measurement_historical_target_points'], 1.0)
        self.assertEqual(
            stats['measurement_clean_active_current_target_points'], 1.0
        )

    def test_xyz_current_and_measurement_use_identical_point_ids(self):
        points = torch.tensor(
            [
                [0.0, 2.0, 0.1, 0.0, 2.0, 0.5, 0.1, 0.0],
                [0.0, 2.1, 0.1, 0.0, 2.0, 0.5, 0.1, 0.0],
                [0.0, 2.2, 0.1, 0.0, 2.0, 0.5, 0.1, -1.0],
                [0.0, 4.0, 0.0, 0.0, 2.0, 0.5, 0.1, 0.0],
            ]
        )
        feature_names = [
            'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        ]
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -5, -5, 6, 5, 5],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=1,
            max_voxels=10,
            batch_size=1,
        )
        batch_dict = {
            'points': points,
            'batch_size': 1,
            'gt_boxes': torch.tensor([[[2.1, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]]]),
        }
        clean_records = {0: [{
            'frame_id': '00001', 'object_id': 0, 'gt_row': 0,
            'class_name': 'Car', 'clean_match_score': 0.9,
            'clean_max_iou': 0.8,
        }]}
        context = build_target_screening_context(
            points, batch_dict, clean_records, feature_names, voxelizer
        )
        measurement_ids, _ = build_measurement_attack_mask(
            points, feature_names, context.target_mask, voxelizer
        )
        xyz_current_ids = (
            context.target_mask & context.current_mask & context.active_mask
        )

        self.assertTrue(torch.equal(xyz_current_ids, measurement_ids))
        self.assertEqual(
            torch.nonzero(measurement_ids).flatten().tolist(), [0]
        )

    def test_target_screening_and_reassignment_diagnostics(self):
        points = torch.tensor(
            [
                [0.0, 0.9, 0.0, 0.0, 2.0, 0.5, 0.1, 0.0],
                [0.0, 1.1, 0.0, 0.0, 2.0, 0.5, 0.1, -1.0],
                [0.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.1, 0.0],
            ]
        )
        feature_names = [
            'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        ]
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -5, -5, 6, 5, 5],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=4,
            max_voxels=10,
            batch_size=1,
        )
        batch_dict = {
            'points': points,
            'batch_size': 1,
            'gt_boxes': torch.tensor([[[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]]]),
        }
        clean_records = {0: [{
            'frame_id': '00001', 'object_id': 0, 'gt_row': 0,
            'class_name': 'Car', 'clean_match_score': 0.9,
            'clean_max_iou': 0.8,
        }]}
        context = build_target_screening_context(
            points, batch_dict, clean_records, feature_names, voxelizer
        )
        attack_mask, _ = build_measurement_attack_mask(
            points, feature_names, context.target_mask, voxelizer
        )
        adversarial = points.clone()
        adversarial[0, 1] = 1.01
        diagnostics = target_attack_diagnostics(
            context, adversarial, attack_mask
        )
        record = diagnostics[(0, 0)]

        self.assertEqual(record['num_object_points_total'], 2)
        self.assertEqual(record['num_time0_points'], 1)
        self.assertEqual(record['num_history_points'], 1)
        self.assertEqual(record['num_active_time0_points'], 1)
        self.assertEqual(record['num_reassigned_points'], 1)
        self.assertEqual(record['pillar_reassignment_rate'], 1.0)
        endpoint = [{
            'batch_index': 0, 'gt_row': 0,
            'adversarial_match_score': 0.5, 'match_score_drop': 0.4,
            'adversarial_max_iou': 0.3, 'max_iou_drop': 0.5,
            'clean_object_evidence': 1.2,
            'adversarial_object_evidence': 0.8,
            'object_evidence_drop': 0.4, 'object_failure': True,
            'adversarial_center_error': 0.2,
        }]
        merge_target_diagnostics(endpoint, diagnostics)
        summary = summarize_current_sweep(endpoint)
        correlations = target_correlations(endpoint)
        self.assertTrue(endpoint[0]['object_attack_success'])
        self.assertEqual(summary['clean_detected_targets'], 1)
        self.assertEqual(summary['active_time0_le_1_fraction'], 1.0)
        self.assertIn('pillar_reassignment_rate_vs_iou_drop', correlations)

    def test_measurement_projection_backtracks_without_cartesian_clipping(self):
        xyz = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)
        original = cartesian_to_radar_measurement(xyz)
        budget = torch.tensor([[0.0, torch.pi, 0.0]], dtype=torch.float64)
        candidate = original + budget

        projected, stats = project_radar_measurements(
            candidate,
            original,
            budget,
            torch.tensor([True]),
            point_cloud_range=[0, -1, -1, 1, 1, 1],
        )
        reconstructed = radar_measurement_to_cartesian(projected)

        self.assertGreater(stats['measurement_out_of_range_backtracks'], 0)
        self.assertGreaterEqual(reconstructed[0, 0].item(), 0.0)
        self.assertLessEqual(
            (projected - original).abs()[0, 1].item(), torch.pi
        )

    def test_measurement_attack_is_current_target_only_and_budgeted(self):
        points = torch.tensor(
            [
                [0.0, 2.0, 0.3, 0.2, 2.0, 0.5, 0.1, 0.0],
                [0.0, 3.0, 0.4, 0.1, 3.0, 0.6, 0.2, -1.0],
                [0.0, 4.0, -0.2, 0.3, 4.0, 0.7, 0.3, 0.0],
            ]
        )
        feature_names = [
            'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        ]
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -5, -5, 6, 5, 5],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=4,
            max_voxels=10,
            batch_size=1,
        )
        target_mask = torch.tensor([True, True, False])

        output = radar_measurement_geometry_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            feature_names=feature_names,
            attack_type='pgd',
            epsilon_range=0.1,
            epsilon_azimuth=0.05,
            epsilon_elevation=0.02,
            pgd_steps=2,
            target_mask=target_mask,
        )

        self.assertFalse(torch.equal(output.adv_points[0, 1:4], points[0, 1:4]))
        self.assertTrue(torch.equal(output.adv_points[1], points[1]))
        self.assertTrue(torch.equal(output.adv_points[2], points[2]))
        self.assertTrue(torch.equal(output.adv_points[:, 4:], points[:, 4:]))
        clean_measurement = cartesian_to_radar_measurement(points[:, 1:4])
        adv_measurement = cartesian_to_radar_measurement(
            output.adv_points[:, 1:4]
        )
        delta = (adv_measurement - clean_measurement).abs()[0]
        self.assertLessEqual(delta[0].item(), 0.1 + 1e-6)
        self.assertLessEqual(delta[1].item(), 0.05 + 1e-6)
        self.assertLessEqual(delta[2].item(), 0.02 + 1e-6)
        reconstructed = radar_measurement_to_cartesian(adv_measurement)
        self.assertTrue(torch.allclose(
            reconstructed, output.adv_points[:, 1:4], atol=1e-6
        ))
        self.assertEqual(
            output.stats['measurement_historical_modification_count'], 0.0
        )
        self.assertEqual(
            output.stats['measurement_non_target_modification_count'], 0.0
        )
        self.assertEqual(
            output.stats['measurement_non_geometry_modification_count'], 0.0
        )

    def test_zero_budget_measurement_attack_is_exactly_clean(self):
        points = torch.tensor(
            [[0.0, 2.0, 0.3, 0.2, 2.0, 0.5, 0.1, 0.0]]
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, -5, -5, 6, 5, 5],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=4,
            max_voxels=10,
            batch_size=1,
        )

        output = radar_measurement_geometry_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            feature_names=[
                'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            ],
            attack_type='pgd',
            epsilon_range=0.0,
            epsilon_azimuth=0.0,
            epsilon_elevation=0.0,
            pgd_steps=2,
            target_mask=torch.tensor([True]),
        )

        self.assertTrue(torch.equal(output.adv_points, points))
        self.assertEqual(output.stats['max_abs_perturbation'], 0.0)

    def test_oriented_box_mask_and_target_class_filter(self):
        xyz = torch.tensor(
            [
                [1.0, 1.8, 1.0],
                [1.8, 1.0, 1.0],
                [4.0, 4.0, 4.0],
            ]
        )
        boxes = torch.tensor(
            [[1.0, 1.0, 1.0, 2.0, 1.0, 2.0, torch.pi / 2]]
        )
        self.assertEqual(
            points_in_oriented_boxes(xyz, boxes).tolist(),
            [True, False, False],
        )

        points = torch.cat(
            (torch.zeros((3, 1)), xyz, torch.ones((3, 4))), dim=1
        )
        gt_boxes = torch.tensor(
            [[
                [1.0, 1.0, 1.0, 2.0, 1.0, 2.0, torch.pi / 2, 1.0],
                [4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 0.0, 2.0],
            ]]
        )
        mask = build_iadv_attack_mask(
            points,
            {'batch_size': 1, 'gt_boxes': gt_boxes},
            scope='gt_boxes',
            target_class_id=1,
        )
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_points_are_assigned_to_individual_boxes(self):
        points = torch.tensor(
            [
                [-0.8, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        )
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )

        assignments = assign_points_to_oriented_boxes(points, boxes)

        self.assertEqual(assignments.tolist(), [0, 1, -1])

    def test_overlapping_box_assignment_uses_normalized_center_distance(self):
        points = torch.tensor([[0.85, 0.0, 0.0]])
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0],
            ]
        )

        assignments = assign_points_to_oriented_boxes(points, boxes)

        self.assertEqual(assignments.item(), 1)

    def test_object_ids_are_batch_local_and_class_filtered(self):
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 3.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        gt_boxes = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0],
                ],
                [
                    [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )

        object_ids = build_iadv_object_ids(
            points,
            {'batch_size': 2, 'gt_boxes': gt_boxes},
            scope='gt_boxes',
            target_class_id=1,
        )

        self.assertEqual(object_ids.tolist(), [0, -1, 0])

        multi_class_ids = build_iadv_object_ids(
            points,
            {'batch_size': 2, 'gt_boxes': gt_boxes},
            scope='gt_boxes',
            target_class_ids=[1, 2],
        )
        self.assertEqual(multi_class_ids.tolist(), [0, 1, 0])

    def test_object_endpoints_are_mutually_exclusive_across_classes(self):
        gt_boxes = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0],
            [6.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0],
            [9.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        ])
        clean = {
            'pred_boxes': gt_boxes[:, :7].clone(),
            'pred_scores': torch.full((4,), 0.9),
            'pred_labels': gt_boxes[:, -1].long(),
        }
        adversarial_boxes = torch.stack((
            gt_boxes[0, :7],
            gt_boxes[1, :7],
            gt_boxes[2, :7].clone(),
        ))
        adversarial_boxes[2, 0] += 0.8
        adversarial = {
            'pred_boxes': adversarial_boxes,
            'pred_scores': torch.full((3,), 0.8),
            'pred_labels': torch.tensor([1, 1, 3]),
        }

        with patch(
            'tools.radar_attack.evaluation.object_endpoints.'
            'iou3d_nms_utils.boxes_iou3d_gpu',
            side_effect=self._axis_aligned_iou,
        ):
            target_count, records = compare_target_object_endpoints(
                clean,
                adversarial,
                gt_boxes,
                target_classes={1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'},
                iou_thresholds={1: 0.5, 2: 0.25, 3: 0.25},
                frame_id='000001',
                object_score_threshold=0.1,
                class_names={1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'},
            )

        self.assertEqual(target_count, 4)
        self.assertEqual(
            [record['outcome'] for record in records],
            ['still_correct', 'misclassification', 'localization_failure', 'pure_hiding'],
        )

    def test_object_neighbours_do_not_cross_targets_and_sparse_pca_falls_back(self):
        points = torch.tensor(
            [
                [0.0, 1.00, 0.0, 0.0, 1.0],
                [0.0, 1.01, 0.0, 0.0, 1.0],
                [0.0, 1.02, 0.0, 0.0, 1.0],
                [0.0, 1.03, 0.0, 0.0, 1.0],
            ]
        )
        attack_mask = torch.ones(4, dtype=torch.bool)
        object_ids = torch.tensor([0, 0, 1, 1])

        _, object_stats = compute_reflectivity_features(
            points,
            attack_mask,
            k_neighbors=3,
            min_neighbors=3,
            neighbor_scope='object',
            object_ids=object_ids,
        )
        _, union_stats = compute_reflectivity_features(
            points,
            attack_mask,
            k_neighbors=3,
            min_neighbors=3,
            neighbor_scope='attack_union',
            object_ids=object_ids,
        )

        self.assertEqual(object_stats['reflectivity_fallback_points'], 4.0)
        self.assertEqual(
            object_stats['reflectivity_cross_target_neighbors'], 0.0
        )
        self.assertGreater(
            union_stats['reflectivity_cross_target_neighbors'], 0.0
        )

    def test_object_id_is_part_of_gradient_fusion_group_key(self):
        points = torch.tensor(
            [
                [0.0, 1.01, 1.01, 1.01, 1.0],
                [0.0, 1.02, 1.02, 1.02, 1.0],
                [0.0, 1.03, 1.03, 1.03, 1.0],
                [0.0, 1.04, 1.04, 1.04, 1.0],
            ]
        )
        attack_mask = torch.ones(4, dtype=torch.bool)
        object_ids = torch.tensor([0, 0, 1, 1])
        group_ids, num_groups = build_iadv_groups(
            points,
            attack_mask,
            voxel_size=0.1,
            object_ids=object_ids,
        )

        directions = extremum_fusion(
            torch.tensor([2.0, 1.0, -3.0, -1.0]), group_ids, num_groups
        )

        self.assertEqual(num_groups, 2)
        self.assertEqual(directions.tolist(), [1.0, 1.0, -1.0, -1.0])

    def test_reflectivity_feature_uses_distance_fallback_when_pca_is_sparse(self):
        points = torch.tensor(
            [
                [0.0, 3.0, 4.0, 0.0, 1.0],
                [0.0, 6.0, 8.0, 0.0, 1.0],
            ]
        )
        features, stats = compute_reflectivity_features(
            points,
            torch.tensor([True, True]),
            k_neighbors=3,
            min_neighbors=3,
            d_max=20.0,
        )

        expected = torch.sin(
            torch.tensor([5.0, 10.0]) / 20.0 * torch.pi / 2
        )
        self.assertTrue(torch.allclose(features, expected))
        self.assertEqual(stats['reflectivity_fallback_points'], 2.0)

    def test_extremum_fusion_uses_largest_signed_extreme(self):
        gradient = torch.tensor([2.0, -3.0, 1.0, -0.5, 0.0])
        group_ids = torch.tensor([0, 0, 1, 1, 2])

        directions = extremum_fusion(gradient, group_ids, num_groups=3)

        self.assertEqual(
            directions.tolist(), [-1.0, -1.0, 1.0, 1.0, 0.0]
        )

    def test_iadv_returns_budgeted_rcs_only_adversarial_points(self):
        points = torch.tensor(
            [
                [0.0, 1.01, 1.01, 1.01, 2.0, 0.5, 0.6, 0.01],
                [0.0, 1.04, 1.01, 1.01, 3.0, 0.7, 0.8, 0.02],
                [0.0, 1.01, 1.04, 1.01, 4.0, 0.9, 1.0, 0.03],
            ]
        )
        voxelizer = PointCloudVoxelizer(
            point_cloud_range=[0, 0, 0, 4, 4, 4],
            voxel_size=[1, 1, 1],
            max_points_per_voxel=8,
            max_voxels=10,
            batch_size=1,
        )

        output = iadv_rcs_attack(
            model=SumVoxelModel(),
            batch_dict={'points': points, 'batch_size': 1},
            voxelizer=voxelizer,
            feature_names=[
                'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            ],
            epsilon_rcs=0.5,
            steps=2,
            attack_voxel_size=0.1,
            gradient_enhancement=1.0,
            k_neighbors=3,
            min_neighbors=3,
            scope='scene',
            neighbor_scope='scene',
        )

        self.assertTrue(torch.equal(output.adv_points[:, :4], points[:, :4]))
        self.assertTrue(
            torch.allclose(output.adv_points[:, 4], points[:, 4] + 0.2)
        )
        self.assertTrue(torch.equal(output.adv_points[:, 5:], points[:, 5:]))
        self.assertLessEqual(output.stats['max_abs_perturbation'], 0.5)
        self.assertEqual(output.stats['iadv_attacked_points'], 3.0)
        self.assertEqual(output.stats['iadv_groups'], 1.0)
        self.assertEqual(output.stats['iadv_cross_target_neighbors'], 0.0)
        self.assertEqual(output.stats['iadv_nonfinite_gradient_steps'], 0.0)

    def test_metrics_support_batches_and_weight_perturbations(self):
        metrics = DetectionAttackMetrics()
        clean = [
            {'pred_scores': torch.tensor([0.8])},
            {'pred_scores': torch.tensor([0.3])},
        ]
        adversarial = [
            {'pred_scores': torch.empty(0)},
            {'pred_scores': torch.tensor([0.2])},
        ]
        metrics.update_predictions(
            clean,
            adversarial,
            {'rcnn_0.5': 3, 'gt': 4},
            {'rcnn_0.5': 1, 'gt': 4},
        )
        metrics.update_perturbation(
            {
                'max_abs_perturbation': 0.2,
                'sum_abs_perturbation': 0.6,
                'perturbation_values': 4,
                'iadv_attacked_points': 8,
                'iadv_valid_targets': 2,
                'iadv_groups': 5,
                'iadv_singleton_groups': 3,
                'reflectivity_fallback_points': 2,
                'iadv_cross_target_neighbors': 0,
                'object_evidence_targets': 2,
                'object_evidence_candidate_anchors': 20,
                'measurement_clean_active_current_target_points': 4,
                'measurement_xyz_l2_sum': 0.8,
                'measurement_max_xyz_l2': 0.3,
                'measurement_historical_modification_count': 0,
            }
        )
        metrics.update_object_endpoints(
            3,
            [
                {
                    'frame_id': '000001',
                    'class_name': 'Car',
                    'outcome': 'pure_hiding',
                    'max_iou_drop': 0.2,
                    'match_score_drop': 0.1,
                    'object_evidence_drop': 0.3,
                    'prediction_center_shift': 0.05,
                    'center_error_increase': 0.02,
                    'prediction_size_l1_shift': 0.01,
                    'prediction_yaw_shift': 0.005,
                    'clean_iou_margin': 0.3,
                    'adversarial_iou_margin': 0.1,
                }
            ]
        )

        result = metrics.compute()
        self.assertEqual(result['total_samples'], 2)
        self.assertEqual(result['original_recall'], 0.75)
        self.assertEqual(result['attacked_recall'], 0.25)
        self.assertNotIn('attack_success_rate_sample', result)
        self.assertEqual(result['mean_abs_perturbation'], 0.15)
        self.assertEqual(result['object_outcomes']['target_objects'], 3)
        self.assertEqual(result['object_outcomes']['eligible_clean_objects'], 1)
        self.assertAlmostEqual(result['object_failure_asr'], 1.0)
        self.assertAlmostEqual(result['pure_hiding_asr'], 1.0)
        diagnostics = result['attack_diagnostics']
        self.assertEqual(diagnostics['iadv_valid_targets'], 2)
        self.assertEqual(diagnostics['iadv_mean_points_per_target'], 4)
        self.assertEqual(diagnostics['iadv_mean_points_per_group'], 1.6)
        self.assertAlmostEqual(diagnostics['measurement_mean_xyz_l2'], 0.2)
        self.assertAlmostEqual(diagnostics['measurement_max_xyz_l2'], 0.3)
        self.assertEqual(
            diagnostics['measurement_historical_modification_count'], 0
        )
        self.assertEqual(diagnostics['iadv_singleton_group_ratio'], 0.6)
        self.assertEqual(diagnostics['iadv_pca_fallback_rate'], 0.25)
        self.assertEqual(diagnostics['object_evidence_targets'], 2)
        self.assertEqual(
            diagnostics['object_evidence_mean_candidates_per_target'], 10
        )
        endpoint = result['object_endpoint_metrics']
        self.assertEqual(endpoint['evaluated_clean_objects'], 1)
        self.assertEqual(endpoint['max_iou_drop']['mean'], 0.2)
        self.assertEqual(
            endpoint['object_evidence_drop']['positive_fraction'], 1.0
        )

    def test_writer_saves_feature_only_arrays_and_manifest(self):
        clean = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 4.0, 5.0, 6.0],
            ]
        )
        adversarial = clean.clone()
        adversarial[:, 1] += 0.1

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            writer = AdversarialPointCloudWriter(
                output_dir,
                feature_names=['x', 'y', 'z'],
                run_metadata={'attack': 'fgsm'},
            )
            writer.save_batch(
                clean,
                adversarial,
                {'batch_size': 2, 'frame_id': ['frame/a', 'frame-b']},
            )

            first = np.load(output_dir / 'frame_a.npy')
            self.assertEqual(first.shape, (1, 3))
            self.assertTrue(
                np.allclose(first, adversarial[:1, 1:].numpy())
            )
            records = [
                json.loads(line)
                for line in (
                    output_dir / 'manifest.jsonl'
                ).read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['feature_names'], ['x', 'y', 'z'])
            self.assertEqual(records[0]['shape'], [1, 3])
            self.assertEqual(records[0]['attack'], 'fgsm')


if __name__ == '__main__':
    unittest.main()
