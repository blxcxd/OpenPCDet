import json
import tempfile
import unittest
from pathlib import Path

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
    build_feature_mask,
    point_cloud_attack,
    voxel_attack,
)
from tools.radar_attack.attacks.gradient import project_points
from tools.radar_attack.evaluation import (
    AdversarialPointCloudWriter,
    DetectionAttackMetrics,
    prepare_prediction_directory,
)
from tools.radar_attack.evaluation.vod import (
    _metric_difference,
    _relative_metric_difference,
    summarize_vod_results,
)
from tools.radar_attack.runner import select_sample_indices


class SumVoxelModel(nn.Module):
    def forward(self, batch_dict):
        return {'loss': batch_dict['voxels'].sum()}, {}, {}


class RadarAttackComponentTest(unittest.TestCase):
    def test_validation_subset_selection_is_reproducible(self):
        uniform = select_sample_indices(10, 3, 'uniform', seed=7)
        random_first = select_sample_indices(20, 5, 'random', seed=7)
        random_second = select_sample_indices(20, 5, 'random', seed=7)

        np.testing.assert_array_equal(uniform, np.array([1, 5, 8]))
        np.testing.assert_array_equal(random_first, random_second)
        self.assertEqual(len(np.unique(random_first)), 5)

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

    def test_metrics_support_batches_and_weight_perturbations(self):
        metrics = DetectionAttackMetrics(score_threshold=0.5)
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
            }
        )

        result = metrics.compute()
        self.assertEqual(result['total_samples'], 2)
        self.assertEqual(result['original_recall'], 0.75)
        self.assertEqual(result['attacked_recall'], 0.25)
        self.assertEqual(result['attack_success_rate_sample'], 1.0)
        self.assertEqual(result['mean_abs_perturbation'], 0.15)

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
