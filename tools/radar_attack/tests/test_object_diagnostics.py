import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.radar_attack.analysis import (
    candidate_anchor_indices,
    domain_sensitivities,
    feature_scales_from_statistics,
    first_order_evidence_drop,
    hybrid_gradient_relationships,
    normalized_probe_budget,
    object_evidence,
    reshape_anchor_cls_logits,
    sensitivity_allocations,
    summarize_records,
    write_diagnostic_report,
)
from tools.radar_attack.attacks.objective import (
    ObjectEvidenceObjective,
    ObjectEvidenceTarget,
)


class ObjectDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.feature_names = [
            'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        ]
        self.scales = {
            'x': 10.0,
            'y': 5.0,
            'z': 2.0,
            'rcs': 8.0,
            'v_r': 4.0,
            'v_r_comp': 3.0,
            'time': 1.0,
        }

    def test_reshape_anchor_logits_preserves_anchor_class_order(self):
        logits = torch.arange(24.0).reshape(1, 2, 2, 6)

        reshaped = reshape_anchor_cls_logits(logits, num_classes=3)

        self.assertEqual(tuple(reshaped.shape), (1, 8, 3))
        torch.testing.assert_close(reshaped[0, 1], torch.tensor([3.0, 4.0, 5.0]))

    def test_candidate_selection_uses_oriented_box_and_fallback(self):
        anchors = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1, 1, 1, 0],
                [0.0, 1.5, 0.0, 1, 1, 1, 0],
                [4.0, 0.0, 0.0, 1, 1, 1, 0],
            ],
            dtype=torch.float32,
        )
        gt_box = torch.tensor([0.0, 0.0, 0.0, 4.0, 1.0, 2.0, 0.0])

        selected = candidate_anchor_indices(
            anchors, gt_box, margin=0.0, fallback_topk=1
        )

        self.assertEqual(selected.tolist(), [0])
        far_box = gt_box.clone()
        far_box[0] = 20.0
        fallback = candidate_anchor_indices(
            anchors, far_box, margin=0.0, fallback_topk=1
        )
        self.assertEqual(fallback.tolist(), [2])

    def test_object_evidence_decreases_when_all_candidates_decrease(self):
        logits = torch.tensor([[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]])
        candidates = torch.tensor([0, 1])
        clean = object_evidence(logits, candidates, class_index=0)
        attacked = object_evidence(logits - 0.4, candidates, class_index=0)

        self.assertAlmostEqual(float((clean - attacked).item()), 0.4, places=6)

    def test_domain_sensitivity_and_coupled_doppler_are_normalized(self):
        gradient = torch.zeros((2, 8))
        gradient[0, 1:] = torch.tensor([1, 2, 3, 4, 5, -5, 9])
        mask = torch.tensor([True, False])

        result = domain_sensitivities(
            gradient, mask, self.feature_names, self.scales
        )

        self.assertEqual(result['geometry_sensitivity_sum'], 26.0)
        self.assertEqual(result['rcs_sensitivity_sum'], 32.0)
        self.assertEqual(result['doppler_sensitivity_sum'], 35.0)
        self.assertEqual(result['doppler_coupled_sensitivity_sum'], 5.0)

    def test_iqr_and_probe_budget_scales_answer_different_questions(self):
        gradient = torch.zeros((1, 8))
        gradient[0, 1] = 1.0
        gradient[0, 4] = 1.0
        mask = torch.tensor([True])

        iqr = domain_sensitivities(
            gradient, mask, self.feature_names, self.scales
        )
        budget_scales = {
            name: 0.1 for name in self.feature_names
        }
        budget = domain_sensitivities(
            gradient, mask, self.feature_names, budget_scales
        )

        self.assertEqual(iqr['geometry_sensitivity_sum'], 10.0)
        self.assertEqual(iqr['rcs_sensitivity_sum'], 8.0)
        self.assertAlmostEqual(budget['geometry_sensitivity_sum'], 0.1)
        self.assertAlmostEqual(budget['rcs_sensitivity_sum'], 0.1)

    def test_object_evidence_objective_suppresses_fixed_candidates(self):
        class DenseHead:
            forward_ret_dict = {
                'cls_preds': torch.tensor(
                    [[[[1.0, -1.0, 2.0, -2.0]]]],
                    requires_grad=True,
                )
            }

        class Model:
            dense_head = DenseHead()

        target = ObjectEvidenceTarget(
            batch_index=0,
            gt_index=0,
            class_id=1,
            candidate_indices=torch.tensor([0, 1]),
            clean_iou=0.8,
            clean_score=0.9,
        )
        objective = ObjectEvidenceObjective(
            targets=[target],
            target_boxes_by_batch={
                0: torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
            },
            num_classes=2,
            temperature=1.0,
        )

        evidence = objective.evidence_by_target(Model())
        loss = objective(Model())
        loss.backward()
        gradient = Model.dense_head.forward_ret_dict['cls_preds'].grad

        self.assertLess(float(loss.item()), 0.0)
        self.assertEqual(set(evidence), {(0, 0)})
        self.assertAlmostEqual(evidence[(0, 0)], -float(loss.item()))
        self.assertTrue((gradient[..., [0, 2]] < 0).all())
        points = torch.tensor(
            [
                [0.0, 0.5, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.assertEqual(objective.point_mask(points).tolist(), [True, False, False])

    def test_hybrid_objective_suppresses_class_and_increases_box_error(self):
        class BoxCoder:
            code_size = 7

        class DenseHead:
            box_coder = BoxCoder()
            forward_ret_dict = {
                'cls_preds': torch.tensor(
                    [[[[1.0, -1.0, 2.0, -2.0]]]],
                    requires_grad=True,
                ),
                'box_preds': torch.tensor(
                    [[[[0.2] + [0.0] * 13]]],
                    requires_grad=True,
                ),
            }

            @staticmethod
            def add_sin_difference(predictions, targets):
                return predictions, targets

        class Model:
            dense_head = DenseHead()

        target = ObjectEvidenceTarget(
            batch_index=0,
            gt_index=0,
            class_id=1,
            candidate_indices=torch.tensor([0, 1]),
            clean_iou=0.8,
            clean_score=0.9,
            localization_indices=torch.tensor([0]),
            localization_targets=torch.zeros((1, 7)),
            localization_weights=torch.ones(1),
        )
        objective = ObjectEvidenceObjective(
            targets=[target],
            target_boxes_by_batch={},
            num_classes=2,
            temperature=1.0,
            localization_weight=1.0,
        )

        loss = objective(Model())
        loss.backward()

        cls_gradient = Model.dense_head.forward_ret_dict['cls_preds'].grad
        box_gradient = Model.dense_head.forward_ret_dict['box_preds'].grad
        self.assertTrue((cls_gradient[..., [0, 2]] < 0).all())
        self.assertGreater(float(box_gradient[..., 0].item()), 0.0)
        self.assertEqual(
            objective.stats['object_hybrid_localization_anchors'], 1.0
        )

    def test_allocations_sum_to_one_and_identify_dominant_domain(self):
        sensitivities = {
            'geometry_sensitivity_sum': 1.0,
            'doppler_sensitivity_sum': 2.0,
            'rcs_sensitivity_sum': 7.0,
        }

        allocations = sensitivity_allocations(sensitivities)

        total = sum(
            allocations[f'{name}_allocation']
            for name in ('geometry', 'doppler', 'rcs')
        )
        self.assertAlmostEqual(total, 1.0)
        self.assertEqual(allocations['dominant_domain'], 'rcs')

    def test_probe_budget_uses_iqr_fraction_and_domain_caps(self):
        points = torch.zeros((2, 8))

        budget = normalized_probe_budget(
            points,
            self.feature_names,
            self.scales,
            fraction=0.1,
            domains=('geometry', 'doppler', 'rcs'),
            domain_caps={'geometry': 0.2, 'doppler': 0.3, 'rcs': 0.5},
        )

        self.assertLessEqual(float(budget[0, 1]), 0.2)
        self.assertLessEqual(float(budget[0, 5]), 0.3)
        self.assertLessEqual(float(budget[0, 4]), 0.5)
        self.assertEqual(float(budget[0, 7]), 0.0)

    def test_object_ascent_direction_has_positive_predicted_drop(self):
        evidence_gradient = torch.tensor([[0.0, 2.0, -3.0]])
        object_ascent_gradient = -evidence_gradient
        mask = torch.tensor([True])
        budget = torch.tensor([[0.0, 0.5, 0.25]])

        drop = first_order_evidence_drop(
            evidence_gradient, object_ascent_gradient, mask, budget
        )

        self.assertAlmostEqual(drop, 1.75)

    def test_hybrid_gradient_relationships_measure_scale_and_conflict(self):
        classification = torch.zeros((2, 8))
        localization = torch.zeros((2, 8))
        classification[:, 4] = torch.tensor([2.0, -1.0])
        localization[:, 4] = torch.tensor([1.0, 2.0])

        result = hybrid_gradient_relationships(
            classification,
            localization,
            torch.tensor([True, True]),
            self.feature_names,
        )

        self.assertEqual(result['hybrid_rcs_classification_l1'], 3.0)
        self.assertEqual(result['hybrid_rcs_localization_l1'], 3.0)
        self.assertEqual(result['hybrid_rcs_balance_beta_l1'], 1.0)
        self.assertAlmostEqual(
            result['hybrid_rcs_cosine_similarity'], 0.0, places=6
        )
        self.assertEqual(result['hybrid_rcs_sign_agreement'], 0.5)
        self.assertEqual(
            result['hybrid_rcs_beta1_sign_change_fraction'], 0.5
        )

    def test_feature_scales_and_report_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            stats_path = root / 'stats.json'
            stats_path.write_text(
                json.dumps(
                    {
                        'statistics': {
                            'features': {
                                name: {'iqr': value}
                                for name, value in self.scales.items()
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            loaded = feature_scales_from_statistics(
                stats_path, self.feature_names
            )
            self.assertEqual(loaded, self.scales)

            record = {
                'frame_id': '000001',
                'object_id': 0,
                'class_name': 'Car',
                'distance_xy': 10.0,
                'active_point_count': 5,
                'mean_abs_v_r_comp': 2.0,
                'training_loss_evidence_drop': 0.1,
                'object_loss_evidence_drop': 0.2,
                'training_loss_predicted_drop': 0.1,
                'object_loss_predicted_drop': 0.3,
                'geometry_sensitivity_sum': 1.0,
                'doppler_sensitivity_sum': 2.0,
                'rcs_sensitivity_sum': 3.0,
                'geometry_allocation': 1 / 6,
                'doppler_allocation': 2 / 6,
                'rcs_allocation': 3 / 6,
                'dominant_domain': 'rcs',
                'iqr_geometry_sensitivity_sum': 1.0,
                'iqr_doppler_sensitivity_sum': 2.0,
                'iqr_rcs_sensitivity_sum': 3.0,
                'iqr_geometry_allocation': 1 / 6,
                'iqr_doppler_allocation': 2 / 6,
                'iqr_rcs_allocation': 3 / 6,
                'iqr_dominant_domain': 'rcs',
                'budget_geometry_sensitivity_sum': 4.0,
                'budget_doppler_sensitivity_sum': 2.0,
                'budget_rcs_sensitivity_sum': 1.0,
                'budget_geometry_allocation': 4 / 7,
                'budget_doppler_allocation': 2 / 7,
                'budget_rcs_allocation': 1 / 7,
                'budget_dominant_domain': 'geometry',
            }
            summary = summarize_records([record])
            self.assertEqual(
                summary['budget_dominant_domain_counts'], {'geometry': 1}
            )
            write_diagnostic_report([record], summary, root / 'report')
            self.assertTrue((root / 'report' / 'per_object.csv').is_file())
            self.assertTrue((root / 'report' / 'summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
