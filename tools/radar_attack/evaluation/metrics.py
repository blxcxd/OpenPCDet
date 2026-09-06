"""Unified metrics for clean/adversarial 3D detection comparisons."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .object_endpoints import OUTCOME_NAMES


ENDPOINT_METRIC_FIELDS = (
    'max_iou_drop',
    'match_score_drop',
    'object_evidence_drop',
    'prediction_center_shift',
    'center_error_increase',
    'prediction_size_l1_shift',
    'prediction_yaw_shift',
    'clean_iou_margin',
    'adversarial_iou_margin',
    'num_attacked_points',
    'pillar_reassignment_rate',
    'mean_xyz_l2_displacement',
    'max_xyz_l2_displacement',
    'active_current_sweep_point_count',
)


def _endpoint_summary(records: Sequence[Dict]) -> Dict:
    summary = {'evaluated_clean_objects': len(records)}
    for key in ENDPOINT_METRIC_FIELDS:
        values = np.asarray(
            [float(record[key]) for record in records if record.get(key) is not None],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        summary[key] = {
            'count': int(values.size),
            'mean': float(values.mean()) if values.size else None,
            'median': float(np.median(values)) if values.size else None,
            'q10': float(np.quantile(values, 0.1)) if values.size else None,
            'q90': float(np.quantile(values, 0.9)) if values.size else None,
            'min': float(values.min()) if values.size else None,
            'max': float(values.max()) if values.size else None,
            'positive_fraction': float(np.mean(values > 0)) if values.size else None,
        }
    return summary


def _outcome_rates(counts: Dict[str, int], eligible: int) -> Dict[str, float]:
    denominator = max(int(eligible), 1)
    return {
        'object_failure_asr': 1.0 - counts['still_correct'] / denominator
        if eligible else 0.0,
        'pure_hiding_asr': counts['pure_hiding'] / denominator,
        'misclassification_rate': counts['misclassification'] / denominator,
        'localization_failure_rate': counts['localization_failure'] / denominator,
        'still_correct_rate': counts['still_correct'] / denominator,
    }


@dataclass
class DetectionAttackMetrics:
    total_samples: int = 0
    original_recall: float = 0.0
    attacked_recall: float = 0.0
    gt_count: float = 0.0
    max_abs_perturbation: float = 0.0
    sum_abs_perturbation: float = 0.0
    perturbation_values: float = 0.0
    object_targets: int = 0
    diagnostic_sums: Dict[str, float] = field(default_factory=dict)
    diagnostic_maxima: Dict[str, float] = field(default_factory=dict)
    diagnostic_updates: int = 0
    object_endpoint_records: list[Dict] = field(default_factory=list)

    def update_predictions(
        self,
        original_predictions: Sequence[Dict],
        attacked_predictions: Sequence[Dict],
        original_recall_dict: Dict,
        attacked_recall_dict: Dict,
    ) -> None:
        if len(original_predictions) != len(attacked_predictions):
            raise ValueError('clean and adversarial prediction batch sizes differ')
        self.total_samples += len(original_predictions)
        self.original_recall += float(original_recall_dict.get('rcnn_0.5', 0))
        self.attacked_recall += float(attacked_recall_dict.get('rcnn_0.5', 0))
        self.gt_count += float(original_recall_dict.get('gt', 0))

    def update_perturbation(self, stats: Dict[str, float]) -> None:
        self.max_abs_perturbation = max(
            self.max_abs_perturbation,
            float(stats.get('max_abs_perturbation', 0.0)),
        )
        self.sum_abs_perturbation += float(stats.get('sum_abs_perturbation', 0.0))
        self.perturbation_values += float(stats.get('perturbation_values', 0.0))
        core_keys = {
            'max_abs_perturbation', 'mean_abs_perturbation',
            'sum_abs_perturbation', 'perturbation_values',
        }
        maximum_keys = {
            'measurement_max_abs_delta_range',
            'measurement_max_abs_delta_azimuth_rad',
            'measurement_max_abs_delta_elevation_rad',
            'measurement_max_xyz_l2',
            'point_max_xyz_l2',
            'temporal_max_alignment_error_m',
            'temporal_shared_delta_max_abs',
            'temporal_parameter_delta_max_abs',
        }
        for key, value in stats.items():
            if key in maximum_keys:
                self.diagnostic_maxima[key] = max(
                    self.diagnostic_maxima.get(key, 0.0), float(value)
                )
            elif key not in core_keys:
                self.diagnostic_sums[key] = self.diagnostic_sums.get(key, 0.0) + float(value)
        self.diagnostic_updates += 1

    def update_object_endpoints(self, target_count: int, records: Sequence[Dict]) -> None:
        self.object_targets += int(target_count)
        self.object_endpoint_records.extend(dict(record) for record in records)

    def write_object_endpoints(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.object_endpoint_records:
            path.write_text('', encoding='utf-8')
            return
        fieldnames = list(self.object_endpoint_records[0])
        with path.open('w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.object_endpoint_records)

    def _object_outcomes(self) -> Dict:
        counts = {name: 0 for name in OUTCOME_NAMES}
        by_class = {}
        for record in self.object_endpoint_records:
            outcome = record['outcome']
            counts[outcome] += 1
            class_name = str(record['class_name'])
            class_entry = by_class.setdefault(
                class_name, {'eligible_clean_objects': 0, 'counts': {name: 0 for name in OUTCOME_NAMES}}
            )
            class_entry['eligible_clean_objects'] += 1
            class_entry['counts'][outcome] += 1
        for class_entry in by_class.values():
            class_entry['rates'] = _outcome_rates(
                class_entry['counts'], class_entry['eligible_clean_objects']
            )
        eligible = len(self.object_endpoint_records)
        return {
            'target_objects': self.object_targets,
            'eligible_clean_objects': eligible,
            'counts': counts,
            'rates': _outcome_rates(counts, eligible),
            'by_class': by_class,
        }

    def compute(self) -> Dict:
        original_recall_rate = self.original_recall / max(self.gt_count, 1.0)
        attacked_recall_rate = self.attacked_recall / max(self.gt_count, 1.0)
        outcomes = self._object_outcomes()
        results = {
            'total_samples': self.total_samples,
            'original_recall': original_recall_rate,
            'attacked_recall': attacked_recall_rate,
            'recall_drop': original_recall_rate - attacked_recall_rate,
            'max_abs_perturbation': self.max_abs_perturbation,
            'mean_abs_perturbation': self.sum_abs_perturbation / max(self.perturbation_values, 1.0),
            'object_outcomes': outcomes,
            **outcomes['rates'],
            'object_endpoint_metrics': _endpoint_summary(self.object_endpoint_records),
        }
        if self.diagnostic_sums:
            diagnostics = {
                key: value / max(self.diagnostic_updates, 1)
                for key, value in self.diagnostic_sums.items()
            }
            diagnostics.update(self.diagnostic_maxima)
            sums = self.diagnostic_sums
            attacked_points = sums.get('iadv_attacked_points', 0.0)
            valid_targets = sums.get('iadv_valid_targets', 0.0)
            groups = sums.get('iadv_groups', 0.0)
            singleton_groups = sums.get('iadv_singleton_groups', 0.0)
            fallback_points = sums.get('reflectivity_fallback_points', 0.0)
            if 'iadv_valid_targets' in sums:
                diagnostics['iadv_valid_targets'] = valid_targets
                diagnostics['iadv_mean_points_per_target'] = attacked_points / max(valid_targets, 1.0)
                diagnostics['iadv_mean_groups_per_target'] = groups / max(valid_targets, 1.0)
                diagnostics['iadv_mean_points_per_group'] = attacked_points / max(groups, 1.0)
                diagnostics['iadv_singleton_group_ratio'] = singleton_groups / max(groups, 1.0)
                diagnostics['iadv_pca_fallback_rate'] = fallback_points / max(attacked_points, 1.0)
            for total_key in (
                'iadv_cross_target_neighbors', 'iadv_nonfinite_gradient_steps',
                'reflectivity_cross_target_neighbors', 'object_evidence_targets',
                'object_evidence_candidate_anchors', 'object_hybrid_localization_anchors',
                'object_iou_s_targets', 'object_iou_s_candidate_anchors',
                'measurement_batches', 'measurement_batches_with_attack_points',
                'measurement_target_points', 'measurement_current_target_points',
                'measurement_historical_target_points',
                'measurement_clean_active_target_points',
                'measurement_clean_active_current_target_points',
                'measurement_clean_active_historical_target_points',
                'measurement_zero_range_target_points',
                'measurement_modified_current_points',
                'measurement_historical_modification_count',
                'measurement_non_target_modification_count',
                'measurement_non_geometry_modification_count',
                'measurement_out_of_range_backtracks',
                'measurement_out_of_range_rejections',
                'measurement_nonfinite_gradient_steps',
                'measurement_zero_gradient_steps',
                'measurement_xyz_l2_sum',
                'temporal_frames',
                'temporal_shared_groups',
                'temporal_active_shared_groups',
                'temporal_parameter_groups',
                'temporal_active_parameter_groups',
                'temporal_current_label_targets',
                'temporal_matched_current_targets',
                'temporal_target_points',
                'temporal_current_target_points',
                'temporal_historical_target_points',
                'temporal_attack_target_points',
                'temporal_attack_current_points',
                'temporal_attack_historical_points',
                'temporal_attack_clean_active_points',
                'temporal_out_of_range_backtrack_points',
                'temporal_out_of_range_backtrack_groups',
                'temporal_out_of_range_rejected_groups',
                'point_attack_mask_points',
                'point_modified_points',
                'point_non_mask_modification_count',
                'point_non_selected_feature_modification_count',
                'point_xyz_l2_sum',
                'point_xyz_l2_count',
                'current_sweep_target_points',
                'current_sweep_time0_target_points',
                'current_sweep_history_target_points',
                'current_sweep_clean_active_target_points',
            ):
                if total_key in sums:
                    diagnostics[total_key] = sums[total_key]
            measurement_active_key = (
                'measurement_clean_active_target_points'
                if 'measurement_clean_active_target_points' in sums
                else 'measurement_clean_active_current_target_points'
            )
            if measurement_active_key in sums:
                diagnostics['measurement_mean_xyz_l2'] = (
                    sums.get('measurement_xyz_l2_sum', 0.0)
                    / max(sums[measurement_active_key], 1.0)
                )
                for suffix in (
                    'r0_10', 'r10_20', 'r20_30', 'r30_50', 'r50_inf'
                ):
                    count_key = f'measurement_xyz_l2_count_{suffix}'
                    sum_key = f'measurement_xyz_l2_sum_{suffix}'
                    count = sums.get(count_key, 0.0)
                    diagnostics[count_key] = count
                    diagnostics[
                        f'measurement_mean_xyz_l2_{suffix}'
                    ] = sums.get(sum_key, 0.0) / max(count, 1.0)
            if 'point_xyz_l2_count' in sums:
                diagnostics['point_mean_xyz_l2'] = (
                    sums.get('point_xyz_l2_sum', 0.0)
                    / max(sums['point_xyz_l2_count'], 1.0)
                )
            if 'object_evidence_targets' in sums:
                diagnostics['object_evidence_mean_candidates_per_target'] = (
                    sums.get('object_evidence_candidate_anchors', 0.0)
                    / max(sums['object_evidence_targets'], 1.0)
                )
            if 'object_iou_s_targets' in sums:
                diagnostics['object_iou_s_mean_candidates_per_target'] = (
                    sums.get('object_iou_s_candidate_anchors', 0.0)
                    / max(sums['object_iou_s_targets'], 1.0)
                )
            results['attack_diagnostics'] = diagnostics
        return results
