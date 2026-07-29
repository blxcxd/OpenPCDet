"""Unified metrics for clean/adversarial 3D detection comparisons."""

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


def _detected(prediction: Dict[str, torch.Tensor], threshold: float) -> bool:
    scores = prediction['pred_scores']
    return scores.numel() > 0 and bool((scores.max() >= threshold).item())


@dataclass
class DetectionAttackMetrics:
    score_threshold: float = 0.5
    total_samples: int = 0
    originally_detected: int = 0
    successful_attacks: int = 0
    original_recall: float = 0.0
    attacked_recall: float = 0.0
    gt_count: float = 0.0
    max_abs_perturbation: float = 0.0
    sum_abs_perturbation: float = 0.0
    perturbation_values: float = 0.0

    def update_predictions(
        self,
        original_predictions: Sequence[Dict[str, torch.Tensor]],
        attacked_predictions: Sequence[Dict[str, torch.Tensor]],
        original_recall_dict: Dict,
        attacked_recall_dict: Dict,
    ) -> None:
        if len(original_predictions) != len(attacked_predictions):
            raise ValueError('clean and adversarial prediction batch sizes differ')

        for original, attacked in zip(original_predictions, attacked_predictions):
            if _detected(original, self.score_threshold):
                self.originally_detected += 1
                if not _detected(attacked, self.score_threshold):
                    self.successful_attacks += 1

        self.total_samples += len(original_predictions)
        self.original_recall += float(original_recall_dict.get('rcnn_0.5', 0))
        self.attacked_recall += float(attacked_recall_dict.get('rcnn_0.5', 0))
        self.gt_count += float(original_recall_dict.get('gt', 0))

    def update_perturbation(self, stats: Dict[str, float]) -> None:
        self.max_abs_perturbation = max(
            self.max_abs_perturbation,
            float(stats.get('max_abs_perturbation', 0.0)),
        )
        self.sum_abs_perturbation += float(
            stats.get('sum_abs_perturbation', 0.0)
        )
        self.perturbation_values += float(stats.get('perturbation_values', 0.0))

    def compute(self) -> Dict[str, float]:
        original_recall_rate = self.original_recall / max(self.gt_count, 1.0)
        attacked_recall_rate = self.attacked_recall / max(self.gt_count, 1.0)
        results = {
            'total_samples': self.total_samples,
            'original_recall': original_recall_rate,
            'attacked_recall': attacked_recall_rate,
            'attack_success_rate_sample': (
                self.successful_attacks / max(self.originally_detected, 1)
            ),
            'attack_success_rate_target': (
                (self.original_recall - self.attacked_recall)
                / max(self.original_recall, 1.0)
            ),
            'recall_drop': original_recall_rate - attacked_recall_rate,
            'max_abs_perturbation': self.max_abs_perturbation,
            'mean_abs_perturbation': (
                self.sum_abs_perturbation / max(self.perturbation_values, 1.0)
            ),
        }
        return results
