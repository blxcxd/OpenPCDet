"""Clean-membership-frozen summaries of where XYZ perturbation is spent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch


MIGRATION_GROUPS = (
    'target_current',
    'target_history',
    'non_target_background',
)
REPORTING_THRESHOLDS_M = {
    '1cm': 0.01,
    '5cm': 0.05,
    '10cm': 0.10,
}


@dataclass
class AttackMigrationAccumulator:
    """Aggregate XYZ L2 displacement by clean target/time membership."""

    change_tolerance_m: float = 1e-9
    modified_l2: Dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {name: [] for name in MIGRATION_GROUPS}
    )
    total_points: Dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in MIGRATION_GROUPS}
    )

    def update(
        self,
        clean_points: torch.Tensor,
        adversarial_points: torch.Tensor,
        target_mask: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> None:
        if clean_points.shape != adversarial_points.shape:
            raise ValueError('clean and adversarial points must have equal shape')
        point_count = clean_points.shape[0]
        if target_mask.shape != (point_count,):
            raise ValueError('target_mask must have shape [N]')
        if current_mask.shape != (point_count,):
            raise ValueError('current_mask must have shape [N]')
        if not torch.equal(clean_points[:, 0], adversarial_points[:, 0]):
            raise ValueError('point order or batch indices changed during attack')

        target_mask = target_mask.detach().bool()
        current_mask = current_mask.detach().bool()
        masks = {
            'target_current': target_mask & current_mask,
            'target_history': target_mask & ~current_mask,
            'non_target_background': ~target_mask,
        }
        if sum(int(mask.sum().item()) for mask in masks.values()) != point_count:
            raise RuntimeError('attack migration groups must partition all points')

        delta_l2 = torch.linalg.vector_norm(
            adversarial_points[:, 1:4] - clean_points[:, 1:4], dim=1
        ).detach().double().cpu().numpy()
        for name, mask in masks.items():
            selected = mask.cpu().numpy()
            values = delta_l2[selected]
            self.total_points[name] += int(values.size)
            self.modified_l2[name].append(
                values[values > float(self.change_tolerance_m)]
            )

    def compute(self) -> Dict:
        groups = {}
        for name in MIGRATION_GROUPS:
            chunks = self.modified_l2[name]
            values = (
                np.concatenate(chunks)
                if chunks else np.empty(0, dtype=np.float64)
            )
            count = int(values.size)
            total = self.total_points[name]
            threshold_counts = {
                label: int(np.sum(values > threshold))
                for label, threshold in REPORTING_THRESHOLDS_M.items()
            }
            groups[name] = {
                'total_points': total,
                'N_modified': count,
                'modified_fraction': count / total if total else None,
                'mean_l2_m': float(values.mean()) if count else None,
                'p95_l2_m': float(np.quantile(values, 0.95)) if count else None,
                'max_l2_m': float(values.max()) if count else None,
                'sum_l2_m': float(values.sum()) if count else 0.0,
                **{
                    f'N_gt_{label}': threshold_count
                    for label, threshold_count in threshold_counts.items()
                },
                **{
                    f'fraction_gt_{label}': (
                        threshold_count / total if total else None
                    )
                    for label, threshold_count in threshold_counts.items()
                },
            }
        return {
            'membership': 'frozen from clean points',
            'change_tolerance_m': float(self.change_tolerance_m),
            'groups': groups,
        }
