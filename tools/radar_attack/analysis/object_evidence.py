"""Object-level gradient diagnostics for radar point-cloud detectors.

The first implementation deliberately targets OpenPCDet's AnchorHeadSingle.
It keeps the analysis primitives independent from model loading so their
mathematics can be unit tested without CUDA or a detector checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr


DOMAIN_FEATURES = {
    'geometry': {'x', 'y', 'z'},
    'doppler': {
        'v_r',
        'v_r_comp',
        'velocity',
        'velocity_comp',
        'doppler',
        'doppler_comp',
    },
    'rcs': {'rcs', 'intensity', 'power'},
}


def reshape_anchor_cls_logits(
    cls_preds: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Convert ``[B, H, W, anchors * classes]`` logits to ``[B, A, C]``."""
    if cls_preds.ndim != 4:
        raise ValueError('AnchorHeadSingle cls_preds must have four dimensions')
    if num_classes <= 0 or cls_preds.shape[-1] % num_classes:
        raise ValueError('cls_preds final dimension is not divisible by classes')
    return cls_preds.reshape(cls_preds.shape[0], -1, num_classes)


def flatten_anchors(dense_head) -> torch.Tensor:
    """Flatten AnchorHeadSingle anchors in the same order as its logits."""
    anchors = dense_head.anchors
    if not isinstance(anchors, list):
        return anchors.reshape(-1, anchors.shape[-1])
    if getattr(dense_head, 'use_multihead', False):
        raise NotImplementedError(
            'object diagnostics do not yet support multi-head anchors'
        )
    return torch.cat(anchors, dim=-3).reshape(-1, anchors[0].shape[-1])


def candidate_anchor_indices(
    anchors: torch.Tensor,
    gt_box: torch.Tensor,
    margin: float = 1.0,
    fallback_topk: int = 32,
) -> torch.Tensor:
    """Select anchors whose centers lie inside an expanded oriented GT box.

    A nearest-center fallback makes the objective defined for very small
    objects or unusual anchor grids. Candidate identities are determined on
    the clean input and remain fixed during confidence probes.
    """
    if anchors.ndim != 2 or anchors.shape[1] < 3:
        raise ValueError('anchors must have shape [A, >=3]')
    if gt_box.ndim != 1 or gt_box.shape[0] < 7:
        raise ValueError('gt_box must have shape [>=7]')
    if margin < 0:
        raise ValueError('candidate margin must be non-negative')
    if fallback_topk <= 0:
        raise ValueError('fallback_topk must be positive')

    relative = anchors[:, :3] - gt_box[None, :3]
    cosine = torch.cos(gt_box[6])
    sine = torch.sin(gt_box[6])
    local_x = relative[:, 0] * cosine + relative[:, 1] * sine
    local_y = -relative[:, 0] * sine + relative[:, 1] * cosine
    local_z = relative[:, 2]
    half_size = gt_box[3:6] * 0.5 + float(margin)
    inside = (
        (local_x.abs() <= half_size[0])
        & (local_y.abs() <= half_size[1])
        & (local_z.abs() <= half_size[2])
    )
    selected = torch.nonzero(inside, as_tuple=False).flatten()
    if selected.numel():
        return selected

    normalized_distance = (
        (local_x / half_size[0].clamp_min(1e-12)).square()
        + (local_y / half_size[1].clamp_min(1e-12)).square()
        + (local_z / half_size[2].clamp_min(1e-12)).square()
    )
    count = min(int(fallback_topk), int(anchors.shape[0]))
    return torch.topk(
        normalized_distance, k=count, largest=False, sorted=True
    ).indices


def object_evidence(
    cls_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    class_index: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return a smooth aggregate of one object's candidate class logits."""
    if cls_logits.ndim != 2:
        raise ValueError('cls_logits must have shape [anchors, classes]')
    if candidate_indices.ndim != 1 or candidate_indices.numel() == 0:
        raise ValueError('candidate_indices must be a non-empty vector')
    if not 0 <= class_index < cls_logits.shape[1]:
        raise ValueError('class_index is outside cls_logits')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    selected = cls_logits[candidate_indices, int(class_index)]
    return float(temperature) * torch.logsumexp(
        selected / float(temperature), dim=0
    )


def feature_scales_from_statistics(
    statistics_path: Path | str,
    feature_names: Sequence[str],
) -> Dict[str, float]:
    """Load positive per-feature IQR values from analyze_features output."""
    path = Path(statistics_path).expanduser()
    with path.open('r', encoding='utf-8') as input_file:
        payload = json.load(input_file)
    features = payload.get('statistics', {}).get('features', {})
    scales = {}
    for name in feature_names:
        if name not in features:
            raise ValueError(f'feature statistics do not contain {name!r}')
        value = float(features[name]['iqr'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'feature {name!r} has invalid IQR {value}')
        scales[name] = value
    return scales


def _domain_columns(
    feature_names: Sequence[str], domain: str
) -> List[tuple[int, str]]:
    if domain not in DOMAIN_FEATURES:
        raise ValueError(f'unknown diagnostic domain {domain!r}')
    return [
        (column + 1, name)
        for column, name in enumerate(feature_names)
        if name.lower() in DOMAIN_FEATURES[domain]
    ]


def domain_sensitivities(
    gradient: torch.Tensor,
    point_mask: torch.Tensor,
    feature_names: Sequence[str],
    feature_scales: Mapping[str, float],
    domains: Sequence[str] = ('geometry', 'doppler', 'rcs'),
) -> Dict[str, float]:
    """Compute feature-scale-normalized first-order sensitivity for one object.

    The supplied scales may be dataset IQRs or exact local probe budgets.
    ``sum`` is the corresponding L-infinity box's first-order gain.
    ``mean_per_point`` removes the target point-count effect while retaining a
    domain's number of feature channels.
    """
    if gradient.ndim != 2 or gradient.shape[1] != len(feature_names) + 1:
        raise ValueError('gradient shape does not match feature_names')
    if point_mask.shape != (gradient.shape[0],):
        raise ValueError('point_mask must have shape [num_points]')
    point_count = int(point_mask.sum().item())
    result: Dict[str, float] = {}
    for domain in domains:
        columns = _domain_columns(feature_names, domain)
        if not columns:
            raise ValueError(
                f'domain {domain!r} is absent from {list(feature_names)}'
            )
        total = gradient.new_zeros(())
        for column, name in columns:
            total = total + (
                gradient[point_mask, column].abs().sum()
                * float(feature_scales[name])
            )
        total_value = float(total.detach().item())
        result[f'{domain}_sensitivity_sum'] = total_value
        result[f'{domain}_sensitivity_mean_per_point'] = (
            total_value / point_count if point_count else 0.0
        )

    doppler_columns = _domain_columns(feature_names, 'doppler')
    doppler_by_name = {name.lower(): column for column, name in doppler_columns}
    raw_name = next(
        (name for name in ('v_r', 'velocity', 'doppler') if name in doppler_by_name),
        None,
    )
    compensated_name = next(
        (
            name
            for name in ('v_r_comp', 'velocity_comp', 'doppler_comp')
            if name in doppler_by_name
        ),
        None,
    )
    if raw_name is not None and compensated_name is not None:
        raw_column = doppler_by_name[raw_name]
        compensated_column = doppler_by_name[compensated_name]
        raw_scale = float(feature_scales[feature_names[raw_column - 1]])
        compensated_scale = float(
            feature_scales[feature_names[compensated_column - 1]]
        )
        coupled = (
            gradient[point_mask, raw_column] * raw_scale
            + gradient[point_mask, compensated_column] * compensated_scale
        ).abs().sum()
        result['doppler_coupled_sensitivity_sum'] = float(
            coupled.detach().item()
        )
        result['doppler_coupled_sensitivity_mean_per_point'] = (
            float(coupled.detach().item()) / point_count if point_count else 0.0
        )
    return result


def sensitivity_allocations(
    sensitivities: Mapping[str, float],
    domains: Sequence[str] = ('geometry', 'doppler', 'rcs'),
) -> Dict[str, float]:
    """Convert non-negative domain sensitivity values to simple proportions."""
    values = np.asarray(
        [float(sensitivities[f'{domain}_sensitivity_sum']) for domain in domains],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('sensitivities must be finite and non-negative')
    total = float(values.sum())
    if total == 0:
        allocations = np.full(values.shape, 1.0 / len(values))
    else:
        allocations = values / total
    result = {
        f'{domain}_allocation': float(value)
        for domain, value in zip(domains, allocations)
    }
    positive = allocations[allocations > 0]
    result['allocation_entropy'] = float(-(positive * np.log(positive)).sum())
    result['dominant_domain'] = domains[int(np.argmax(allocations))]
    return result


def normalized_probe_budget(
    points: torch.Tensor,
    feature_names: Sequence[str],
    feature_scales: Mapping[str, float],
    fraction: float,
    domains: Sequence[str],
    domain_caps: Mapping[str, float | None],
) -> torch.Tensor:
    """Build one local diagnostic step, separate from formal attack budgets."""
    if fraction <= 0:
        raise ValueError('probe fraction must be positive')
    budget = points.new_zeros((1, points.shape[1]))
    for domain in domains:
        cap = domain_caps.get(domain)
        if cap is not None and cap <= 0:
            raise ValueError(f'{domain} probe cap must be positive')
        for column, name in _domain_columns(feature_names, domain):
            value = fraction * float(feature_scales[name])
            if cap is not None:
                value = min(value, float(cap))
            budget[0, column] = value
    return torch.where(
        budget > 0,
        torch.nextafter(budget, torch.zeros_like(budget)),
        budget,
    )


def first_order_evidence_drop(
    evidence_gradient: torch.Tensor,
    ascent_gradient: torch.Tensor,
    point_mask: torch.Tensor,
    probe_budget: torch.Tensor,
) -> float:
    """Predict evidence reduction from a sign step ascending another loss."""
    if evidence_gradient.shape != ascent_gradient.shape:
        raise ValueError('gradient shapes must match')
    if point_mask.shape != (evidence_gradient.shape[0],):
        raise ValueError('point_mask must have shape [num_points]')
    if probe_budget.shape != (1, evidence_gradient.shape[1]):
        raise ValueError('probe_budget shape does not match gradients')
    delta = ascent_gradient.sign() * probe_budget
    delta = delta * point_mask.to(delta.dtype).unsqueeze(1)
    predicted_change = (evidence_gradient * delta).sum()
    return float((-predicted_change).detach().item())


def hybrid_gradient_relationships(
    classification_gradient: torch.Tensor,
    localization_gradient: torch.Tensor,
    point_mask: torch.Tensor,
    feature_names: Sequence[str],
    domains: Sequence[str] = ('geometry', 'doppler', 'rcs'),
    zero_tolerance: float = 1e-12,
) -> Dict[str, float]:
    """Measure scale and conflict between two ascent gradients per domain.

    ``balance_beta_l1`` is the multiplier that gives the localization gradient
    the same L1 norm as the classification gradient for one object and domain.
    Sign statistics are computed only on attacked coordinates with at least
    one non-negligible component.
    """
    if classification_gradient.shape != localization_gradient.shape:
        raise ValueError('classification and localization gradients must match')
    if classification_gradient.ndim != 2:
        raise ValueError('gradients must have shape [num_points, features]')
    if classification_gradient.shape[1] != len(feature_names) + 1:
        raise ValueError('gradient shape does not match feature_names')
    if point_mask.shape != (classification_gradient.shape[0],):
        raise ValueError('point_mask must have shape [num_points]')
    if zero_tolerance <= 0:
        raise ValueError('zero_tolerance must be positive')

    result: Dict[str, float] = {}
    for domain in domains:
        columns = [column for column, _ in _domain_columns(feature_names, domain)]
        if not columns:
            raise ValueError(
                f'domain {domain!r} is absent from {list(feature_names)}'
            )
        classification = classification_gradient[point_mask][:, columns].reshape(-1)
        localization = localization_gradient[point_mask][:, columns].reshape(-1)
        classification_l1 = classification.abs().sum()
        localization_l1 = localization.abs().sum()
        classification_l2 = torch.linalg.vector_norm(classification)
        localization_l2 = torch.linalg.vector_norm(localization)
        balance_beta = classification_l1 / localization_l1.clamp_min(
            float(zero_tolerance)
        )

        classification_active = classification.abs() > float(zero_tolerance)
        localization_active = localization.abs() > float(zero_tolerance)
        active = classification_active | localization_active
        overlap = classification_active & localization_active
        overlap_count = int(overlap.sum().item())
        active_count = int(active.sum().item())
        if overlap_count:
            sign_agreement = (
                classification[overlap].sign() == localization[overlap].sign()
            ).to(classification.dtype).mean()
        else:
            sign_agreement = classification.new_zeros(())

        denominator = (classification_l2 * localization_l2).clamp_min(
            float(zero_tolerance)
        )
        cosine = torch.dot(classification, localization) / denominator
        combined_beta1 = classification + localization
        combined_balanced = classification + balance_beta * localization
        if active_count:
            beta1_sign_change = (
                combined_beta1[active].sign() != classification[active].sign()
            ).to(classification.dtype).mean()
            balanced_sign_change = (
                combined_balanced[active].sign()
                != classification[active].sign()
            ).to(classification.dtype).mean()
            overlap_fraction = classification.new_tensor(
                overlap_count / active_count
            )
        else:
            beta1_sign_change = classification.new_zeros(())
            balanced_sign_change = classification.new_zeros(())
            overlap_fraction = classification.new_zeros(())

        prefix = f'hybrid_{domain}_'
        result.update(
            {
                f'{prefix}classification_l1': float(
                    classification_l1.detach().item()
                ),
                f'{prefix}localization_l1': float(
                    localization_l1.detach().item()
                ),
                f'{prefix}localization_to_classification_l1_ratio': float(
                    (
                        localization_l1
                        / classification_l1.clamp_min(float(zero_tolerance))
                    ).detach().item()
                ),
                f'{prefix}balance_beta_l1': float(
                    balance_beta.detach().item()
                ),
                f'{prefix}cosine_similarity': float(cosine.detach().item()),
                f'{prefix}sign_agreement': float(sign_agreement.detach().item()),
                f'{prefix}sign_overlap_fraction': float(
                    overlap_fraction.detach().item()
                ),
                f'{prefix}beta1_sign_change_fraction': float(
                    beta1_sign_change.detach().item()
                ),
                f'{prefix}balanced_sign_change_fraction': float(
                    balanced_sign_change.detach().item()
                ),
                f'{prefix}classification_near_zero': float(
                    (classification_l1 <= float(zero_tolerance)).item()
                ),
                f'{prefix}localization_near_zero': float(
                    (localization_l1 <= float(zero_tolerance)).item()
                ),
            }
        )
    return result


def summarize_records(records: Sequence[Mapping]) -> Dict:
    """Aggregate loss comparison, allocations, classes, and correlations."""
    records = list(records)
    summary: Dict = {'diagnosed_objects': len(records)}
    if not records:
        return summary

    numeric_groups = [
        'training_loss_evidence_drop',
        'object_loss_evidence_drop',
        'training_loss_predicted_drop',
        'object_loss_predicted_drop',
        'geometry_sensitivity_sum',
        'doppler_sensitivity_sum',
        'rcs_sensitivity_sum',
        'geometry_allocation',
        'doppler_allocation',
        'rcs_allocation',
    ]
    for prefix in ('iqr_', 'budget_'):
        for domain in ('geometry', 'doppler', 'rcs'):
            for suffix in (
                'sensitivity_sum',
                'sensitivity_mean_per_point',
                'allocation',
            ):
                key = f'{prefix}{domain}_{suffix}'
                if key in records[0]:
                    numeric_groups.append(key)
    numeric_groups.extend(
        key
        for key, value in records[0].items()
        if key.startswith('hybrid_') and isinstance(value, (int, float))
    )
    numeric_groups = list(dict.fromkeys(numeric_groups))
    aggregates = {}
    for key in numeric_groups:
        values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        aggregates[key] = {
            'count': int(finite.size),
            'mean': float(finite.mean()) if finite.size else None,
            'median': float(np.median(finite)) if finite.size else None,
            'std': float(finite.std()) if finite.size else None,
            'q10': float(np.quantile(finite, 0.1)) if finite.size else None,
            'q90': float(np.quantile(finite, 0.9)) if finite.size else None,
        }
    summary['aggregates'] = aggregates
    summary['object_loss_better_fraction'] = float(
        np.mean(
            [
                float(row['object_loss_evidence_drop'])
                > float(row['training_loss_evidence_drop'])
                for row in records
            ]
        )
    )
    dominant_counts = {}
    for row in records:
        name = str(row['dominant_domain'])
        dominant_counts[name] = dominant_counts.get(name, 0) + 1
    summary['dominant_domain_counts'] = dominant_counts
    for prefix in ('iqr_', 'budget_'):
        field = f'{prefix}dominant_domain'
        if field not in records[0]:
            continue
        counts = {}
        for row in records:
            name = str(row[field])
            counts[name] = counts.get(name, 0) + 1
        summary[f'{prefix}dominant_domain_counts'] = counts

    correlations = {}
    predictors = [
        'distance_xy',
        'active_point_count',
        'mean_abs_v_r_comp',
        'clean_detection_iou',
        'clean_detection_score',
    ]
    predictors = [key for key in predictors if key in records[0]]
    outcomes = [
        'geometry_sensitivity_sum',
        'doppler_sensitivity_sum',
        'rcs_sensitivity_sum',
        'geometry_allocation',
        'doppler_allocation',
        'rcs_allocation',
    ]
    for prefix in ('iqr_', 'budget_'):
        for domain in ('geometry', 'doppler', 'rcs'):
            for suffix in ('sensitivity_sum', 'allocation'):
                key = f'{prefix}{domain}_{suffix}'
                if key in records[0]:
                    outcomes.append(key)
    outcomes.extend(
        key
        for key, value in records[0].items()
        if key.startswith('hybrid_')
        and isinstance(value, (int, float))
        and (
            key.endswith('balance_beta_l1')
            or key.endswith('cosine_similarity')
            or key.endswith('sign_agreement')
            or key.endswith('beta1_sign_change_fraction')
        )
    )
    outcomes = list(dict.fromkeys(outcomes))
    for predictor in predictors:
        for outcome in outcomes:
            x = np.asarray([float(row[predictor]) for row in records])
            y = np.asarray([float(row[outcome]) for row in records])
            finite = np.isfinite(x) & np.isfinite(y)
            key = f'{outcome}_vs_{predictor}'
            if finite.sum() < 3 or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
                correlations[key] = {'rho': None, 'pvalue': None, 'count': int(finite.sum())}
            else:
                result = spearmanr(x[finite], y[finite])
                correlations[key] = {
                    'rho': float(result.statistic),
                    'pvalue': float(result.pvalue),
                    'count': int(finite.sum()),
                }
    summary['spearman_correlations'] = correlations

    class_summaries = {}
    for class_name in sorted({str(row['class_name']) for row in records}):
        subset = [row for row in records if str(row['class_name']) == class_name]
        class_summary = {'count': len(subset)}
        for key in outcomes:
            values = np.asarray([float(row[key]) for row in subset])
            class_summary[key] = {
                'mean': float(values.mean()),
                'median': float(np.median(values)),
            }
        class_summaries[class_name] = class_summary
    summary['by_class'] = class_summaries
    return summary


def write_diagnostic_report(
    records: Sequence[Mapping], summary: Mapping, output_dir: Path | str
) -> None:
    """Write deterministic per-object CSV and aggregate JSON outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if records:
        fieldnames = list(records[0].keys())
        with (output_dir / 'per_object.csv').open(
            'w', newline='', encoding='utf-8'
        ) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        (output_dir / 'per_object.csv').write_text('', encoding='utf-8')
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False, allow_nan=False)
