"""Target-level diagnostics for current-sweep Radar geometry attacks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml

from ..attacks.iadv import assign_points_to_oriented_boxes
from ..attacks.measurement import (
    cartesian_to_radar_measurement,
    current_sweep_mask,
    model_active_point_mask,
)


@dataclass
class TargetScreeningContext:
    """Frozen clean target/point assignments shared by both attack spaces."""

    target_mask: torch.Tensor
    point_gt_rows: torch.Tensor
    current_mask: torch.Tensor
    active_mask: torch.Tensor
    records: Dict[Tuple[int, int], Dict]
    voxelizer: object
    clean_points: torch.Tensor
    source_xyz: torch.Tensor | None = None
    source_to_reference_rotations: torch.Tensor | None = None


def set_temporal_screening_assignments(
    context: TargetScreeningContext,
    point_gt_rows: torch.Tensor,
    source_xyz: torch.Tensor,
    source_to_reference_rotations: torch.Tensor,
) -> None:
    """Replace current-box assignments with source-sweep track assignments."""
    if point_gt_rows.shape != (context.clean_points.shape[0],):
        raise ValueError('point_gt_rows must align with screening points')
    point_gt_rows = point_gt_rows.to(
        device=context.clean_points.device, dtype=torch.long
    )
    if source_xyz.shape != (context.clean_points.shape[0], 3):
        raise ValueError('source_xyz must have shape [N, 3]')
    if source_to_reference_rotations.shape != (
        context.clean_points.shape[0], 3, 3
    ):
        raise ValueError(
            'source_to_reference_rotations must have shape [N, 3, 3]'
        )
    context.point_gt_rows = point_gt_rows
    context.target_mask = point_gt_rows >= 0
    context.source_xyz = source_xyz.to(
        device=context.clean_points.device,
        dtype=context.clean_points.dtype,
    )
    context.source_to_reference_rotations = (
        source_to_reference_rotations.to(
            device=context.clean_points.device,
            dtype=context.clean_points.dtype,
        )
    )
    batch_indices = context.clean_points[:, 0].long()
    for (batch_index, gt_row), record in context.records.items():
        object_mask = (
            (batch_indices == batch_index) & (point_gt_rows == gt_row)
        )
        time0_mask = object_mask & context.current_mask
        active_object = object_mask & context.active_mask
        active_time0 = time0_mask & context.active_mask
        active_total = int(active_object.sum().item())
        active_current = int(active_time0.sum().item())
        record.update({
            'num_object_points_total': int(object_mask.sum().item()),
            'num_time0_points': int(time0_mask.sum().item()),
            'num_history_points': int(
                (object_mask & ~context.current_mask).sum().item()
            ),
            'num_active_object_points_total': active_total,
            'num_active_time0_points': active_current,
            'current_sweep_point_ratio': (
                active_current / active_total if active_total else 0.0
            ),
        })


def build_target_screening_context(
    points: torch.Tensor,
    batch_dict: Mapping,
    clean_records_by_batch: Mapping[int, Sequence[Mapping]],
    feature_names: Sequence[str],
    voxelizer,
    box_margin: float = 0.0,
) -> TargetScreeningContext:
    """Assign every point to one evaluator-eligible clean target."""
    point_gt_rows = torch.full(
        (points.shape[0],), -1, dtype=torch.long, device=points.device
    )
    batch_indices = points[:, 0].long()
    current = current_sweep_mask(points, feature_names)
    active = model_active_point_mask(points, voxelizer)
    records = {}

    for batch_index in range(int(batch_dict['batch_size'])):
        clean_records = list(clean_records_by_batch.get(batch_index, ()))
        if not clean_records:
            continue
        sample_indices = torch.nonzero(
            batch_indices == batch_index, as_tuple=False
        ).flatten()
        gt_rows = [int(record['gt_row']) for record in clean_records]
        boxes = batch_dict['gt_boxes'][batch_index, gt_rows, :7].clone()
        assignments = assign_points_to_oriented_boxes(
            points[sample_indices, 1:4], boxes, margin=box_margin
        )
        assigned = assignments >= 0
        if assigned.any():
            row_tensor = torch.as_tensor(
                gt_rows, dtype=torch.long, device=points.device
            )
            point_gt_rows[sample_indices[assigned]] = row_tensor[
                assignments[assigned]
            ]

        for clean_record in clean_records:
            gt_row = int(clean_record['gt_row'])
            object_mask = (
                (batch_indices == batch_index) & (point_gt_rows == gt_row)
            )
            time0_mask = object_mask & current
            active_object_mask = object_mask & active
            active_time0_mask = time0_mask & active
            gt_box = batch_dict['gt_boxes'][batch_index, gt_row]
            active_total = int(active_object_mask.sum().item())
            active_time0 = int(active_time0_mask.sum().item())
            records[(batch_index, gt_row)] = {
                'frame_id': str(clean_record['frame_id']),
                'batch_index': batch_index,
                'object_id': int(clean_record['object_id']),
                'gt_row': gt_row,
                'class_name': str(clean_record['class_name']),
                'distance': float(
                    torch.linalg.vector_norm(gt_box[:3]).item()
                ),
                'clean_score': float(clean_record['clean_match_score']),
                'clean_iou': float(clean_record['clean_max_iou']),
                'num_object_points_total': int(object_mask.sum().item()),
                'num_time0_points': int(time0_mask.sum().item()),
                'num_history_points': int(
                    (object_mask & ~current).sum().item()
                ),
                'num_active_object_points_total': active_total,
                'num_active_time0_points': active_time0,
                'current_sweep_point_ratio': (
                    active_time0 / active_total if active_total else 0.0
                ),
            }

    return TargetScreeningContext(
        target_mask=point_gt_rows >= 0,
        point_gt_rows=point_gt_rows,
        current_mask=current,
        active_mask=active,
        records=records,
        voxelizer=voxelizer,
        clean_points=points.detach().clone(),
    )


def _hard_pillar_coordinates(points: torch.Tensor, voxelizer) -> torch.Tensor:
    """Use the detector's hard grid convention to obtain [b,z,y,x] IDs."""
    range_min = points.new_tensor(voxelizer.point_cloud_range[:3])
    range_max = points.new_tensor(voxelizer.point_cloud_range[3:])
    voxel_size = points.new_tensor(voxelizer.voxel_size)
    xyz = points[:, 1:4]
    coords_xyz = torch.floor((xyz - range_min) / voxel_size).long()
    valid = ((xyz >= range_min) & (xyz < range_max)).all(dim=1)
    coordinates = torch.full(
        (points.shape[0], 4), -1, dtype=torch.long, device=points.device
    )
    coordinates[valid, 0] = points[valid, 0].long()
    coordinates[valid, 1:] = coords_xyz[valid][:, [2, 1, 0]]
    return coordinates


def _unique_coordinate_set(coordinates: torch.Tensor) -> set:
    return {tuple(row) for row in coordinates.detach().cpu().tolist()}


def target_attack_diagnostics(
    context: TargetScreeningContext,
    adversarial_points: torch.Tensor,
    attack_mask: torch.Tensor,
) -> Dict[Tuple[int, int], Dict]:
    """Measure displacement and true hard-pillar reassignment per target."""
    clean = context.clean_points
    if adversarial_points.shape != clean.shape:
        raise ValueError('clean/adversarial point shapes differ')
    if attack_mask.shape != (clean.shape[0],):
        raise ValueError('attack_mask must have shape [N]')
    attack_mask = attack_mask.to(device=clean.device, dtype=torch.bool)

    clean_coords = _hard_pillar_coordinates(clean, context.voxelizer)
    adversarial_coords = _hard_pillar_coordinates(
        adversarial_points, context.voxelizer
    )
    reassigned = (clean_coords != adversarial_coords).any(dim=1)
    xyz_l2 = torch.linalg.vector_norm(
        adversarial_points[:, 1:4] - clean[:, 1:4], dim=1
    )
    if context.source_xyz is None:
        clean_measurement_xyz = clean[:, 1:4]
        adversarial_measurement_xyz = adversarial_points[:, 1:4]
    else:
        reference_displacement = (
            adversarial_points[:, 1:4] - clean[:, 1:4]
        )
        inverse_rotations = context.source_to_reference_rotations.transpose(
            1, 2
        )
        source_displacement = torch.bmm(
            inverse_rotations, reference_displacement.unsqueeze(-1)
        ).squeeze(-1)
        clean_measurement_xyz = context.source_xyz
        adversarial_measurement_xyz = (
            context.source_xyz + source_displacement
        )
    clean_measurement = cartesian_to_radar_measurement(
        clean_measurement_xyz
    )
    adversarial_measurement = cartesian_to_radar_measurement(
        adversarial_measurement_xyz
    )
    measurement_delta = adversarial_measurement - clean_measurement
    measurement_delta[:, 1] = torch.atan2(
        torch.sin(measurement_delta[:, 1]),
        torch.cos(measurement_delta[:, 1]),
    )
    absolute_delta = measurement_delta.abs()

    output = {}
    for key, clean_record in context.records.items():
        batch_index, gt_row = key
        object_attack_mask = (
            attack_mask
            & (clean[:, 0].long() == batch_index)
            & (context.point_gt_rows == gt_row)
        )
        attacked_count = int(object_attack_mask.sum().item())
        reassigned_count = int(
            (object_attack_mask & reassigned).sum().item()
        )
        clean_set = _unique_coordinate_set(clean_coords[object_attack_mask])
        adversarial_set = _unique_coordinate_set(
            adversarial_coords[object_attack_mask]
        )
        displacement = xyz_l2[object_attack_mask]
        delta = absolute_delta[object_attack_mask]
        output[key] = {
            **clean_record,
            'clean_point_count': clean_record['num_object_points_total'],
            'current_sweep_point_count': clean_record['num_time0_points'],
            'active_current_sweep_point_count': clean_record[
                'num_active_time0_points'
            ],
            'num_attacked_points': attacked_count,
            'num_reassigned_points': reassigned_count,
            'pillar_reassignment_rate': (
                reassigned_count / attacked_count if attacked_count else 0.0
            ),
            'num_unique_clean_pillars': len(clean_set),
            'num_unique_adv_pillars': len(adversarial_set),
            'num_empty_clean_pillars_after_attack': len(
                clean_set - adversarial_set
            ),
            'num_new_adv_pillars': len(adversarial_set - clean_set),
            'mean_xyz_l2_displacement': (
                float(displacement.mean().item())
                if displacement.numel() else 0.0
            ),
            'max_xyz_l2_displacement': (
                float(displacement.max().item())
                if displacement.numel() else 0.0
            ),
            'mean_abs_delta_range': (
                float(delta[:, 0].mean().item()) if delta.numel() else 0.0
            ),
            'max_abs_delta_range': (
                float(delta[:, 0].max().item()) if delta.numel() else 0.0
            ),
            'mean_abs_delta_azimuth_deg': (
                float(torch.rad2deg(delta[:, 1]).mean().item())
                if delta.numel() else 0.0
            ),
            'max_abs_delta_azimuth_deg': (
                float(torch.rad2deg(delta[:, 1]).max().item())
                if delta.numel() else 0.0
            ),
            'mean_abs_delta_elevation_deg': (
                float(torch.rad2deg(delta[:, 2]).mean().item())
                if delta.numel() else 0.0
            ),
            'max_abs_delta_elevation_deg': (
                float(torch.rad2deg(delta[:, 2]).max().item())
                if delta.numel() else 0.0
            ),
        }
    return output


def merge_target_diagnostics(
    endpoint_records: Sequence[Dict],
    diagnostics: Mapping[Tuple[int, int], Mapping],
) -> None:
    """Add clean point/reassignment diagnostics and convenient aliases."""
    for record in endpoint_records:
        key = (int(record['batch_index']), int(record['gt_row']))
        if key not in diagnostics:
            raise RuntimeError(
                f'missing point diagnostics for clean target {key}'
            )
        record.update(diagnostics[key])
        record.update({
            'adv_score': record['adversarial_match_score'],
            'score_drop': record['match_score_drop'],
            'adv_iou': record['adversarial_max_iou'],
            'iou_drop': record['max_iou_drop'],
            'clean_evidence': record['clean_object_evidence'],
            'adv_evidence': record['adversarial_object_evidence'],
            'evidence_drop': record['object_evidence_drop'],
            'object_attack_success': bool(record['object_failure']),
            'center_error': record['adversarial_center_error'],
        })


def _distribution(values: Sequence[float]) -> Dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in (
            'mean', 'median', 'p10', 'p25', 'p75', 'p90', 'min', 'max'
        )}
    return {
        'mean': float(array.mean()),
        'median': float(np.median(array)),
        'p10': float(np.quantile(array, 0.10)),
        'p25': float(np.quantile(array, 0.25)),
        'p75': float(np.quantile(array, 0.75)),
        'p90': float(np.quantile(array, 0.90)),
        'min': float(array.min()),
        'max': float(array.max()),
    }


def summarize_current_sweep(records: Sequence[Mapping]) -> Dict:
    fields = (
        'num_object_points_total',
        'num_time0_points',
        'num_history_points',
        'num_active_object_points_total',
        'num_active_time0_points',
        'current_sweep_point_ratio',
    )
    summary = {
        'clean_detected_targets': len(records),
        'distributions': {
            field: _distribution([record[field] for record in records])
            for field in fields
        },
    }
    active = np.asarray(
        [record['num_active_time0_points'] for record in records],
        dtype=np.float64,
    )
    for threshold in (0, 1, 2, 5):
        summary[f'active_time0_le_{threshold}_fraction'] = (
            float(np.mean(active <= threshold)) if active.size else 0.0
        )
    return summary


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(records: Sequence[Mapping], x_key: str, y_key: str) -> Dict:
    pairs = [
        (float(record[x_key]), float(record[y_key]))
        for record in records
        if record.get(x_key) is not None and record.get(y_key) is not None
        and np.isfinite(float(record[x_key]))
        and np.isfinite(float(record[y_key]))
    ]
    if len(pairs) < 2:
        return {'count': len(pairs), 'rho': None}
    x, y = np.asarray(pairs, dtype=np.float64).T
    x_rank, y_rank = _rankdata(x), _rankdata(y)
    if x_rank.std() == 0 or y_rank.std() == 0:
        return {'count': len(pairs), 'rho': None}
    return {
        'count': len(pairs),
        'rho': float(np.corrcoef(x_rank, y_rank)[0, 1]),
    }


def target_correlations(records: Sequence[Mapping]) -> Dict:
    pairs = (
        ('pillar_reassignment_rate', 'iou_drop'),
        ('pillar_reassignment_rate', 'score_drop'),
        ('pillar_reassignment_rate', 'object_attack_success'),
        ('active_current_sweep_point_count', 'iou_drop'),
        ('active_current_sweep_point_count', 'object_attack_success'),
        ('distance', 'mean_xyz_l2_displacement'),
        ('distance', 'iou_drop'),
    )
    correlations = {
        f'{x_key}_vs_{y_key}': _spearman(records, x_key, y_key)
        for x_key, y_key in pairs
    }
    successful = [
        record['pillar_reassignment_rate'] for record in records
        if record.get('object_attack_success')
    ]
    failed = [
        record['pillar_reassignment_rate'] for record in records
        if not record.get('object_attack_success')
    ]
    correlations['reassignment_by_outcome'] = {
        'successful_count': len(successful),
        'successful_mean': (
            float(np.mean(successful)) if successful else None
        ),
        'failed_count': len(failed),
        'failed_mean': float(np.mean(failed)) if failed else None,
    }
    return correlations


def write_current_sweep_outputs(
    records: Sequence[Mapping], output_dir: Path | str
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    csv_path = output_dir / 'current_sweep_targets.csv'
    yaml_path = output_dir / 'current_sweep_summary.yaml'
    markdown_path = output_dir / 'current_sweep_summary.md'
    fields = (
        'frame_id', 'object_id', 'gt_row', 'distance', 'clean_score',
        'clean_iou', 'num_object_points_total', 'num_time0_points',
        'num_history_points', 'num_active_object_points_total',
        'num_active_time0_points', 'current_sweep_point_ratio',
    )
    with csv_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in fields} for record in records)
    summary = summarize_current_sweep(records)
    yaml_path.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )
    lines = [
        '# Current-sweep clean Car target statistics', '',
        f'Targets: {summary["clean_detected_targets"]}', '',
        '| Metric | Mean | Median | P10 | P25 | P75 | P90 | Min | Max |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for field, values in summary['distributions'].items():
        lines.append('| ' + ' | '.join(
            [field] + [
                '' if values[key] is None else f'{values[key]:.6f}'
                for key in ('mean', 'median', 'p10', 'p25', 'p75', 'p90', 'min', 'max')
            ]
        ) + ' |')
    lines.extend(['', '| Threshold | Fraction |', '| --- | ---: |'])
    for threshold in (0, 1, 2, 5):
        lines.append(
            f'| active time=0 points <= {threshold} | '
            f'{summary[f"active_time0_le_{threshold}_fraction"]:.6f} |'
        )
    markdown_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {
        'csv': str(csv_path), 'yaml': str(yaml_path),
        'markdown': str(markdown_path),
    }


def write_correlation_output(
    correlations: Mapping, output_dir: Path | str
) -> str:
    path = Path(output_dir) / 'target_correlations.json'
    path.write_text(
        json.dumps(correlations, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return str(path)
