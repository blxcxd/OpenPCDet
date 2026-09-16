"""Clean-variation-normalized Radar geometry perturbation metrics."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


MEASUREMENT_DIMENSIONS = ('range', 'azimuth', 'elevation')


@dataclass(frozen=True)
class Q95Bin:
    label: str
    range_min_m: float
    range_max_m: float
    include_upper: bool
    matched_pairs: int
    q95: np.ndarray
    joint_a_q95: float | None
    joint_a_q99: float | None


@dataclass(frozen=True)
class MeasurementQ95Reference:
    """Stage 2 clean matched-pair P95 values fixed to the 1 m gate."""

    path: Path
    metadata: Dict
    bins: tuple[Q95Bin, ...]

    @classmethod
    def load(cls, path: Path | str) -> 'MeasurementQ95Reference':
        path = Path(path).expanduser().resolve()
        with path.open(encoding='utf-8') as stream:
            payload = json.load(stream)
        if float(payload.get('gate_m', -1)) != 1.0:
            raise ValueError(
                'measurement Q95 reference must use the Stage 2 1 m gate'
            )
        if float(payload.get('quantile', -1)) != 0.95:
            raise ValueError('measurement reference must contain Q95 values')
        if payload.get('units') != 'm':
            raise ValueError('measurement Q95 reference units must be metres')
        bins = []
        for row in payload.get('bins', []):
            q95 = np.asarray([
                row['q95_range_m'],
                row['q95_azimuth_m'],
                row['q95_elevation_m'],
            ], dtype=np.float64)
            if not np.isfinite(q95).all() or np.any(q95 <= 0):
                raise ValueError('measurement Q95 values must be finite and positive')
            range_min = float(row['range_min_m'])
            range_max = float(row['range_max_m'])
            if not np.isfinite([range_min, range_max]).all() or range_min >= range_max:
                raise ValueError('invalid measurement Q95 range bin')
            joint_a_q95 = row.get('joint_a_q95')
            joint_a_q99 = row.get('joint_a_q99')
            if (joint_a_q95 is None) != (joint_a_q99 is None):
                raise ValueError(
                    'joint A calibration requires both Q95 and Q99'
                )
            if joint_a_q95 is not None:
                joint_a_q95 = float(joint_a_q95)
                joint_a_q99 = float(joint_a_q99)
                if (
                    not np.isfinite([joint_a_q95, joint_a_q99]).all()
                    or joint_a_q95 <= 0
                    or joint_a_q99 < joint_a_q95
                ):
                    raise ValueError('invalid joint A Q95/Q99 calibration')
            bins.append(Q95Bin(
                label=str(row['label']),
                range_min_m=range_min,
                range_max_m=range_max,
                include_upper=bool(row.get('include_upper', False)),
                matched_pairs=int(row['matched_pairs']),
                q95=q95,
                joint_a_q95=joint_a_q95,
                joint_a_q99=joint_a_q99,
            ))
        bins.sort(key=lambda row: row.range_min_m)
        if not bins:
            raise ValueError('measurement Q95 reference contains no bins')
        for previous, current in zip(bins[:-1], bins[1:]):
            if previous.range_max_m != current.range_min_m:
                raise ValueError('measurement Q95 range bins must be contiguous')
        metadata = {key: value for key, value in payload.items() if key != 'bins'}
        metadata['path'] = str(path)
        metadata['bins'] = [
            {
                'label': row.label,
                'range_min_m': row.range_min_m,
                'range_max_m': row.range_max_m,
                'include_upper': row.include_upper,
                'matched_pairs': row.matched_pairs,
                'q95_range_m': float(row.q95[0]),
                'q95_azimuth_m': float(row.q95[1]),
                'q95_elevation_m': float(row.q95[2]),
                'joint_a_q95': row.joint_a_q95,
                'joint_a_q99': row.joint_a_q99,
            }
            for row in bins
        ]
        return cls(path=path, metadata=metadata, bins=tuple(bins))

    @property
    def has_joint_calibration(self) -> bool:
        return all(row.joint_a_q95 is not None for row in self.bins)

    def lookup(self, clean_range_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Q95 triplets, bin indices, and coverage for clean ranges."""
        clean_range_m = np.asarray(clean_range_m, dtype=np.float64)
        q95 = np.full((clean_range_m.size, 3), np.nan, dtype=np.float64)
        bin_indices = np.full(clean_range_m.size, -1, dtype=np.int16)
        for index, row in enumerate(self.bins):
            upper = (
                clean_range_m <= row.range_max_m
                if row.include_upper
                else clean_range_m < row.range_max_m
            )
            selected = (
                (clean_range_m >= row.range_min_m)
                & upper
                & np.isfinite(clean_range_m)
            )
            if np.any(bin_indices[selected] >= 0):
                raise ValueError('measurement Q95 range bins overlap')
            q95[selected] = row.q95
            bin_indices[selected] = index
        covered = bin_indices >= 0
        return q95, bin_indices, covered

    def lookup_joint_a(
        self, bin_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return clean joint-A P95/P99 thresholds for assigned bins."""
        bin_indices = np.asarray(bin_indices)
        q95 = np.full(bin_indices.size, np.nan, dtype=np.float64)
        q99 = np.full(bin_indices.size, np.nan, dtype=np.float64)
        for index, row in enumerate(self.bins):
            if row.joint_a_q95 is None:
                continue
            selected = bin_indices == index
            q95[selected] = row.joint_a_q95
            q99[selected] = row.joint_a_q99
        return q95, q99


def point_measurement_naturalness(
    clean_points: torch.Tensor,
    adversarial_points: torch.Tensor,
    reference: MeasurementQ95Reference,
    change_tolerance_m: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """Compute per-point ``x``, Q95-normalized ``z``, and max anomaly A."""
    if clean_points.shape != adversarial_points.shape:
        raise ValueError('clean and adversarial points must have identical shape')
    if clean_points.ndim != 2 or clean_points.shape[1] < 4:
        raise ValueError('points must contain batch index and XYZ')
    if change_tolerance_m < 0:
        raise ValueError('change_tolerance_m must be non-negative')
    if not torch.equal(clean_points[:, 0], adversarial_points[:, 0]):
        raise ValueError('point order or batch indices changed during attack')

    clean_xyz = clean_points[:, 1:4].detach().double().cpu().numpy()
    adversarial_xyz = adversarial_points[:, 1:4].detach().double().cpu().numpy()
    clean_range = np.linalg.norm(clean_xyz, axis=1)
    adversarial_range = np.linalg.norm(adversarial_xyz, axis=1)
    clean_horizontal = np.linalg.norm(clean_xyz[:, :2], axis=1)
    adversarial_horizontal = np.linalg.norm(adversarial_xyz[:, :2], axis=1)
    clean_azimuth = np.arctan2(clean_xyz[:, 1], clean_xyz[:, 0])
    adversarial_azimuth = np.arctan2(
        adversarial_xyz[:, 1], adversarial_xyz[:, 0]
    )
    clean_elevation = np.arctan2(clean_xyz[:, 2], clean_horizontal)
    adversarial_elevation = np.arctan2(
        adversarial_xyz[:, 2], adversarial_horizontal
    )

    delta_azimuth = np.arctan2(
        np.sin(adversarial_azimuth - clean_azimuth),
        np.cos(adversarial_azimuth - clean_azimuth),
    )
    delta_elevation = adversarial_elevation - clean_elevation
    mean_range = 0.5 * (clean_range + adversarial_range)
    mean_elevation = 0.5 * (clean_elevation + adversarial_elevation)
    x = np.column_stack((
        np.abs(adversarial_range - clean_range),
        mean_range * np.cos(mean_elevation) * np.abs(delta_azimuth),
        mean_range * np.abs(delta_elevation),
    ))
    q95, bin_indices, covered = reference.lookup(clean_range)
    covered &= clean_range > 1e-12
    covered &= np.isfinite(x).all(axis=1)
    z = np.full_like(x, np.nan)
    z[covered] = x[covered] / q95[covered]
    anomaly = np.full(clean_range.size, np.nan, dtype=np.float64)
    anomaly[covered] = np.max(z[covered], axis=1)
    joint_a_q95, joint_a_q99 = reference.lookup_joint_a(bin_indices)
    joint_calibrated = (
        covered & np.isfinite(joint_a_q95) & np.isfinite(joint_a_q99)
    )
    max_dimension = np.full(clean_range.size, -1, dtype=np.int8)
    max_dimension[covered] = np.argmax(z[covered], axis=1).astype(np.int8)
    modified = np.max(x, axis=1) > float(change_tolerance_m)
    batch_indices = clean_points[:, 0].detach().long().cpu().numpy()
    return {
        'batch_index': batch_indices,
        'clean_range_m': clean_range,
        'bin_index': bin_indices,
        'covered': covered,
        'modified': modified,
        'x': x,
        'q95': q95,
        'z': z,
        'anomaly': anomaly,
        'joint_a_q95': joint_a_q95,
        'joint_a_q99': joint_a_q99,
        'joint_calibrated': joint_calibrated,
        'max_dimension': max_dimension,
    }


def _distribution(values: np.ndarray) -> Dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            'count': 0, 'mean': None, 'p50': None, 'p90': None,
            'p95': None, 'p99': None, 'max': None,
        }
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'p50': float(np.quantile(values, 0.50)),
        'p90': float(np.quantile(values, 0.90)),
        'p95': float(np.quantile(values, 0.95)),
        'p99': float(np.quantile(values, 0.99)),
        'max': float(values.max()),
    }


@dataclass
class MeasurementNaturalnessAccumulator:
    reference: MeasurementQ95Reference
    batches: list[Dict[str, np.ndarray]] = field(default_factory=list)

    def update(
        self,
        clean_points: torch.Tensor,
        adversarial_points: torch.Tensor,
        frame_ids: Sequence,
        reference_region_mask: torch.Tensor | None = None,
    ) -> None:
        values = point_measurement_naturalness(
            clean_points, adversarial_points, self.reference
        )
        batch_indices = values['batch_index']
        if batch_indices.size and (
            batch_indices.min() < 0 or batch_indices.max() >= len(frame_ids)
        ):
            raise ValueError('point batch index does not match frame_ids')
        frame_lookup = np.asarray([str(value) for value in frame_ids])
        values['frame_id'] = frame_lookup[batch_indices]
        local_indices = np.empty(batch_indices.size, dtype=np.int64)
        for batch_index in np.unique(batch_indices):
            selected = np.flatnonzero(batch_indices == batch_index)
            local_indices[selected] = np.arange(selected.size)
        values['point_index'] = local_indices
        if reference_region_mask is None:
            values['reference_region'] = np.zeros(
                batch_indices.size, dtype=bool
            )
        else:
            if reference_region_mask.shape != (batch_indices.size,):
                raise ValueError('reference_region_mask must have shape [N]')
            values['reference_region'] = (
                reference_region_mask.detach().bool().cpu().numpy()
            )
        self.batches.append(values)

    def _combined(self) -> Dict[str, np.ndarray]:
        if not self.batches:
            return {}
        return {
            key: np.concatenate([batch[key] for batch in self.batches], axis=0)
            for key in self.batches[0]
        }

    def compute(self) -> Dict:
        values = self._combined()
        if not values:
            return {'reference': self.reference.metadata, 'total_points': 0}
        covered = values['covered']
        modified = values['modified']
        modified_covered = covered & modified

        def subset_summary(mask: np.ndarray) -> Dict:
            z = values['z'][mask]
            anomaly = values['anomaly'][mask]
            dimensions = values['max_dimension'][mask]
            count = int(mask.sum())
            joint_mask = mask & values['joint_calibrated']
            joint_count = int(joint_mask.sum())
            joint_anomaly = values['anomaly'][joint_mask]
            return {
                'count': count,
                'z': {
                    name: _distribution(z[:, index])
                    for index, name in enumerate(MEASUREMENT_DIMENSIONS)
                },
                'A': _distribution(anomaly),
                'A_exceedance_fraction': {
                    'gt_1': float(np.mean(anomaly > 1)) if count else None,
                    'gt_2': float(np.mean(anomaly > 2)) if count else None,
                    'gt_5': float(np.mean(anomaly > 5)) if count else None,
                },
                'joint_calibrated_count': joint_count,
                'joint_exceedance_fraction': {
                    'R_joint95': float(np.mean(
                        joint_anomaly > values['joint_a_q95'][joint_mask]
                    )) if joint_count else None,
                    'R_joint99': float(np.mean(
                        joint_anomaly > values['joint_a_q99'][joint_mask]
                    )) if joint_count else None,
                },
                'dimension_gt_1_fraction': {
                    name: float(np.mean(z[:, index] > 1)) if count else None
                    for index, name in enumerate(MEASUREMENT_DIMENSIONS)
                },
                'max_dimension_fraction': {
                    name: float(np.mean(dimensions == index)) if count else None
                    for index, name in enumerate(MEASUREMENT_DIMENSIONS)
                },
            }

        by_range_bin = {}
        for index, row in enumerate(self.reference.bins):
            in_bin = covered & (values['bin_index'] == index)
            by_range_bin[row.label] = {
                'all_covered': subset_summary(in_bin),
                'modified_covered': subset_summary(in_bin & modified),
            }

        total = int(values['clean_range_m'].size)
        reference_region = values['reference_region'] & covered
        reference_region_modified = reference_region & modified
        return {
            'reference': self.reference.metadata,
            'total_points': total,
            'covered_points': int(covered.sum()),
            'uncovered_points': int((~covered).sum()),
            'coverage_fraction': float(covered.mean()) if total else None,
            'modified_points': int(modified.sum()),
            'modified_covered_points': int(modified_covered.sum()),
            'all_covered': subset_summary(covered),
            'modified_covered': subset_summary(modified_covered),
            'reference_region_covered': subset_summary(reference_region),
            'reference_region_modified_covered': subset_summary(
                reference_region_modified
            ),
            'by_range_bin': by_range_bin,
        }

    def write_csv(self, path: Path | str) -> None:
        """Write all per-point values, retaining the three-dimensional z tuple."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = self._combined()
        fieldnames = [
            'frame_id', 'point_index', 'clean_range_m', 'range_bin',
            'reference_covered', 'modified',
            'x_range_m', 'x_azimuth_m', 'x_elevation_m',
            'q95_range_m', 'q95_azimuth_m', 'q95_elevation_m',
            'z_range', 'z_azimuth', 'z_elevation', 'A', 'max_dimension',
            'joint_A_q95', 'joint_A_q99', 'joint95_exceeded',
            'joint99_exceeded', 'reference_region',
        ]
        with gzip.open(path, 'wt', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            if not values:
                return
            for index in range(values['clean_range_m'].size):
                bin_index = int(values['bin_index'][index])
                covered = bool(values['covered'][index])
                maximum = int(values['max_dimension'][index])
                writer.writerow({
                    'frame_id': values['frame_id'][index],
                    'point_index': int(values['point_index'][index]),
                    'clean_range_m': values['clean_range_m'][index],
                    'range_bin': (
                        self.reference.bins[bin_index].label if covered else ''
                    ),
                    'reference_covered': int(covered),
                    'modified': int(values['modified'][index]),
                    'x_range_m': values['x'][index, 0],
                    'x_azimuth_m': values['x'][index, 1],
                    'x_elevation_m': values['x'][index, 2],
                    'q95_range_m': values['q95'][index, 0] if covered else '',
                    'q95_azimuth_m': (
                        values['q95'][index, 1] if covered else ''
                    ),
                    'q95_elevation_m': (
                        values['q95'][index, 2] if covered else ''
                    ),
                    'z_range': values['z'][index, 0] if covered else '',
                    'z_azimuth': values['z'][index, 1] if covered else '',
                    'z_elevation': values['z'][index, 2] if covered else '',
                    'A': values['anomaly'][index] if covered else '',
                    'max_dimension': (
                        MEASUREMENT_DIMENSIONS[maximum] if covered else ''
                    ),
                    'joint_A_q95': (
                        values['joint_a_q95'][index]
                        if values['joint_calibrated'][index] else ''
                    ),
                    'joint_A_q99': (
                        values['joint_a_q99'][index]
                        if values['joint_calibrated'][index] else ''
                    ),
                    'joint95_exceeded': (
                        int(values['anomaly'][index]
                            > values['joint_a_q95'][index])
                        if values['joint_calibrated'][index] else ''
                    ),
                    'joint99_exceeded': (
                        int(values['anomaly'][index]
                            > values['joint_a_q99'][index])
                        if values['joint_calibrated'][index] else ''
                    ),
                    'reference_region': int(
                        values['reference_region'][index]
                    ),
                })
