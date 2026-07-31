"""Streaming statistics for model-input 4D-radar point features."""

from typing import Dict, Sequence

import numpy as np
import torch


DEFAULT_QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
MAX_REPORTED_UNIQUE_VALUES = 32


def _as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _format_number(value) -> str:
    if value is None:
        return 'n/a'
    value = float(value)
    if value == 0:
        return '0'
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f'{value:.6e}'
    return f'{value:.6f}'


def extract_voxelized_features(voxels, voxel_num_points) -> np.ndarray:
    """Flatten valid hard-voxel entries while excluding zero padding."""
    voxels = _as_numpy(voxels)
    voxel_num_points = np.asarray(voxel_num_points)
    if voxels.ndim != 3:
        raise ValueError(f'voxels must have shape [M, T, F], got {voxels.shape}')
    if (
        voxel_num_points.ndim != 1
        or voxel_num_points.shape[0] != voxels.shape[0]
    ):
        raise ValueError(
            'voxel_num_points must have shape [M] matching voxels'
        )
    if np.any(voxel_num_points < 0) or np.any(
        voxel_num_points > voxels.shape[1]
    ):
        raise ValueError('voxel_num_points contains an invalid point count')

    valid = (
        np.arange(voxels.shape[1])[None, :]
        < voxel_num_points.astype(np.int64, copy=False)[:, None]
    )
    return voxels[valid]


class StreamingFeatureStatistics:
    """Compute exact moments and sampled quantiles without retaining all points.

    Quantile rows are selected with independent random priorities. Keeping the
    rows with the smallest priorities is a uniform sample without replacement
    over every point seen so far.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        max_quantile_points: int = 1_000_000,
        seed: int = 1024,
    ):
        self.feature_names = list(feature_names)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.max_quantile_points = int(max_quantile_points)
        self.seed = int(seed)

        if not self.feature_names:
            raise ValueError('feature_names must not be empty')
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('feature_names must be unique')
        if not self.quantiles or any(
            value < 0 or value > 1 for value in self.quantiles
        ):
            raise ValueError('quantiles must be values in [0, 1]')
        if len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError('quantiles must be unique')
        if self.max_quantile_points <= 0:
            raise ValueError('max_quantile_points must be positive')
        if self.seed < 0:
            raise ValueError('seed must be non-negative')

        feature_count = len(self.feature_names)
        self.total_points = 0
        self.finite_counts = np.zeros(feature_count, dtype=np.int64)
        self.means = np.zeros(feature_count, dtype=np.float64)
        self.m2 = np.zeros(feature_count, dtype=np.float64)
        self.minimums = np.full(feature_count, np.inf, dtype=np.float64)
        self.maximums = np.full(feature_count, -np.inf, dtype=np.float64)
        self._unique_values = [set() for _ in range(feature_count)]
        self._sample = np.empty((0, feature_count), dtype=np.float64)
        self._priorities = np.empty(0, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    def update(self, values) -> None:
        values = _as_numpy(values)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f'values must have shape [N, {len(self.feature_names)}], '
                f'got {values.shape}'
            )
        if values.shape[0] == 0:
            return

        self.total_points += int(values.shape[0])
        for column in range(values.shape[1]):
            finite_values = values[np.isfinite(values[:, column]), column]
            batch_count = int(finite_values.size)
            if batch_count == 0:
                continue

            batch_mean = float(finite_values.mean(dtype=np.float64))
            centered = finite_values - batch_mean
            batch_m2 = float(np.dot(centered, centered))
            previous_count = int(self.finite_counts[column])
            combined_count = previous_count + batch_count
            delta = batch_mean - self.means[column]

            self.means[column] += delta * batch_count / combined_count
            self.m2[column] += (
                batch_m2
                + delta * delta * previous_count * batch_count / combined_count
            )
            self.finite_counts[column] = combined_count
            self.minimums[column] = min(
                self.minimums[column], float(finite_values.min())
            )
            self.maximums[column] = max(
                self.maximums[column], float(finite_values.max())
            )
            if self._unique_values[column] is not None:
                self._unique_values[column].update(
                    float(value) for value in np.unique(finite_values)
                )
                if (
                    len(self._unique_values[column])
                    > MAX_REPORTED_UNIQUE_VALUES
                ):
                    self._unique_values[column] = None

        priorities = self._rng.random(values.shape[0])
        combined_values = np.concatenate((self._sample, values), axis=0)
        combined_priorities = np.concatenate(
            (self._priorities, priorities), axis=0
        )
        if combined_values.shape[0] > self.max_quantile_points:
            keep = np.argpartition(
                combined_priorities,
                self.max_quantile_points - 1,
            )[:self.max_quantile_points]
            combined_values = combined_values[keep]
            combined_priorities = combined_priorities[keep]
        self._sample = combined_values
        self._priorities = combined_priorities

    def compute(self) -> Dict:
        features = {}
        for column, feature_name in enumerate(self.feature_names):
            finite_count = int(self.finite_counts[column])
            sample_values = self._sample[:, column]
            sample_values = sample_values[np.isfinite(sample_values)]
            if finite_count:
                variance = max(self.m2[column] / finite_count, 0.0)
                quantile_values = np.quantile(
                    sample_values,
                    self.quantiles,
                )
                quantiles = {
                    str(value): float(result)
                    for value, result in zip(
                        self.quantiles,
                        np.atleast_1d(quantile_values),
                    )
                }
                q01 = float(np.quantile(sample_values, 0.01))
                q25 = float(np.quantile(sample_values, 0.25))
                q75 = float(np.quantile(sample_values, 0.75))
                q99 = float(np.quantile(sample_values, 0.99))
                feature_result = {
                    'finite_count': finite_count,
                    'nonfinite_count': self.total_points - finite_count,
                    'min': float(self.minimums[column]),
                    'max': float(self.maximums[column]),
                    'mean': float(self.means[column]),
                    'std': float(np.sqrt(variance)),
                    'quantiles': quantiles,
                    'iqr': q75 - q25,
                    'robust_range_q01_q99': q99 - q01,
                    'unique_count': (
                        len(self._unique_values[column])
                        if self._unique_values[column] is not None
                        else None
                    ),
                    'unique_values': (
                        sorted(self._unique_values[column])
                        if self._unique_values[column] is not None
                        else None
                    ),
                    'unique_values_truncated': (
                        self._unique_values[column] is None
                    ),
                }
            else:
                feature_result = {
                    'finite_count': 0,
                    'nonfinite_count': self.total_points,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'quantiles': {
                        str(value): None for value in self.quantiles
                    },
                    'iqr': None,
                    'robust_range_q01_q99': None,
                    'unique_count': 0,
                    'unique_values': [],
                    'unique_values_truncated': False,
                }
            features[feature_name] = feature_result

        return {
            'total_points': self.total_points,
            'quantile_sample_points': int(self._sample.shape[0]),
            'quantile_sampling': (
                'exact'
                if self.total_points <= self.max_quantile_points
                else 'uniform_without_replacement'
            ),
            'max_quantile_points': self.max_quantile_points,
            'seed': self.seed,
            'quantile_probabilities': list(self.quantiles),
            'features': features,
        }


def format_statistics_table(statistics: Dict) -> str:
    """Format the most useful feature scales as a plain-text table."""
    headers = (
        'feature',
        'count',
        'min',
        'q01',
        'mean',
        'std',
        'q99',
        'max',
        'IQR',
        'q99-q01',
        'unique',
    )
    rows = []
    for name, values in statistics['features'].items():
        quantiles = values['quantiles']
        rows.append(
            (
                name,
                str(values['finite_count']),
                _format_number(values['min']),
                _format_number(quantiles.get('0.01')),
                _format_number(values['mean']),
                _format_number(values['std']),
                _format_number(quantiles.get('0.99')),
                _format_number(values['max']),
                _format_number(values['iqr']),
                _format_number(values['robust_range_q01_q99']),
                (
                    f'>{MAX_REPORTED_UNIQUE_VALUES}'
                    if values['unique_values_truncated']
                    else str(values['unique_count'])
                ),
            )
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def render(row):
        return '  '.join(
            value.ljust(widths[column])
            for column, value in enumerate(row)
        )

    separator = '  '.join('-' * width for width in widths)
    return '\n'.join((render(headers), separator, *(render(row) for row in rows)))
