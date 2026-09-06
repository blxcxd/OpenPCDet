"""Temporal metadata recovery for accumulated VoD Radar point clouds.

VoD stores historical sweeps in the coordinate system of the reference sweep.
This module recovers the original single-sweep geometry, the exact rigid
transform used by the released accumulation, and object-track assignments.
It deliberately does not implement an adversarial optimizer; the recovered
context is the data layer needed by a later multi-sweep measurement attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from pcdet.utils.calibration_kitti import Calibration


TEMPORAL_CACHE_VERSION = 1
DEFAULT_TARGET_CLASSES = ('Car', 'Pedestrian', 'Cyclist')


def _frame_name(frame_id: str | int) -> str:
    value = str(frame_id)
    if not value.isdigit():
        raise ValueError(f'VoD frame id must be numeric, got {frame_id!r}')
    return value.zfill(5)


def load_radar_points(path: Path) -> np.ndarray:
    """Load one VoD Radar file as an ``[N, 7]`` float32 array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 7:
        raise ValueError(f'{path} does not contain an N x 7 Radar array')
    return values.reshape(-1, 7)


def apply_rigid_transform(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to row-major XYZ points."""
    xyz = np.asarray(xyz)
    transform = np.asarray(transform)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must have shape [N, 3]')
    if transform.shape != (4, 4):
        raise ValueError('transform must have shape [4, 4]')
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def fit_rigid_transform(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the proper rigid transform ``target = T(source)`` with Kabsch."""
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2:
        raise ValueError('source_xyz and target_xyz must have equal [N, 3] shapes')
    if source_xyz.shape[1] != 3 or source_xyz.shape[0] < 3:
        raise ValueError('at least three 3D correspondences are required')
    if not np.isfinite(source_xyz).all() or not np.isfinite(target_xyz).all():
        raise ValueError('rigid-transform correspondences must be finite')

    source_center = source_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)
    centered_source = source_xyz - source_center
    centered_target = target_xyz - target_center
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        centered_source.T @ centered_target
    )
    rotation = right_vectors.T @ left_vectors.T
    if np.linalg.det(rotation) < 0:
        right_vectors[-1] *= -1
        rotation = right_vectors.T @ left_vectors.T
    if singular_values[1] <= np.finfo(np.float64).eps * singular_values[0]:
        raise ValueError('Radar correspondences are collinear or degenerate')

    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    residuals = np.linalg.norm(
        apply_rigid_transform(source_xyz, transform) - target_xyz,
        axis=1,
    )
    return transform, residuals


def _cartesian_to_measurement(xyz: np.ndarray) -> np.ndarray:
    horizontal = np.linalg.norm(xyz[:, :2], axis=1)
    distance = np.linalg.norm(xyz, axis=1)
    return np.column_stack((
        distance,
        np.arctan2(xyz[:, 1], xyz[:, 0]),
        np.arctan2(xyz[:, 2], horizontal),
    ))


def _measurement_to_cartesian(measurement: np.ndarray) -> np.ndarray:
    distance, azimuth, elevation = measurement.T
    horizontal = distance * np.cos(elevation)
    return np.column_stack((
        horizontal * np.cos(azimuth),
        horizontal * np.sin(azimuth),
        distance * np.sin(elevation),
    ))


@dataclass(frozen=True)
class SweepTransform:
    """Recovered metadata for one source sweep in an accumulated frame."""

    sweep_id: int
    source_frame_id: str
    source_to_reference: np.ndarray
    point_count: int
    residual_mean_m: float
    residual_max_m: float

    @property
    def reference_to_source(self) -> np.ndarray:
        return np.linalg.inv(self.source_to_reference)


@dataclass(frozen=True)
class TrackedBox:
    """One tracking-labelled object box in Radar coordinates."""

    frame_id: str
    track_id: int
    class_name: str
    class_id: int
    box: np.ndarray
    gt_row: int = -1


@dataclass(frozen=True)
class TemporalTrackAssignments:
    """Track and class identity assigned to each accumulated point."""

    track_ids: np.ndarray
    class_ids: np.ndarray
    current_targets: Tuple[TrackedBox, ...]
    diagnostics: Mapping[str, object]

    @property
    def target_mask(self) -> np.ndarray:
        return self.track_ids >= 0


@dataclass(frozen=True)
class TemporalBatchData:
    """CPU temporal metadata aligned to one OpenPCDet point batch."""

    source_xyz: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    group_ids: np.ndarray
    track_ids: np.ndarray
    gt_rows: np.ndarray
    sweep_ids: np.ndarray
    diagnostics: Mapping[str, float]

    @property
    def target_mask(self) -> np.ndarray:
        return self.group_ids >= 0


@dataclass
class TemporalFrameContext:
    """Raw point correspondence and transforms for one accumulated frame."""

    frame_id: str
    reference_points: np.ndarray
    source_points_by_sweep: Dict[int, np.ndarray]
    sweeps: Tuple[SweepTransform, ...]
    cache_hit: bool = False

    @property
    def sweep_map(self) -> Dict[int, SweepTransform]:
        return {record.sweep_id: record for record in self.sweeps}

    def aligned_source_xyz(self) -> np.ndarray:
        """Return original single-sweep XYZ aligned to reference point rows."""
        source_xyz = np.empty_like(self.reference_points[:, :3])
        for record in self.sweeps:
            rows = self.reference_points[:, 6] == float(record.sweep_id)
            source_xyz[rows] = self.source_points_by_sweep[
                record.sweep_id
            ][:, :3]
        return source_xyz

    def source_to_reference_xyz(self, source_xyz: np.ndarray) -> np.ndarray:
        """Transform row-aligned source XYZ into the reference Radar frame."""
        source_xyz = np.asarray(source_xyz)
        if source_xyz.shape != self.reference_points[:, :3].shape:
            raise ValueError('source_xyz must align with all reference point rows')
        reconstructed = np.empty_like(source_xyz)
        for record in self.sweeps:
            rows = self.reference_points[:, 6] == float(record.sweep_id)
            reconstructed[rows] = apply_rigid_transform(
                source_xyz[rows], record.source_to_reference
            )
        return reconstructed

    def reference_to_source_xyz(self, reference_xyz: np.ndarray) -> np.ndarray:
        """Undo accumulation for row-aligned reference-frame XYZ."""
        reference_xyz = np.asarray(reference_xyz)
        if reference_xyz.shape != self.reference_points[:, :3].shape:
            raise ValueError('reference_xyz must align with all point rows')
        restored = np.empty_like(reference_xyz)
        for record in self.sweeps:
            rows = self.reference_points[:, 6] == float(record.sweep_id)
            restored[rows] = apply_rigid_transform(
                reference_xyz[rows], record.reference_to_source
            )
        return restored

    def reconstruct_points(
        self,
        source_xyz: np.ndarray,
        modification_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compose reference points while preserving every non-XYZ feature.

        When ``modification_mask`` is all false, the returned point cloud is
        bit-identical to the released accumulated cloud. This is the intended
        zero-perturbation path for the later adversarial optimizer.
        """
        reconstructed_xyz = self.source_to_reference_xyz(source_xyz)
        if modification_mask is None:
            modification_mask = np.ones(
                len(self.reference_points), dtype=np.bool_
            )
        modification_mask = np.asarray(modification_mask, dtype=np.bool_)
        if modification_mask.shape != (len(self.reference_points),):
            raise ValueError('modification_mask must have shape [N]')
        output = self.reference_points.copy()
        output[modification_mask, :3] = reconstructed_xyz[modification_mask]
        return output

    def zero_delta_diagnostics(self) -> Dict[str, object]:
        """Measure source recovery and measurement round-trip fidelity."""
        clean_source = self.aligned_source_xyz().astype(np.float64)
        inverse_source = self.reference_to_source_xyz(
            self.reference_points[:, :3].astype(np.float64)
        )
        source_error = np.linalg.norm(inverse_source - clean_source, axis=1)
        measurement_xyz = _measurement_to_cartesian(
            _cartesian_to_measurement(clean_source)
        )
        reconstructed = self.source_to_reference_xyz(measurement_xyz)
        reference_error = np.linalg.norm(
            reconstructed - self.reference_points[:, :3], axis=1
        )
        exact_zero = self.reconstruct_points(
            clean_source,
            modification_mask=np.zeros(len(clean_source), dtype=np.bool_),
        )
        return {
            'source_recovery_mean_m': float(source_error.mean()),
            'source_recovery_max_m': float(source_error.max()),
            'measurement_roundtrip_mean_m': float(reference_error.mean()),
            'measurement_roundtrip_max_m': float(reference_error.max()),
            'zero_mask_exact': bool(
                np.array_equal(exact_zero, self.reference_points)
            ),
            'non_xyz_change_count': int(np.count_nonzero(
                exact_zero[:, 3:] != self.reference_points[:, 3:]
            )),
        }

    def align_reference_subset(self, subset: np.ndarray) -> np.ndarray:
        """Map an exact filtered subset back to raw accumulated row indices."""
        subset = np.asarray(subset)
        if subset.ndim != 2 or subset.shape[1] != 7:
            raise ValueError('subset must have shape [M, 7]')
        row_queues: Dict[bytes, list[int]] = {}
        for index, row in enumerate(np.ascontiguousarray(self.reference_points)):
            row_queues.setdefault(row.tobytes(), []).append(index)
        offsets: Dict[bytes, int] = {}
        indices = []
        for row in np.ascontiguousarray(subset):
            key = row.tobytes()
            offset = offsets.get(key, 0)
            candidates = row_queues.get(key, ())
            if offset >= len(candidates):
                raise ValueError('subset contains a point absent from reference')
            indices.append(candidates[offset])
            offsets[key] = offset + 1
        return np.asarray(indices, dtype=np.int64)


class TemporalSweepResolver:
    """Recover and cache exact VoD single-to-accumulated sweep transforms."""

    def __init__(
        self,
        dataset_root: Path | str,
        cache_dir: Optional[Path | str] = None,
        max_residual_m: float = 2e-5,
    ):
        self.dataset_root = Path(dataset_root)
        self.accumulated_dir = (
            self.dataset_root / 'radar_5frames/training/velodyne'
        )
        self.single_dir = self.dataset_root / 'radar/training/velodyne'
        self.label_dir = self.dataset_root / 'lidar/training/label_2'
        self.calibration_dir = self.dataset_root / 'radar/training/calib'
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_residual_m = float(max_residual_m)
        if self.max_residual_m <= 0:
            raise ValueError('max_residual_m must be positive')
        for path in (self.accumulated_dir, self.single_dir):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, frame_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f'{frame_id}.npz'

    def _load_cached_transforms(
        self,
        frame_id: str,
        sweep_ids: np.ndarray,
        source_frame_ids: np.ndarray,
        point_counts: np.ndarray,
    ) -> Optional[np.ndarray]:
        cache_path = self._cache_path(frame_id)
        if cache_path is None or not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                if int(cache['version']) != TEMPORAL_CACHE_VERSION:
                    return None
                if str(cache['frame_id']) != frame_id:
                    return None
                if not np.array_equal(cache['sweep_ids'], sweep_ids):
                    return None
                if not np.array_equal(
                    cache['source_frame_ids'], source_frame_ids
                ):
                    return None
                if not np.array_equal(cache['point_counts'], point_counts):
                    return None
                transforms = np.asarray(cache['source_to_reference'])
        except (OSError, KeyError, ValueError):
            return None
        expected = (len(sweep_ids), 4, 4)
        if transforms.shape != expected or not np.isfinite(transforms).all():
            return None
        return transforms.astype(np.float64, copy=False)

    def _write_cache(
        self,
        frame_id: str,
        sweep_ids: np.ndarray,
        source_frame_ids: np.ndarray,
        point_counts: np.ndarray,
        transforms: np.ndarray,
    ) -> None:
        cache_path = self._cache_path(frame_id)
        if cache_path is None:
            return
        temporary = cache_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(
                stream,
                version=np.asarray(TEMPORAL_CACHE_VERSION, dtype=np.int64),
                frame_id=np.asarray(frame_id),
                sweep_ids=sweep_ids,
                source_frame_ids=source_frame_ids,
                point_counts=point_counts,
                source_to_reference=transforms,
            )
        temporary.replace(cache_path)

    def resolve(self, frame_id: str | int) -> TemporalFrameContext:
        frame_id = _frame_name(frame_id)
        reference = load_radar_points(
            self.accumulated_dir / f'{frame_id}.bin'
        )
        raw_sweep_ids = np.unique(reference[:, 6])
        rounded = np.rint(raw_sweep_ids).astype(np.int64)
        if not np.allclose(raw_sweep_ids, rounded, atol=1e-6, rtol=0):
            raise ValueError(f'{frame_id} contains non-integral sweep ids')
        if np.any(rounded > 0):
            raise ValueError(f'{frame_id} contains future sweep ids')
        sweep_ids = np.sort(rounded)
        source_ids = np.asarray(
            [int(frame_id) + int(value) for value in sweep_ids],
            dtype=np.int64,
        )
        if np.any(source_ids < 0):
            raise ValueError(f'{frame_id} crosses the beginning of a scene')
        point_counts = np.asarray([
            int(np.count_nonzero(reference[:, 6] == float(value)))
            for value in sweep_ids
        ], dtype=np.int64)
        cached = self._load_cached_transforms(
            frame_id, sweep_ids, source_ids, point_counts
        )
        cache_hit = cached is not None

        sources: Dict[int, np.ndarray] = {}
        transforms = []
        records = []
        for index, sweep_id in enumerate(sweep_ids.tolist()):
            source_frame_id = _frame_name(source_ids[index])
            source = load_radar_points(
                self.single_dir / f'{source_frame_id}.bin'
            )
            reference_rows = reference[:, 6] == float(sweep_id)
            target = reference[reference_rows]
            if len(source) != len(target):
                raise ValueError(
                    f'{frame_id} sweep {sweep_id}: source/reference point '
                    f'count mismatch ({len(source)} != {len(target)})'
                )
            if not np.array_equal(source[:, 3:6], target[:, 3:6]):
                raise ValueError(
                    f'{frame_id} sweep {sweep_id}: RCS/Doppler values or '
                    'point order differ from the single-sweep file'
                )
            if cached is None:
                transform, residuals = fit_rigid_transform(
                    source[:, :3], target[:, :3]
                )
            else:
                transform = cached[index]
                residuals = np.linalg.norm(
                    apply_rigid_transform(source[:, :3], transform)
                    - target[:, :3],
                    axis=1,
                )
            maximum = float(residuals.max())
            if maximum > self.max_residual_m:
                raise ValueError(
                    f'{frame_id} sweep {sweep_id}: rigid reconstruction '
                    f'error {maximum:.9g} m exceeds '
                    f'{self.max_residual_m:.9g} m'
                )
            sources[sweep_id] = source
            transforms.append(transform)
            records.append(SweepTransform(
                sweep_id=sweep_id,
                source_frame_id=source_frame_id,
                source_to_reference=transform,
                point_count=len(source),
                residual_mean_m=float(residuals.mean()),
                residual_max_m=maximum,
            ))

        if not cache_hit:
            self._write_cache(
                frame_id,
                sweep_ids,
                source_ids,
                point_counts,
                np.stack(transforms),
            )
        return TemporalFrameContext(
            frame_id=frame_id,
            reference_points=reference,
            source_points_by_sweep=sources,
            sweeps=tuple(records),
            cache_hit=cache_hit,
        )

    def load_tracked_boxes(
        self,
        frame_id: str | int,
        target_classes: Optional[Iterable[str]] = None,
        dataset_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
    ) -> Tuple[TrackedBox, ...]:
        """Load tracking-labelled boxes and convert them to Radar coordinates."""
        frame_id = _frame_name(frame_id)
        label_path = self.label_dir / f'{frame_id}.txt'
        calibration_path = self.calibration_dir / f'{frame_id}.txt'
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(calibration_path)
        selected = (
            set(DEFAULT_TARGET_CLASSES)
            if target_classes is None
            else set(target_classes)
        )
        dataset_classes = tuple(dataset_classes)
        dataset_class_set = set(dataset_classes)
        class_ids = {
            class_name: index + 1
            for index, class_name in enumerate(dataset_classes)
        }
        calibration = Calibration(calibration_path)
        parsed = []
        gt_row = -1
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] in dataset_class_set:
                gt_row += 1
            if fields[0] not in selected:
                continue
            if len(fields) < 15:
                raise ValueError(f'invalid KITTI label in {label_path}')
            raw_track_id = float(fields[1])
            track_id = int(round(raw_track_id))
            if not np.isclose(raw_track_id, track_id, atol=1e-6):
                raise ValueError(
                    f'{label_path} does not contain integral tracking IDs'
                )
            height, width, length = map(float, fields[8:11])
            location_camera = np.asarray(
                [[float(value) for value in fields[11:14]]],
                dtype=np.float32,
            )
            location_radar = calibration.rect_to_lidar(location_camera)[0]
            location_radar[2] += height / 2.0
            rotation_y = float(fields[14])
            box = np.asarray([
                location_radar[0], location_radar[1], location_radar[2],
                length, width, height, -(rotation_y + np.pi / 2.0),
            ], dtype=np.float32)
            parsed.append(TrackedBox(
                frame_id=frame_id,
                track_id=track_id,
                class_name=fields[0],
                class_id=class_ids.get(fields[0], -1),
                box=box,
                gt_row=gt_row,
            ))
        track_ids = [record.track_id for record in parsed]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError(
                f'{label_path} has duplicate tracking IDs; tracking labels '
                'may not be installed'
            )
        return tuple(parsed)

    def assign_tracks(
        self,
        context: TemporalFrameContext,
        target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
        dataset_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
        box_margin: float = 0.0,
    ) -> TemporalTrackAssignments:
        """Assign every source-sweep point to the current targets' tracks."""
        if box_margin < 0:
            raise ValueError('box_margin must be non-negative')
        current_targets = self.load_tracked_boxes(
            context.frame_id, target_classes, dataset_classes
        )
        current_by_track = {
            record.track_id: record for record in current_targets
        }
        track_ids = np.full(
            len(context.reference_points), -1, dtype=np.int64
        )
        class_ids = np.full(
            len(context.reference_points), -1, dtype=np.int64
        )
        sweep_diagnostics = []
        for record in context.sweeps:
            source_boxes = tuple(
                box for box in self.load_tracked_boxes(
                    record.source_frame_id,
                    target_classes,
                    dataset_classes,
                )
                if box.track_id in current_by_track
                and box.class_name
                == current_by_track[box.track_id].class_name
            )
            source_points = context.source_points_by_sweep[record.sweep_id]
            source_track_ids, source_class_ids = assign_points_to_tracked_boxes(
                source_points[:, :3], source_boxes, margin=box_margin
            )
            rows = context.reference_points[:, 6] == float(record.sweep_id)
            track_ids[rows] = source_track_ids
            class_ids[rows] = source_class_ids
            sweep_diagnostics.append({
                'sweep_id': record.sweep_id,
                'source_frame_id': record.source_frame_id,
                'point_count': record.point_count,
                'matched_target_count': len(source_boxes),
                'assigned_target_point_count': int(
                    np.count_nonzero(source_track_ids >= 0)
                ),
            })
        historical = [
            item for item in sweep_diagnostics if item['sweep_id'] < 0
        ]
        return TemporalTrackAssignments(
            track_ids=track_ids,
            class_ids=class_ids,
            current_targets=current_targets,
            diagnostics={
                'frame_id': context.frame_id,
                'current_target_count': len(current_targets),
                'sweep_count': len(context.sweeps),
                'historical_sweep_count': len(historical),
                'assigned_target_point_count': int(
                    np.count_nonzero(track_ids >= 0)
                ),
                'historical_assigned_target_point_count': int(sum(
                    item['assigned_target_point_count'] for item in historical
                )),
                'sweeps': sweep_diagnostics,
            },
        )


def assign_points_to_tracked_boxes(
    points_xyz: np.ndarray,
    boxes: Sequence[TrackedBox],
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign points to boxes using normalized object-local center distance."""
    points_xyz = np.asarray(points_xyz)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError('points_xyz must have shape [N, 3]')
    if margin < 0:
        raise ValueError('margin must be non-negative')
    assigned_tracks = np.full(len(points_xyz), -1, dtype=np.int64)
    assigned_classes = np.full(len(points_xyz), -1, dtype=np.int64)
    if not boxes or not len(points_xyz):
        return assigned_tracks, assigned_classes
    geometry = np.stack([record.box for record in boxes]).astype(np.float64)
    relative = points_xyz[:, None, :] - geometry[None, :, :3]
    cosine = np.cos(geometry[:, 6])
    sine = np.sin(geometry[:, 6])
    local_x = relative[..., 0] * cosine + relative[..., 1] * sine
    local_y = -relative[..., 0] * sine + relative[..., 1] * cosine
    local_z = relative[..., 2]
    half_size = geometry[:, 3:6] * 0.5 + float(margin)
    if np.any(half_size <= 0):
        raise ValueError('tracked boxes must have positive dimensions')
    local = np.stack((local_x, local_y, local_z), axis=-1)
    inside = np.all(np.abs(local) <= half_size[None, :, :], axis=-1)
    distance = np.sum(
        np.square(local / half_size[None, :, :]), axis=-1
    )
    distance[~inside] = np.inf
    best = distance.argmin(axis=1)
    valid = np.isfinite(distance[np.arange(len(points_xyz)), best])
    track_values = np.asarray([record.track_id for record in boxes])
    class_values = np.asarray([record.class_id for record in boxes])
    assigned_tracks[valid] = track_values[best[valid]]
    assigned_classes[valid] = class_values[best[valid]]
    return assigned_tracks, assigned_classes


def _angle_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = first - second
    return np.abs(np.arctan2(np.sin(difference), np.cos(difference)))


def match_tracks_to_gt_rows(
    tracked_boxes: Sequence[TrackedBox],
    gt_boxes: np.ndarray,
    tolerance: float = 2e-3,
) -> Dict[int, int]:
    """Match label tracks to processed OpenPCDet GT rows by box geometry."""
    gt_boxes = np.asarray(gt_boxes)
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 8:
        raise ValueError('gt_boxes must have shape [M, >=8]')
    valid_rows = np.flatnonzero(np.all(gt_boxes[:, 3:6] > 0, axis=1))
    available = set(valid_rows.tolist())
    matched: Dict[int, int] = {}
    for tracked in tracked_boxes:
        candidates = np.asarray([
            row for row in available
            if int(round(float(gt_boxes[row, -1]))) == tracked.class_id
        ], dtype=np.int64)
        if not len(candidates):
            continue
        geometry = gt_boxes[candidates, :7]
        center_error = np.linalg.norm(
            geometry[:, :3] - tracked.box[None, :3], axis=1
        )
        size_error = np.max(
            np.abs(geometry[:, 3:6] - tracked.box[None, 3:6]), axis=1
        )
        heading_error = _angle_distance(
            geometry[:, 6], np.full(len(candidates), tracked.box[6])
        )
        cost = center_error + size_error + heading_error
        best_local = int(np.argmin(cost))
        best_row = int(candidates[best_local])
        if (
            center_error[best_local] <= tolerance
            and size_error[best_local] <= tolerance
            and heading_error[best_local] <= tolerance
        ):
            matched[tracked.track_id] = best_row
            available.remove(best_row)
    return matched


def build_temporal_batch_data(
    resolver: TemporalSweepResolver,
    batched_points: np.ndarray,
    frame_ids: Sequence[str],
    gt_boxes: np.ndarray,
    target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
    dataset_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
    allowed_gt_rows: Optional[Mapping[int, Sequence[int]]] = None,
    box_margin: float = 0.0,
) -> TemporalBatchData:
    """Align temporal metadata to an OpenPCDet ``[batch, 7 features]`` batch."""
    batched_points = np.asarray(batched_points)
    gt_boxes = np.asarray(gt_boxes)
    if batched_points.ndim != 2 or batched_points.shape[1] != 8:
        raise ValueError('batched_points must have shape [N, 8]')
    if gt_boxes.ndim != 3 or gt_boxes.shape[0] != len(frame_ids):
        raise ValueError('gt_boxes must have shape [batch_size, M, >=8]')
    batch_indices = np.rint(batched_points[:, 0]).astype(np.int64)
    if not np.allclose(batched_points[:, 0], batch_indices, atol=1e-6):
        raise ValueError('point batch indices must be integral')
    if np.any(batch_indices < 0) or np.any(batch_indices >= len(frame_ids)):
        raise ValueError('point batch index is out of range')

    point_count = len(batched_points)
    source_xyz = np.empty((point_count, 3), dtype=np.float64)
    rotations = np.empty((point_count, 3, 3), dtype=np.float64)
    translations = np.empty((point_count, 3), dtype=np.float64)
    group_ids = np.full(point_count, -1, dtype=np.int64)
    track_ids = np.full(point_count, -1, dtype=np.int64)
    point_gt_rows = np.full(point_count, -1, dtype=np.int64)
    sweep_ids = np.empty(point_count, dtype=np.int64)
    next_group = 0
    maximum_alignment_error = 0.0
    total_current_targets = 0
    total_matched_targets = 0
    total_temporal_target_points = 0

    for batch_index, raw_frame_id in enumerate(frame_ids):
        frame_id = _frame_name(raw_frame_id)
        batch_rows = np.flatnonzero(batch_indices == batch_index)
        context = resolver.resolve(frame_id)
        subset = np.ascontiguousarray(batched_points[batch_rows, 1:])
        raw_rows = context.align_reference_subset(subset)
        aligned_source = context.aligned_source_xyz().astype(np.float64)
        source_xyz[batch_rows] = aligned_source[raw_rows]
        raw_sweeps = np.rint(
            context.reference_points[raw_rows, 6]
        ).astype(np.int64)
        sweep_ids[batch_rows] = raw_sweeps
        for sweep in context.sweeps:
            local = raw_sweeps == sweep.sweep_id
            rotations[batch_rows[local]] = sweep.source_to_reference[:3, :3]
            translations[batch_rows[local]] = sweep.source_to_reference[:3, 3]

        assignments = resolver.assign_tracks(
            context,
            target_classes=target_classes,
            dataset_classes=dataset_classes,
            box_margin=box_margin,
        )
        track_to_gt = match_tracks_to_gt_rows(
            assignments.current_targets, gt_boxes[batch_index]
        )
        target_class_ids = {
            list(dataset_classes).index(class_name) + 1
            for class_name in target_classes
        }
        valid_gt = np.all(gt_boxes[batch_index, :, 3:6] > 0, axis=1)
        valid_gt &= np.isin(
            np.rint(gt_boxes[batch_index, :, -1]).astype(np.int64),
            list(target_class_ids),
        )
        expected_matches = int(np.count_nonzero(valid_gt))
        if len(track_to_gt) != expected_matches:
            raise RuntimeError(
                f'{frame_id}: matched {len(track_to_gt)} tracking boxes to '
                f'{expected_matches} processed target GT rows'
            )
        total_current_targets += len(assignments.current_targets)
        total_matched_targets += len(track_to_gt)
        if allowed_gt_rows is not None:
            allowed = {
                int(value)
                for value in allowed_gt_rows.get(batch_index, ())
            }
            track_to_gt = {
                track: row for track, row in track_to_gt.items()
                if row in allowed
            }
        raw_tracks = assignments.track_ids[raw_rows]
        selected_tracks = set(track_to_gt)
        for track_id in sorted(selected_tracks):
            local = raw_tracks == track_id
            if not local.any():
                continue
            selected_rows = batch_rows[local]
            group_ids[selected_rows] = next_group
            track_ids[selected_rows] = track_id
            point_gt_rows[selected_rows] = track_to_gt[track_id]
            total_temporal_target_points += int(local.sum())
            next_group += 1

        reconstructed = np.einsum(
            'nij,nj->ni', rotations[batch_rows], source_xyz[batch_rows]
        ) + translations[batch_rows]
        alignment_error = np.linalg.norm(
            reconstructed - batched_points[batch_rows, 1:4], axis=1
        )
        if len(alignment_error):
            maximum_alignment_error = max(
                maximum_alignment_error, float(alignment_error.max())
            )
        if len(alignment_error) and (
            float(alignment_error.max()) > resolver.max_residual_m
        ):
            raise RuntimeError(
                f'{frame_id}: model-input temporal alignment error exceeds '
                f'{resolver.max_residual_m:g} m'
            )

    target = group_ids >= 0
    return TemporalBatchData(
        source_xyz=source_xyz,
        rotations=rotations,
        translations=translations,
        group_ids=group_ids,
        track_ids=track_ids,
        gt_rows=point_gt_rows,
        sweep_ids=sweep_ids,
        diagnostics={
            'temporal_frames': float(len(frame_ids)),
            'temporal_shared_groups': float(next_group),
            'temporal_current_label_targets': float(total_current_targets),
            'temporal_matched_current_targets': float(total_matched_targets),
            'temporal_target_points': float(total_temporal_target_points),
            'temporal_current_target_points': float(
                np.count_nonzero(target & (sweep_ids == 0))
            ),
            'temporal_historical_target_points': float(
                np.count_nonzero(target & (sweep_ids < 0))
            ),
            'temporal_max_alignment_error_m': maximum_alignment_error,
        },
    )
