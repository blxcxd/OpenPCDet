"""Attack objectives aligned with object-level detection disappearance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from pcdet.ops.iou3d_nms import iou3d_nms_utils

from ..analysis.object_evidence import (
    candidate_anchor_indices,
    flatten_anchors,
    object_evidence,
    reshape_anchor_cls_logits,
)
from .iadv import assign_points_to_oriented_boxes


OBJECT_OBJECTIVE_TYPES = {
    'object_evidence',
    'object_hybrid',
    'object_iou_s',
}


def _cross_2d(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _oriented_bev_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Return four counter-clockwise BEV corners for aligned 3D boxes."""
    unit_corners = boxes.new_tensor([
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ])
    local = unit_corners[None, :, :] * boxes[:, None, 3:5]
    cosine = torch.cos(boxes[:, 6])[:, None]
    sine = torch.sin(boxes[:, 6])[:, None]
    corner_x = local[..., 0] * cosine - local[..., 1] * sine
    corner_y = local[..., 0] * sine + local[..., 1] * cosine
    return torch.stack((corner_x, corner_y), dim=-1) + boxes[:, None, :2]


def _points_inside_oriented_bev(
    points: torch.Tensor,
    boxes: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    relative = points - boxes[:, None, :2]
    cosine = torch.cos(boxes[:, 6])[:, None]
    sine = torch.sin(boxes[:, 6])[:, None]
    local_x = relative[..., 0] * cosine + relative[..., 1] * sine
    local_y = -relative[..., 0] * sine + relative[..., 1] * cosine
    half_size = boxes[:, 3:5] * 0.5
    return (
        (local_x.abs() <= half_size[:, None, 0] + float(tolerance))
        & (local_y.abs() <= half_size[:, None, 1] + float(tolerance))
    )


def _oriented_bev_intersection_areas(
    first_boxes: torch.Tensor,
    second_boxes: torch.Tensor,
    tolerance: float = 1e-7,
) -> torch.Tensor:
    """Piecewise-differentiable aligned rotated-rectangle intersections.

    Polygon membership and vertex ordering are discrete, as they are for any
    exact rotated-IoU implementation. Once the local overlap topology is fixed,
    the selected vertices and shoelace area remain differentiable with respect
    to centers, dimensions, and yaw. This is sufficient for local attack
    gradients and avoids OpenPCDet's forward-only CUDA BEV-overlap kernel.
    """
    count = first_boxes.shape[0]
    relative_yaw = first_boxes[:, 6] - second_boxes[:, 6]
    parallel = torch.sin(relative_yaw).abs() <= float(tolerance)
    relative_center = first_boxes[:, :2] - second_boxes[:, :2]
    cosine = torch.cos(second_boxes[:, 6])
    sine = torch.sin(second_boxes[:, 6])
    local_center = torch.stack((
        relative_center[:, 0] * cosine + relative_center[:, 1] * sine,
        -relative_center[:, 0] * sine + relative_center[:, 1] * cosine,
    ), dim=-1)
    first_half = first_boxes[:, 3:5] * 0.5
    second_half = second_boxes[:, 3:5] * 0.5
    parallel_overlap = (
        torch.minimum(local_center + first_half, second_half)
        - torch.maximum(local_center - first_half, -second_half)
    ).clamp_min(0.0)
    parallel_area = parallel_overlap.prod(dim=-1)

    first_corners = _oriented_bev_corners(first_boxes)
    second_corners = _oriented_bev_corners(second_boxes)
    points = [first_corners, second_corners]
    validity = [
        _points_inside_oriented_bev(
            first_corners, second_boxes, tolerance
        ),
        _points_inside_oriented_bev(
            second_corners, first_boxes, tolerance
        ),
    ]

    first_end = torch.roll(first_corners, shifts=-1, dims=1)
    second_end = torch.roll(second_corners, shifts=-1, dims=1)
    first_start = first_corners[:, :, None, :].expand(count, 4, 4, 2)
    first_direction = (
        first_end - first_corners
    )[:, :, None, :].expand(count, 4, 4, 2)
    second_start = second_corners[:, None, :, :].expand(count, 4, 4, 2)
    second_direction = (
        second_end - second_corners
    )[:, None, :, :].expand(count, 4, 4, 2)
    offset = second_start - first_start
    denominator = _cross_2d(first_direction, second_direction)
    non_parallel = denominator.abs() > float(tolerance)
    safe_denominator = torch.where(
        non_parallel, denominator, torch.ones_like(denominator)
    )
    first_fraction = _cross_2d(offset, second_direction) / safe_denominator
    second_fraction = _cross_2d(offset, first_direction) / safe_denominator
    intersections = first_start + first_fraction[..., None] * first_direction
    intersects = (
        non_parallel
        & (first_fraction >= -float(tolerance))
        & (first_fraction <= 1.0 + float(tolerance))
        & (second_fraction >= -float(tolerance))
        & (second_fraction <= 1.0 + float(tolerance))
    )
    points.append(intersections.reshape(count, -1, 2))
    validity.append(intersects.reshape(count, -1))

    all_points = torch.cat(points, dim=1)
    valid = torch.cat(validity, dim=1)
    point_count = all_points.shape[1]
    # Remove repeated corner/edge-intersection vertices with one batched,
    # topology-only mask. This avoids per-anchor CPU/GPU synchronization.
    detached = all_points.detach()
    pair_distance = (
        detached[:, :, None, :] - detached[:, None, :, :]
    ).abs().amax(dim=-1)
    earlier = torch.tril(
        torch.ones(
            (point_count, point_count),
            device=all_points.device,
            dtype=torch.bool,
        ),
        diagonal=-1,
    )
    repeated = (
        (pair_distance <= float(tolerance))
        & earlier[None, :, :]
        & valid[:, None, :]
    ).any(dim=-1)
    valid = valid & ~repeated
    valid_count = valid.sum(dim=1)
    safe_count = valid_count.clamp_min(1)
    center = (
        all_points * valid[..., None].to(all_points.dtype)
    ).sum(dim=1) / safe_count[:, None]
    angles = torch.atan2(
        all_points[..., 1] - center[:, None, 1],
        all_points[..., 0] - center[:, None, 0],
    )
    angles = torch.where(valid, angles, torch.full_like(angles, float('inf')))
    order = torch.argsort(angles, dim=1)
    polygon = torch.gather(
        all_points, 1, order[..., None].expand(-1, -1, 2)
    )
    sorted_valid = torch.gather(valid, 1, order)
    positions = torch.arange(point_count, device=all_points.device)[None, :]
    next_positions = torch.remainder(positions + 1, safe_count[:, None])
    following = torch.gather(
        polygon, 1, next_positions[..., None].expand(-1, -1, 2)
    )
    polygon_area = 0.5 * (
        _cross_2d(polygon, following)
        * sorted_valid.to(polygon.dtype)
    ).sum(dim=1).abs()
    polygon_area = torch.where(
        valid_count >= 3, polygon_area, torch.zeros_like(polygon_area)
    )
    return torch.where(parallel, parallel_area, polygon_area)


def differentiable_oriented_iou3d(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """Aligned yaw-aware 3D IoU with local gradients for decoded boxes."""
    if predicted_boxes.ndim != 2 or predicted_boxes.shape[1] < 7:
        raise ValueError('predicted_boxes must have shape [N, >=7]')
    if target_boxes.shape != predicted_boxes.shape:
        raise ValueError('target_boxes must match predicted_boxes')
    if predicted_boxes.shape[0] == 0:
        return predicted_boxes.new_empty((0,))

    predicted = predicted_boxes[:, :7]
    target = target_boxes[:, :7]
    intersection_area = _oriented_bev_intersection_areas(predicted, target)
    predicted_z_min = predicted[:, 2] - 0.5 * predicted[:, 5]
    predicted_z_max = predicted[:, 2] + 0.5 * predicted[:, 5]
    target_z_min = target[:, 2] - 0.5 * target[:, 5]
    target_z_max = target[:, 2] + 0.5 * target[:, 5]
    intersection_height = (
        torch.minimum(predicted_z_max, target_z_max)
        - torch.maximum(predicted_z_min, target_z_min)
    ).clamp_min(0.0)
    intersection = intersection_area * intersection_height
    predicted_volume = predicted[:, 3:6].clamp_min(0.0).prod(dim=-1)
    target_volume = target[:, 3:6].clamp_min(0.0).prod(dim=-1)
    union = (predicted_volume + target_volume - intersection).clamp_min(1e-7)
    return (intersection / union).clamp(0.0, 1.0)


def radar_object_iou_s(
    class_logits: torch.Tensor,
    predicted_boxes: torch.Tensor,
    target_box: torch.Tensor,
    score_weight: float = 1.0,
    iou_weight: float = 1.0,
    log_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return the ascent form of IoU-S for one Radar target object.

    The optimizer used by the attacks performs gradient ascent. Therefore this
    returns ``log(1-score) + log(1-IoU)``: increasing it suppresses confidence
    and overlap. Candidate reduction is a mean so targets with more anchors do
    not receive a larger budget merely because of anchor density.
    """
    if class_logits.ndim != 1 or class_logits.numel() == 0:
        raise ValueError('class_logits must be a non-empty vector')
    if predicted_boxes.shape != (class_logits.numel(), 7):
        raise ValueError('predicted_boxes must have shape [N, 7]')
    if target_box.shape != (7,):
        raise ValueError('target_box must have shape [7]')
    if score_weight < 0 or iou_weight < 0:
        raise ValueError('IoU-S weights must be non-negative')
    if score_weight == 0 and iou_weight == 0:
        raise ValueError('at least one IoU-S weight must be positive')
    if not 0 < log_epsilon < 1:
        raise ValueError('IoU-S log epsilon must be in (0, 1)')
    scores = torch.sigmoid(class_logits)
    repeated_target = target_box[None, :].expand_as(predicted_boxes)
    ious = differentiable_oriented_iou3d(predicted_boxes, repeated_target)
    objective = (
        float(score_weight) * torch.log(1.0 - scores + float(log_epsilon))
        + float(iou_weight) * torch.log(1.0 - ious + float(log_epsilon))
    )
    return objective.mean()


@dataclass(frozen=True)
class ObjectEvidenceTarget:
    batch_index: int
    gt_index: int
    class_id: int
    candidate_indices: torch.Tensor
    clean_iou: float
    clean_score: float
    localization_indices: Optional[torch.Tensor] = None
    localization_targets: Optional[torch.Tensor] = None
    localization_weights: Optional[torch.Tensor] = None
    iou_s_indices: Optional[torch.Tensor] = None
    iou_s_gt_box: Optional[torch.Tensor] = None


class ObjectEvidenceObjective:
    """Attack fixed pre-NMS candidates around clean-detected objects.

    The classification term returns negative evidence, while the optional
    localization term increases encoded-box error. The existing point attacks
    maximize their sum. Candidate identities, weights, and regression targets
    are frozen on the clean input so iterative attacks cannot switch targets.
    ``object_iou_s`` instead uses decoded candidate confidence and yaw-aware
    3D IoU, while retaining the same fixed clean targets and point mask.
    """

    def __init__(
        self,
        targets: Sequence[ObjectEvidenceTarget],
        target_boxes_by_batch: Dict[int, torch.Tensor],
        num_classes: int,
        temperature: float,
        localization_weight: float = 0.0,
        objective_type: str = 'object_evidence',
        iou_s_score_weight: float = 1.0,
        iou_s_iou_weight: float = 1.0,
        iou_s_log_epsilon: float = 1e-6,
    ):
        if temperature <= 0:
            raise ValueError('object evidence temperature must be positive')
        if localization_weight < 0:
            raise ValueError('localization weight must be non-negative')
        if objective_type not in OBJECT_OBJECTIVE_TYPES:
            raise ValueError(f'unknown object objective {objective_type!r}')
        if iou_s_score_weight < 0 or iou_s_iou_weight < 0:
            raise ValueError('IoU-S weights must be non-negative')
        if (objective_type == 'object_iou_s'
                and iou_s_score_weight == 0 and iou_s_iou_weight == 0):
            raise ValueError('object_iou_s requires a positive loss weight')
        if not 0 < iou_s_log_epsilon < 1:
            raise ValueError('IoU-S log epsilon must be in (0, 1)')
        self.targets = list(targets)
        self.target_boxes_by_batch = {
            int(index): boxes.detach().clone()
            for index, boxes in target_boxes_by_batch.items()
        }
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.localization_weight = float(localization_weight)
        self.objective_type = objective_type
        self.iou_s_score_weight = float(iou_s_score_weight)
        self.iou_s_iou_weight = float(iou_s_iou_weight)
        self.iou_s_log_epsilon = float(iou_s_log_epsilon)

    @classmethod
    def from_clean_predictions(
        cls,
        model,
        batch_dict,
        clean_predictions,
        target_class_ids: Sequence[int],
        iou_threshold: Optional[float] = None,
        iou_thresholds: Optional[Mapping[int, float]] = None,
        candidate_margin: float = 1.0,
        candidate_topk: int = 32,
        temperature: float = 1.0,
        point_box_margin: float = 0.0,
        localization_weight: float = 0.0,
        localization_topk: int = 32,
        objective_type: str = 'object_evidence',
        iou_s_candidate_topk: int = 32,
        iou_s_score_weight: float = 1.0,
        iou_s_iou_weight: float = 1.0,
        iou_s_log_epsilon: float = 1e-6,
    ):
        if iou_threshold is None and iou_thresholds is None:
            raise ValueError('object evidence requires IoU thresholds')
        if iou_threshold is not None and not 0 <= iou_threshold <= 1:
            raise ValueError('object evidence IoU threshold must be in [0, 1]')
        resolved_thresholds = {
            int(class_id): float(value)
            for class_id, value in (iou_thresholds or {}).items()
        }
        if any(not 0 <= value <= 1 for value in resolved_thresholds.values()):
            raise ValueError('object evidence IoU thresholds must be in [0, 1]')
        if len(clean_predictions) != int(batch_dict['batch_size']):
            raise ValueError('clean prediction count does not match batch size')
        if batch_dict['gt_boxes'].shape[-1] < 8:
            raise ValueError('object evidence requires GT class ids')
        if point_box_margin < 0:
            raise ValueError('point box margin must be non-negative')
        if localization_weight < 0:
            raise ValueError('localization weight must be non-negative')
        if localization_topk <= 0:
            raise ValueError('localization top-k must be positive')
        if objective_type not in OBJECT_OBJECTIVE_TYPES:
            raise ValueError(f'unknown object objective {objective_type!r}')
        if iou_s_candidate_topk <= 0:
            raise ValueError('IoU-S candidate top-k must be positive')
        target_class_ids = {int(value) for value in target_class_ids}
        if not target_class_ids:
            raise ValueError('object evidence requires at least one target class')
        for class_id in target_class_ids:
            if class_id not in resolved_thresholds:
                if iou_threshold is None:
                    raise ValueError(
                        f'missing object evidence IoU threshold for class {class_id}'
                    )
                resolved_thresholds[class_id] = float(iou_threshold)

        anchors = flatten_anchors(model.dense_head)
        clean_cls_logits = None
        clean_box_encodings = None
        if localization_weight > 0 or objective_type == 'object_iou_s':
            clean_cls_logits = reshape_anchor_cls_logits(
                model.dense_head.forward_ret_dict['cls_preds'],
                model.num_class,
            ).detach()
            raw_box_preds = model.dense_head.forward_ret_dict['box_preds']
            code_size = int(model.dense_head.box_coder.code_size)
            clean_box_encodings = raw_box_preds.reshape(
                raw_box_preds.shape[0], -1, code_size
            ).detach()
            if clean_box_encodings.shape[1] != anchors.shape[0]:
                raise RuntimeError(
                    'clean box encodings do not match flattened anchors'
                )
        targets: List[ObjectEvidenceTarget] = []
        target_boxes_by_batch: Dict[int, torch.Tensor] = {}
        for batch_index in range(int(batch_dict['batch_size'])):
            gt_boxes = batch_dict['gt_boxes'][batch_index]
            valid = (gt_boxes[:, 3:6] > 0).all(dim=1)
            valid &= torch.isin(
                gt_boxes[:, -1].long(),
                torch.as_tensor(
                    sorted(target_class_ids),
                    device=gt_boxes.device,
                    dtype=torch.long,
                ),
            )
            gt_indices = torch.nonzero(valid, as_tuple=False).flatten()
            selected_boxes = []
            prediction = clean_predictions[batch_index]
            for gt_index in gt_indices.tolist():
                class_id = int(gt_boxes[gt_index, -1].item())
                prediction_mask = prediction['pred_labels'].long() == class_id
                predicted_boxes = prediction['pred_boxes'][prediction_mask, :7]
                predicted_scores = prediction['pred_scores'][prediction_mask]
                if predicted_boxes.shape[0] == 0:
                    continue
                ious = iou3d_nms_utils.boxes_iou3d_gpu(
                    predicted_boxes, gt_boxes[gt_index:gt_index + 1, :7]
                ).squeeze(1)
                best_iou, best_index = ious.max(dim=0)
                if float(best_iou.item()) <= resolved_thresholds[class_id]:
                    continue
                candidates = candidate_anchor_indices(
                    anchors,
                    gt_boxes[gt_index, :7],
                    margin=candidate_margin,
                    fallback_topk=candidate_topk,
                )
                localization_indices = None
                localization_targets = None
                localization_weights = None
                iou_s_indices = None
                iou_s_gt_box = None
                if localization_weight > 0 or objective_type == 'object_iou_s':
                    candidate_boxes = model.dense_head.box_coder.decode_torch(
                        clean_box_encodings[batch_index, candidates],
                        anchors[candidates],
                    )
                    candidate_ious = iou3d_nms_utils.boxes_iou3d_gpu(
                        candidate_boxes[:, :7],
                        gt_boxes[gt_index:gt_index + 1, :7],
                    ).squeeze(1)
                    candidate_logits = clean_cls_logits[
                        batch_index, candidates, class_id - 1
                    ]
                    # IoU determines relevance; clean confidence breaks ties.
                    ranking = candidate_ious + 1e-4 * torch.sigmoid(
                        candidate_logits
                    )
                    requested_topk = (
                        iou_s_candidate_topk
                        if objective_type == 'object_iou_s'
                        else localization_topk
                    )
                    count = min(int(requested_topk), int(candidates.numel()))
                    local_rows = torch.topk(
                        ranking, k=count, largest=True, sorted=True
                    ).indices
                    selected_indices = candidates[local_rows].detach().clone()
                    if localization_weight > 0:
                        localization_indices = selected_indices
                        localization_weights = torch.softmax(
                            candidate_logits[local_rows] / float(temperature),
                            dim=0,
                        ).detach().clone()
                        localization_anchors = anchors[
                            localization_indices
                        ].detach().clone()
                        repeated_gt = localization_anchors.new_zeros(
                            (count, code_size)
                        )
                        repeated_gt[:, :7] = gt_boxes[gt_index, :7]
                        localization_targets = (
                            model.dense_head.box_coder.encode_torch(
                                repeated_gt, localization_anchors
                            ).detach().clone()
                        )
                    if objective_type == 'object_iou_s':
                        iou_s_indices = selected_indices
                        iou_s_gt_box = gt_boxes[gt_index, :7].detach().clone()
                targets.append(
                    ObjectEvidenceTarget(
                        batch_index=batch_index,
                        gt_index=int(gt_index),
                        class_id=class_id,
                        candidate_indices=candidates.detach().clone(),
                        clean_iou=float(best_iou.item()),
                        clean_score=float(predicted_scores[best_index].item()),
                        localization_indices=localization_indices,
                        localization_targets=localization_targets,
                        localization_weights=localization_weights,
                        iou_s_indices=iou_s_indices,
                        iou_s_gt_box=iou_s_gt_box,
                    )
                )
                point_box = gt_boxes[gt_index, :7].clone()
                point_box[3:6] += 2.0 * float(point_box_margin)
                selected_boxes.append(point_box)
            if selected_boxes:
                target_boxes_by_batch[batch_index] = torch.stack(selected_boxes)

        return cls(
            targets=targets,
            target_boxes_by_batch=target_boxes_by_batch,
            num_classes=model.num_class,
            temperature=temperature,
            localization_weight=localization_weight,
            objective_type=objective_type,
            iou_s_score_weight=iou_s_score_weight,
            iou_s_iou_weight=iou_s_iou_weight,
            iou_s_log_epsilon=iou_s_log_epsilon,
        )

    def __call__(self, model) -> torch.Tensor:
        if self.objective_type == 'object_iou_s':
            return self._iou_s_objective(model)
        objectives = []
        for _, classification, localization in self.component_objectives(model):
            objectives.append(
                classification + self.localization_weight * localization
            )
        if not objectives:
            return self._current_logits(model).sum() * 0.0
        return torch.stack(objectives).mean()

    def _iou_s_objective(self, model) -> torch.Tensor:
        cls_logits, decoded_boxes = self._current_decoded_predictions(model)
        objectives_by_class: Dict[int, List[torch.Tensor]] = {}
        for target in self.targets:
            if target.iou_s_indices is None or target.iou_s_gt_box is None:
                raise RuntimeError('IoU-S target lacks fixed candidates or GT box')
            indices = target.iou_s_indices
            objective = radar_object_iou_s(
                class_logits=cls_logits[
                    target.batch_index, indices, target.class_id - 1
                ],
                predicted_boxes=decoded_boxes[
                    target.batch_index, indices, :7
                ],
                target_box=target.iou_s_gt_box,
                score_weight=self.iou_s_score_weight,
                iou_weight=self.iou_s_iou_weight,
                log_epsilon=self.iou_s_log_epsilon,
            )
            objectives_by_class.setdefault(target.class_id, []).append(
                objective
            )
        if not objectives_by_class:
            return cls_logits.sum() * 0.0
        class_objectives = [
            torch.stack(objectives).mean()
            for objectives in objectives_by_class.values()
        ]
        return torch.stack(class_objectives).mean()

    def restricted_to_gt_rows(
        self,
        allowed_rows_by_batch: Mapping[int, Sequence[int]],
        batch_dict,
        point_box_margin: float = 0.0,
    ):
        """Return the same objective restricted to evaluator-eligible GT rows.

        This changes only the selected clean targets. Candidate anchors and the
        classification/localization objective for every retained target remain
        exactly as constructed on the clean forward pass.
        """
        allowed = {
            int(batch_index): {int(row) for row in rows}
            for batch_index, rows in allowed_rows_by_batch.items()
        }
        targets = [
            target for target in self.targets
            if target.gt_index in allowed.get(target.batch_index, set())
        ]
        target_boxes_by_batch = {}
        for batch_index, rows in allowed.items():
            selected = [
                target.gt_index for target in targets
                if target.batch_index == batch_index
            ]
            if not selected:
                continue
            boxes = batch_dict['gt_boxes'][batch_index, selected, :7].clone()
            boxes[:, 3:6] += 2.0 * float(point_box_margin)
            target_boxes_by_batch[batch_index] = boxes
        return ObjectEvidenceObjective(
            targets=targets,
            target_boxes_by_batch=target_boxes_by_batch,
            num_classes=self.num_classes,
            temperature=self.temperature,
            localization_weight=self.localization_weight,
            objective_type=self.objective_type,
            iou_s_score_weight=self.iou_s_score_weight,
            iou_s_iou_weight=self.iou_s_iou_weight,
            iou_s_log_epsilon=self.iou_s_log_epsilon,
        )

    def component_objectives(self, model):
        """Return fixed-target classification and localization objectives.

        This keeps the production attack and gradient diagnostics on exactly
        the same mathematical terms. Both components are ascent objectives:
        classification suppresses object evidence and localization increases
        encoded regression error.
        """
        cls_logits = self._current_logits(model)
        box_encodings = (
            self._current_box_encodings(model)
            if self.localization_weight > 0
            else None
        )
        components = []
        for target in self.targets:
            classification = -object_evidence(
                cls_logits[target.batch_index],
                target.candidate_indices,
                target.class_id - 1,
                self.temperature,
            )
            localization = classification.new_zeros(())
            if self.localization_weight > 0:
                if (
                    target.localization_indices is None
                    or target.localization_targets is None
                    or target.localization_weights is None
                ):
                    raise RuntimeError(
                        'hybrid objective target lacks localization candidates'
                    )
                predictions = box_encodings[
                    target.batch_index, target.localization_indices
                ]
                predictions_sin, targets_sin = (
                    model.dense_head.add_sin_difference(
                        predictions, target.localization_targets
                    )
                )
                localization_errors = F.smooth_l1_loss(
                    predictions_sin,
                    targets_sin,
                    reduction='none',
                    beta=1.0,
                ).sum(dim=-1)
                localization = (
                    localization_errors * target.localization_weights
                ).sum()
            components.append((target, classification, localization))
        return components

    def _current_logits(self, model) -> torch.Tensor:
        raw_logits = model.dense_head.forward_ret_dict['cls_preds']
        cls_logits = reshape_anchor_cls_logits(raw_logits, self.num_classes)
        if self.targets:
            largest_index = max(
                int(target.candidate_indices.max().item())
                for target in self.targets
            )
            if largest_index >= cls_logits.shape[1]:
                raise RuntimeError(
                    'fixed object candidate index exceeds current anchor logits'
                )
        return cls_logits

    def _current_box_encodings(self, model) -> torch.Tensor:
        raw_box_preds = model.dense_head.forward_ret_dict['box_preds']
        code_size = int(model.dense_head.box_coder.code_size)
        box_encodings = raw_box_preds.reshape(
            raw_box_preds.shape[0], -1, code_size
        )
        if self.targets:
            largest_index = max(
                int(target.localization_indices.max().item())
                for target in self.targets
                if target.localization_indices is not None
            )
            if largest_index >= box_encodings.shape[1]:
                raise RuntimeError(
                    'fixed localization candidate exceeds current box outputs'
                )
        return box_encodings

    def _current_decoded_predictions(self, model):
        forward = model.dense_head.forward_ret_dict
        cls_logits, decoded_boxes = model.dense_head.generate_predicted_boxes(
            batch_size=int(forward['cls_preds'].shape[0]),
            cls_preds=forward['cls_preds'],
            box_preds=forward['box_preds'],
            dir_cls_preds=forward.get('dir_cls_preds'),
        )
        if self.targets:
            largest_index = max(
                int(target.iou_s_indices.max().item())
                for target in self.targets
                if target.iou_s_indices is not None
            )
            if largest_index >= cls_logits.shape[1]:
                raise RuntimeError(
                    'fixed IoU-S candidate exceeds current decoded outputs'
                )
        return cls_logits, decoded_boxes

    def evidence_by_target(self, model) -> Dict[tuple[int, int], float]:
        """Read current fixed-candidate evidence without retaining its graph."""
        cls_logits = self._current_logits(model)
        values = {}
        with torch.no_grad():
            for target in self.targets:
                evidence = object_evidence(
                    cls_logits[target.batch_index],
                    target.candidate_indices,
                    target.class_id - 1,
                    self.temperature,
                )
                values[(target.batch_index, target.gt_index)] = float(
                    evidence.item()
                )
        return values

    def point_mask(self, points: torch.Tensor) -> torch.Tensor:
        """Return the union of clean-detected target boxes for each sample."""
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError('points must have batch index and xyz columns')
        mask = torch.zeros(
            points.shape[0], dtype=torch.bool, device=points.device
        )
        batch_indices = points[:, 0].long()
        for batch_index, boxes in self.target_boxes_by_batch.items():
            sample_indices = torch.nonzero(
                batch_indices == batch_index, as_tuple=False
            ).flatten()
            if sample_indices.numel() == 0:
                continue
            assignments = assign_points_to_oriented_boxes(
                points[sample_indices, 1:4], boxes
            )
            mask[sample_indices] = assignments >= 0
        return mask

    @property
    def stats(self) -> Dict[str, float]:
        return {
            'object_evidence_targets': float(len(self.targets)),
            'object_evidence_candidate_anchors': float(
                sum(target.candidate_indices.numel() for target in self.targets)
            ),
            'object_hybrid_localization_anchors': float(
                sum(
                    0
                    if target.localization_indices is None
                    else target.localization_indices.numel()
                    for target in self.targets
                )
            ),
            'object_hybrid_localization_weight': self.localization_weight,
            'object_iou_s_targets': float(
                sum(target.iou_s_indices is not None for target in self.targets)
            ),
            'object_iou_s_candidate_anchors': float(
                sum(
                    0 if target.iou_s_indices is None
                    else target.iou_s_indices.numel()
                    for target in self.targets
                )
            ),
            'object_iou_s_score_weight': self.iou_s_score_weight,
            'object_iou_s_iou_weight': self.iou_s_iou_weight,
        }
