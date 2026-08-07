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


class ObjectEvidenceObjective:
    """Attack fixed pre-NMS candidates around clean-detected objects.

    The classification term returns negative evidence, while the optional
    localization term increases encoded-box error. The existing point attacks
    maximize their sum. Candidate identities, weights, and regression targets
    are frozen on the clean input so iterative attacks cannot switch targets.
    """

    def __init__(
        self,
        targets: Sequence[ObjectEvidenceTarget],
        target_boxes_by_batch: Dict[int, torch.Tensor],
        num_classes: int,
        temperature: float,
        localization_weight: float = 0.0,
    ):
        if temperature <= 0:
            raise ValueError('object evidence temperature must be positive')
        if localization_weight < 0:
            raise ValueError('localization weight must be non-negative')
        self.targets = list(targets)
        self.target_boxes_by_batch = {
            int(index): boxes.detach().clone()
            for index, boxes in target_boxes_by_batch.items()
        }
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.localization_weight = float(localization_weight)

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
        if localization_weight > 0:
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
                if localization_weight > 0:
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
                    count = min(int(localization_topk), int(candidates.numel()))
                    local_rows = torch.topk(
                        ranking, k=count, largest=True, sorted=True
                    ).indices
                    localization_indices = candidates[local_rows].detach().clone()
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
        )

    def __call__(self, model) -> torch.Tensor:
        objectives = []
        for _, classification, localization in self.component_objectives(model):
            objectives.append(
                classification + self.localization_weight * localization
            )
        if not objectives:
            return self._current_logits(model).sum() * 0.0
        return torch.stack(objectives).mean()

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
        }
