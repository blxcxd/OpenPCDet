"""Object-level, mutually exclusive outcomes for detector attacks."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from pcdet.ops.iou3d_nms import iou3d_nms_utils


OUTCOME_NAMES = (
    'still_correct',
    'misclassification',
    'localization_failure',
    'pure_hiding',
)


def _filtered_prediction(prediction, score_threshold: float):
    keep = prediction['pred_scores'] >= float(score_threshold)
    return {
        'boxes': prediction['pred_boxes'][keep, :7],
        'scores': prediction['pred_scores'][keep],
        'labels': prediction['pred_labels'][keep].long(),
    }


def _greedy_match(
    ious: torch.Tensor,
    gt_rows: Sequence[int],
    pred_rows: Sequence[int],
    valid_pair,
):
    """Greedily select one-to-one pairs in descending 3D-IoU order."""
    candidates = []
    for gt_row in gt_rows:
        for pred_row in pred_rows:
            iou = float(ious[pred_row, gt_row].item())
            if valid_pair(gt_row, pred_row, iou):
                candidates.append((iou, int(gt_row), int(pred_row)))
    candidates.sort(reverse=True)
    matches = {}
    used_predictions = set()
    for iou, gt_row, pred_row in candidates:
        if gt_row in matches or pred_row in used_predictions:
            continue
        matches[gt_row] = (pred_row, iou)
        used_predictions.add(pred_row)
    return matches, used_predictions


def _box_change_metrics(clean_box, adversarial_box, gt_box) -> Dict:
    if clean_box is None or adversarial_box is None:
        return {
            'prediction_center_shift': None,
            'clean_center_error': None,
            'adversarial_center_error': None,
            'center_error_increase': None,
            'prediction_size_l1_shift': None,
            'prediction_yaw_shift': None,
        }
    clean_center_error = torch.linalg.vector_norm(clean_box[:3] - gt_box[:3])
    adversarial_center_error = torch.linalg.vector_norm(
        adversarial_box[:3] - gt_box[:3]
    )
    yaw_difference = adversarial_box[6] - clean_box[6]
    yaw_shift = torch.atan2(
        torch.sin(yaw_difference), torch.cos(yaw_difference)
    ).abs()
    return {
        'prediction_center_shift': float(
            torch.linalg.vector_norm(
                adversarial_box[:3] - clean_box[:3]
            ).item()
        ),
        'clean_center_error': float(clean_center_error.item()),
        'adversarial_center_error': float(adversarial_center_error.item()),
        'center_error_increase': float(
            (adversarial_center_error - clean_center_error).item()
        ),
        'prediction_size_l1_shift': float(
            (adversarial_box[3:6] - clean_box[3:6]).abs().sum().item()
        ),
        'prediction_yaw_shift': float(yaw_shift.item()),
    }


def _best_iou_and_center_distance(prediction, gt_box, class_id=None):
    labels = prediction['labels']
    keep = torch.ones_like(labels, dtype=torch.bool)
    if class_id is not None:
        keep &= labels == int(class_id)
    boxes = prediction['boxes'][keep]
    if boxes.shape[0] == 0:
        return 0.0, None
    ious = iou3d_nms_utils.boxes_iou3d_gpu(
        boxes, gt_box[None, :7]
    ).squeeze(1)
    center_distances = torch.linalg.vector_norm(
        boxes[:, :3] - gt_box[None, :3], dim=1
    )
    return float(ious.max().item()), float(center_distances.min().item())


def compare_target_object_endpoints(
    clean_prediction,
    adversarial_prediction,
    gt_boxes: torch.Tensor,
    target_classes: Mapping[int, str],
    iou_thresholds: Mapping[int, float],
    frame_id: str,
    object_score_threshold: float,
    class_names: Optional[Mapping[int, str]] = None,
    batch_index: int = 0,
    clean_evidence: Optional[Mapping[Tuple[int, int], float]] = None,
    adversarial_evidence: Optional[Mapping[Tuple[int, int], float]] = None,
):
    """Classify every clean-detected target into one attack endpoint.

    Clean eligibility and all attacked matches are one-to-one. Attacked
    targets are assigned in priority order: correct same-class detection,
    spatially correct wrong-class detection, any overlapping prediction, then
    no overlapping prediction (pure hiding).
    """
    if gt_boxes.shape[1] < 8:
        raise ValueError('object endpoint metrics require GT class ids')
    if not 0 <= object_score_threshold <= 1:
        raise ValueError('object score threshold must be in [0, 1]')
    class_ids = {int(value) for value in target_classes}
    if not class_ids:
        raise ValueError('at least one target class is required')
    missing = class_ids.difference(int(value) for value in iou_thresholds)
    if missing:
        raise ValueError(f'missing IoU thresholds for class ids: {sorted(missing)}')

    valid = (gt_boxes[:, 3:6] > 0).all(dim=1)
    valid &= torch.isin(
        gt_boxes[:, -1].long(),
        torch.as_tensor(sorted(class_ids), device=gt_boxes.device),
    )
    selected_gt_rows = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    if not selected_gt_rows:
        return 0, []

    clean = _filtered_prediction(clean_prediction, object_score_threshold)
    adversarial = _filtered_prediction(
        adversarial_prediction, object_score_threshold
    )
    gt_geometry = gt_boxes[:, :7]
    clean_ious = (
        iou3d_nms_utils.boxes_iou3d_gpu(clean['boxes'], gt_geometry)
        if clean['boxes'].shape[0]
        else gt_boxes.new_zeros((0, gt_boxes.shape[0]))
    )
    adversarial_ious = (
        iou3d_nms_utils.boxes_iou3d_gpu(adversarial['boxes'], gt_geometry)
        if adversarial['boxes'].shape[0]
        else gt_boxes.new_zeros((0, gt_boxes.shape[0]))
    )

    def strict_threshold(gt_row):
        class_id = int(gt_boxes[gt_row, -1].item())
        return float(iou_thresholds[class_id])

    clean_matches, _ = _greedy_match(
        clean_ious,
        selected_gt_rows,
        range(clean['boxes'].shape[0]),
        lambda gt_row, pred_row, iou: (
            int(clean['labels'][pred_row].item())
            == int(gt_boxes[gt_row, -1].item())
            and iou > strict_threshold(gt_row)
        ),
    )
    eligible_gt_rows = sorted(clean_matches)
    if not eligible_gt_rows:
        return len(selected_gt_rows), []

    remaining_gt = set(eligible_gt_rows)
    remaining_pred = set(range(adversarial['boxes'].shape[0]))
    assigned = {}
    stages = (
        (
            'still_correct',
            lambda gt_row, pred_row, iou: (
                int(adversarial['labels'][pred_row].item())
                == int(gt_boxes[gt_row, -1].item())
                and iou > strict_threshold(gt_row)
            ),
        ),
        (
            'misclassification',
            lambda gt_row, pred_row, iou: (
                int(adversarial['labels'][pred_row].item())
                != int(gt_boxes[gt_row, -1].item())
                and iou > strict_threshold(gt_row)
            ),
        ),
        ('localization_failure', lambda _gt, _pred, iou: iou > 0.0),
    )
    for outcome, predicate in stages:
        matches, used_predictions = _greedy_match(
            adversarial_ious,
            sorted(remaining_gt),
            sorted(remaining_pred),
            predicate,
        )
        for gt_row, (pred_row, iou) in matches.items():
            assigned[gt_row] = (outcome, pred_row, iou)
        remaining_gt.difference_update(matches)
        remaining_pred.difference_update(used_predictions)
    for gt_row in remaining_gt:
        assigned[gt_row] = ('pure_hiding', None, 0.0)

    records = []
    for object_id, gt_row in enumerate(eligible_gt_rows):
        class_id = int(gt_boxes[gt_row, -1].item())
        threshold = strict_threshold(gt_row)
        clean_pred_row, clean_iou = clean_matches[gt_row]
        outcome, adversarial_pred_row, assigned_iou = assigned[gt_row]
        clean_box = clean['boxes'][clean_pred_row].detach()
        clean_score = float(clean['scores'][clean_pred_row].item())
        if adversarial_pred_row is None:
            adversarial_box = None
            adversarial_score = 0.0
            adversarial_label_id = None
            adversarial_label_name = None
        else:
            adversarial_box = adversarial['boxes'][adversarial_pred_row].detach()
            adversarial_score = float(
                adversarial['scores'][adversarial_pred_row].item()
            )
            adversarial_label_id = int(
                adversarial['labels'][adversarial_pred_row].item()
            )
            adversarial_label_name = (class_names or target_classes).get(
                adversarial_label_id, str(adversarial_label_id)
            )

        same_class_iou, _ = _best_iou_and_center_distance(
            adversarial, gt_boxes[gt_row, :7], class_id=class_id
        )
        any_class_iou, nearest_center_distance = _best_iou_and_center_distance(
            adversarial, gt_boxes[gt_row, :7]
        )
        key = (int(batch_index), int(gt_row))
        clean_object_evidence = (
            None if clean_evidence is None else clean_evidence.get(key)
        )
        adversarial_object_evidence = (
            None
            if adversarial_evidence is None
            else adversarial_evidence.get(key)
        )
        records.append({
            'frame_id': str(frame_id),
            'batch_index': int(batch_index),
            'object_id': int(object_id),
            'gt_row': int(gt_row),
            'class_name': str(target_classes[class_id]),
            'class_id': class_id,
            'iou_threshold': threshold,
            'object_score_threshold': float(object_score_threshold),
            'outcome': outcome,
            'object_failure': outcome != 'still_correct',
            'pure_hiding': outcome == 'pure_hiding',
            'clean_max_iou': float(clean_iou),
            'adversarial_max_iou': same_class_iou,
            'adversarial_any_class_max_iou': any_class_iou,
            'assigned_adversarial_iou': float(assigned_iou),
            'nearest_adversarial_center_distance': nearest_center_distance,
            'max_iou_drop': float(clean_iou) - same_class_iou,
            'clean_iou_margin': float(clean_iou) - threshold,
            'adversarial_iou_margin': same_class_iou - threshold,
            'clean_match_score': clean_score,
            'adversarial_match_score': adversarial_score,
            'match_score_drop': clean_score - adversarial_score,
            'adversarial_label_id': adversarial_label_id,
            'adversarial_label_name': adversarial_label_name,
            'clean_object_evidence': clean_object_evidence,
            'adversarial_object_evidence': adversarial_object_evidence,
            'object_evidence_drop': (
                None
                if clean_object_evidence is None
                or adversarial_object_evidence is None
                else clean_object_evidence - adversarial_object_evidence
            ),
            **_box_change_metrics(
                clean_box, adversarial_box, gt_boxes[gt_row, :7]
            ),
        })
    return len(selected_gt_rows), records
