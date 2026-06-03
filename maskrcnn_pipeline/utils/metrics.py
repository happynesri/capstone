from __future__ import annotations

import cv2
import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def mask_area_consistency(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> dict[str, float]:
    if not pred_masks or not gt_masks:
        return {"mean_abs_area_ratio_error": float("nan")}

    ratios = []
    for pred, gt in zip(pred_masks, gt_masks):
        gt_area = float(gt.astype(bool).sum())
        if gt_area > 0:
            ratios.append(abs(float(pred.astype(bool).sum()) / gt_area - 1.0))
    value = float(np.mean(ratios)) if ratios else float("nan")
    return {"mean_abs_area_ratio_error": value}


def boundary_smoothness(mask: np.ndarray) -> float:
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    area = float(binary.sum())
    if area <= 0.0:
        return 0.0
    return float((perimeter * perimeter) / (4.0 * np.pi * area))
