from __future__ import annotations

import cv2
import numpy as np


def draw_instances(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    output = image_bgr.copy()
    overlay = output.copy()

    for mask, box, score in zip(masks, boxes, scores):
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (0, 255, 0)

        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            output,
            f"rock {score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, dst=output)
    return output
