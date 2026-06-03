from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as T

from maskrcnn_pipeline.models import build_maskrcnn
from maskrcnn_pipeline.utils import draw_instances, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mask R-CNN rock segmentation inference.")
    parser.add_argument("--weights", type=Path, default=Path("maskrcnn_pipeline/runs/maskrcnn_rock/best.pth"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("maskrcnn_pipeline/runs/maskrcnn_rock/debug.png"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = build_maskrcnn(num_classes=2, weights=None)
    checkpoint = torch.load(args.weights, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    tensor = preprocess(image_bgr).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        pred = model([tensor])[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    masks, boxes, scores = filter_predictions(pred, args.threshold)
    debug = draw_instances(image_bgr, masks, boxes, scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), debug)

    print(f"inference_ms={elapsed_ms:.2f}")
    print(f"instances={len(scores)}")
    for idx, (box, score, mask) in enumerate(zip(boxes, scores, masks), start=1):
        print(f"{idx}: score={score:.4f} bbox={box.tolist()} mask_area={int(mask.sum())}")
    print(f"debug_image={args.output}")


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms(image_rgb)


def filter_predictions(pred: dict[str, torch.Tensor], threshold: float):
    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    keep = (labels == 1) & (scores >= threshold)

    boxes = pred["boxes"].detach().cpu().numpy()[keep]
    kept_scores = scores[keep]
    masks = pred["masks"].detach().cpu().numpy()[keep, 0] >= threshold
    return masks.astype(np.uint8), boxes.astype(np.float32), kept_scores.astype(np.float32)


if __name__ == "__main__":
    main()
