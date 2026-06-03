from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from maskrcnn_pipeline.dataset import RockCocoDataset, collate_fn
from maskrcnn_pipeline.models import build_maskrcnn
from maskrcnn_pipeline.utils import seed_everything


# Validation evaluation:
# python3 -u -m maskrcnn_pipeline.evaluate_maskrcnn \
#   --weights /home/sanghwon/capstone/runs/maskrcnn_rock/best.pth \
#   --split val \
#   --conf-thres 0.001 \
#   --iou-thres 0.5 \
#   --output-dir /home/sanghwon/capstone/runs/maskrcnn_rock/eval_val \
#   --num-workers 4
#
# Test evaluation:
# python3 -u -m maskrcnn_pipeline.evaluate_maskrcnn \
#   --weights /home/sanghwon/capstone/runs/maskrcnn_rock/best.pth \
#   --split test \
#   --conf-thres 0.001 \
#   --iou-thres 0.5 \
#   --output-dir /home/sanghwon/capstone/runs/maskrcnn_rock/eval_test \
#   --num-workers 4
#
# Fast sample evaluation:
# python3 -u -m maskrcnn_pipeline.evaluate_maskrcnn \
#   --weights /home/sanghwon/capstone/runs/maskrcnn_rock/best.pth \
#   --split val \
#   --max-images 500 \
#   --output-dir /home/sanghwon/capstone/runs/maskrcnn_rock/eval_sample


CLASS_NAMES = ["rock"]
CURVE_COLOR = "#1f77b4"


@dataclass
class ImageEval:
    scores: np.ndarray
    pred_labels: np.ndarray
    gt_labels: np.ndarray
    box_iou: np.ndarray
    mask_iou: np.ndarray
    gt_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN and generate YOLO-style plots.")
    parser.add_argument("--weights", type=Path, default=Path("/home/sanghwon/capstone/runs/maskrcnn_rock/best.pth"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/sanghwon/capstone/runs/maskrcnn_rock/eval"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("maskrcnn_pipeline/dataset"))
    parser.add_argument("--image-root", type=Path, default=Path("/home/sanghwon/capstone/datasets/rock_det"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-pred-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    json_path = args.data_dir / f"{args.split}.json"
    dataset = RockCocoDataset(json_path, args.image_root)
    eval_dataset = Subset(dataset, range(min(args.max_images, len(dataset)))) if args.max_images else dataset
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    print(f"[INFO] device: {device}", flush=True)
    weights_path = resolve_weights_path(args.weights)
    print(f"[INFO] weights: {weights_path}", flush=True)
    print(f"[INFO] split: {args.split}", flush=True)
    print(f"[INFO] images: {len(eval_dataset)}", flush=True)
    print(f"[INFO] conf_thres: {args.conf_thres}", flush=True)
    print(f"[INFO] iou_thres: {args.iou_thres}", flush=True)

    model = build_maskrcnn(num_classes=2, weights=None)
    checkpoint = load_checkpoint(weights_path, device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()

    image_evals, timing = collect_predictions(model, loader, device, args)
    thresholds = np.linspace(0.0, 1.0, 101)

    box_curve = compute_curve(image_evals, thresholds, args.iou_thres, metric="box")
    mask_curve = compute_curve(image_evals, thresholds, args.iou_thres, metric="mask")
    box_best = best_curve_point(box_curve)
    mask_best = best_curve_point(mask_curve)
    box_ap50 = compute_ap50(box_curve["recall"], box_curve["precision"])
    mask_ap50 = compute_ap50(mask_curve["recall"], mask_curve["precision"])

    num_gt = int(sum(item.gt_count for item in image_evals))
    num_pred = int(sum((item.scores >= args.conf_thres).sum() for item in image_evals))
    summary = {
        "box_precision_best": box_best["precision"],
        "box_recall_best": box_best["recall"],
        "box_f1_best": box_best["f1"],
        "box_best_conf": box_best["conf"],
        "box_ap50": box_ap50,
        "mask_precision_best": mask_best["precision"],
        "mask_recall_best": mask_best["recall"],
        "mask_f1_best": mask_best["f1"],
        "mask_best_conf": mask_best["conf"],
        "mask_ap50": mask_ap50,
        "num_images": len(image_evals),
        "num_gt_instances": num_gt,
        "num_pred_instances": num_pred,
        "inference_ms_per_image": timing["inference_ms_per_image"],
        "total_evaluation_time_s": timing["total_evaluation_time_s"],
    }

    print(f"[RESULT] box AP50={box_ap50:.4f}", flush=True)
    print(f"[RESULT] mask AP50={mask_ap50:.4f}", flush=True)
    print(f"[RESULT] best F1={box_best['f1']:.4f} at conf={box_best['conf']:.2f}", flush=True)

    save_curve_plots(args.output_dir, "Box", box_curve, box_ap50, box_best)
    save_curve_plots(args.output_dir, "Mask", mask_curve, mask_ap50, mask_best)

    matrix = confusion_matrix_from_evals(image_evals, box_best["conf"], args.iou_thres, metric="box")
    save_confusion_matrix(args.output_dir / "confusion_matrix.png", matrix, normalized=False)
    save_confusion_matrix(args.output_dir / "confusion_matrix_normalized.png", matrix, normalized=True)

    save_labels_plot(json_path, args.output_dir / "labels.jpg", max_images=args.max_images)
    save_results(args.output_dir, summary, checkpoint)

    summary_path = args.output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] {summary_path.name}", flush=True)

    if args.yolo_pred_json:
        compare_yolo_predictions(args.yolo_pred_json)


def collect_predictions(model, loader, device, args) -> tuple[list[ImageEval], dict[str, float]]:
    image_evals: list[ImageEval] = []
    inference_ms: list[float] = []
    total_images = len(loader.dataset)
    start_total = time.perf_counter()
    seen = 0

    for images, targets in loader:
        images_on_device = [image.to(device, non_blocking=True) for image in images]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            preds = model(images_on_device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - start
        inference_ms.extend([batch_elapsed * 1000.0 / max(1, len(images))] * len(images))

        for pred, target in zip(preds, targets):
            item = build_image_eval(pred, target, args.conf_thres)
            image_evals.append(item)
            seen += 1
            if seen == 1 or seen % 100 == 0 or seen == total_images:
                elapsed = time.perf_counter() - start_total
                eta = (elapsed / seen) * (total_images - seen) if seen else 0.0
                print(
                    f"[EVAL] image {seen}/{total_images} | preds={len(item.scores)} | "
                    f"gt={item.gt_count} | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}",
                    flush=True,
                )

    total_elapsed = time.perf_counter() - start_total
    return image_evals, {
        "inference_ms_per_image": float(np.mean(inference_ms)) if inference_ms else 0.0,
        "total_evaluation_time_s": float(total_elapsed),
    }


def resolve_weights_path(path: Path) -> Path:
    if path.exists():
        return path

    candidates = [
        Path("maskrcnn_pipeline/runs/maskrcnn_rock/best.pth"),
        Path("maskrcnn_pipeline/runs/maskrcnn_rock/last.pth"),
        Path("/home/sanghwon/capstone/maskrcnn_pipeline/runs/maskrcnn_rock/best.pth"),
        Path("/home/sanghwon/capstone/maskrcnn_pipeline/runs/maskrcnn_rock/last.pth"),
        Path("/home/sanghwon/capstone/runs/maskrcnn_rock/best.pth"),
        Path("/home/sanghwon/capstone/runs/maskrcnn_rock/last.pth"),
    ]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        print(f"[WARN] requested weights not found: {path}", flush=True)
        print(f"[WARN] using existing checkpoint: {existing[0]}", flush=True)
        return existing[0]

    print(f"[ERROR] weights file not found: {path}", flush=True)
    print("[ERROR] checked candidate paths:", flush=True)
    for candidate in candidates:
        print(f"[ERROR]   {candidate}", flush=True)
    raise SystemExit(1)


def load_checkpoint(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_image_eval(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], conf_thres: float) -> ImageEval:
    scores_all = pred["scores"].detach().cpu().numpy().astype(np.float32)
    labels_all = pred["labels"].detach().cpu().numpy().astype(np.int64)
    keep = (labels_all == 1) & (scores_all >= conf_thres)

    pred_boxes = pred["boxes"].detach().cpu().numpy().astype(np.float32)[keep]
    pred_labels = labels_all[keep]
    scores = scores_all[keep]
    pred_masks = (pred["masks"].detach().cpu().numpy()[keep, 0] >= 0.5).astype(np.uint8)

    gt_boxes = target["boxes"].detach().cpu().numpy().astype(np.float32)
    gt_labels = target["labels"].detach().cpu().numpy().astype(np.int64)
    gt_masks = target["masks"].detach().cpu().numpy().astype(np.uint8)

    order = np.argsort(-scores)
    scores = scores[order]
    pred_labels = pred_labels[order]
    pred_boxes = pred_boxes[order]
    pred_masks = pred_masks[order]

    return ImageEval(
        scores=scores,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        box_iou=box_iou_matrix(pred_boxes, gt_boxes),
        mask_iou=mask_iou_matrix(pred_masks, gt_masks),
        gt_count=len(gt_labels),
    )


def compute_curve(image_evals: list[ImageEval], thresholds: np.ndarray, iou_thres: float, metric: str) -> dict[str, np.ndarray]:
    precision = []
    recall = []
    f1 = []
    tp_values = []
    fp_values = []
    fn_values = []

    for threshold in thresholds:
        tp = fp = fn = 0
        for item in image_evals:
            counts = match_counts(item, threshold, iou_thres, metric)
            tp += counts[0]
            fp += counts[1]
            fn += counts[2]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        score = 2.0 * p * r / (p + r) if (p + r) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
        tp_values.append(tp)
        fp_values.append(fp)
        fn_values.append(fn)

    return {
        "thresholds": thresholds,
        "precision": np.asarray(precision, dtype=np.float32),
        "recall": np.asarray(recall, dtype=np.float32),
        "f1": np.asarray(f1, dtype=np.float32),
        "tp": np.asarray(tp_values, dtype=np.int64),
        "fp": np.asarray(fp_values, dtype=np.int64),
        "fn": np.asarray(fn_values, dtype=np.int64),
    }


def match_counts(item: ImageEval, conf_thres: float, iou_thres: float, metric: str) -> tuple[int, int, int]:
    scores_keep = np.where(item.scores >= conf_thres)[0]
    ious = item.box_iou if metric == "box" else item.mask_iou
    matched_gt: set[int] = set()
    tp = fp = 0

    for pred_idx in scores_keep:
        if item.gt_count == 0:
            fp += 1
            continue
        row = ious[pred_idx]
        best_gt = -1
        best_iou = -1.0
        for gt_idx, value in enumerate(row):
            if gt_idx in matched_gt:
                continue
            if value > best_iou:
                best_iou = float(value)
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_thres:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = item.gt_count - len(matched_gt)
    return tp, fp, fn


def confusion_matrix_from_evals(image_evals: list[ImageEval], conf_thres: float, iou_thres: float, metric: str) -> np.ndarray:
    tp = fp = fn = 0
    for item in image_evals:
        counts = match_counts(item, conf_thres, iou_thres, metric)
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
    return np.asarray([[tp, fp], [fn, 0]], dtype=np.float32)


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def mask_iou_matrix(masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
    if len(masks_a) == 0 or len(masks_b) == 0:
        return np.zeros((len(masks_a), len(masks_b)), dtype=np.float32)
    output = np.zeros((len(masks_a), len(masks_b)), dtype=np.float32)
    masks_a_bool = masks_a.astype(bool)
    masks_b_bool = masks_b.astype(bool)
    for pred_idx, pred_mask in enumerate(masks_a_bool):
        for gt_idx, gt_mask in enumerate(masks_b_bool):
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            output[pred_idx, gt_idx] = float(inter / union) if union else 0.0
    return output


def best_curve_point(curve: dict[str, np.ndarray]) -> dict[str, float]:
    best_idx = int(np.argmax(curve["f1"])) if len(curve["f1"]) else 0
    return {
        "conf": float(curve["thresholds"][best_idx]),
        "precision": float(curve["precision"][best_idx]),
        "recall": float(curve["recall"][best_idx]),
        "f1": float(curve["f1"][best_idx]),
    }


def compute_ap50(recall: np.ndarray, precision: np.ndarray) -> float:
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    unique_recall, unique_indices = np.unique(recall_sorted, return_index=True)
    precision_unique = np.maximum.reduceat(precision_sorted, unique_indices)
    mrec = np.concatenate(([0.0], unique_recall, [1.0]))
    mpre = np.concatenate(([0.0], precision_unique, [0.0]))
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    return float(np.trapz(mpre, mrec))


def save_curve_plots(output_dir: Path, prefix: str, curve: dict[str, np.ndarray], ap50: float, best: dict[str, float]) -> None:
    plot_line(
        output_dir / f"{prefix}F1_curve.png",
        curve["thresholds"],
        curve["f1"],
        "Confidence",
        "F1",
        "F1-Confidence Curve",
        f"all classes best F1 {best['f1']:.3f} at {best['conf']:.2f}",
    )
    plot_line(
        output_dir / f"{prefix}P_curve.png",
        curve["thresholds"],
        curve["precision"],
        "Confidence",
        "Precision",
        "Precision-Confidence Curve",
        "rock",
    )
    plot_line(
        output_dir / f"{prefix}R_curve.png",
        curve["thresholds"],
        curve["recall"],
        "Confidence",
        "Recall",
        "Recall-Confidence Curve",
        "rock",
    )
    plot_line(
        output_dir / f"{prefix}PR_curve.png",
        curve["recall"],
        curve["precision"],
        "Recall",
        "Precision",
        "Precision-Recall Curve",
        f"rock AP@0.5 {ap50:.3f}",
    )


def plot_line(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, label: str) -> None:
    plt.figure(figsize=(8, 6), facecolor="white")
    plt.plot(x, y, color=CURVE_COLOR, linewidth=3, label=label)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVE] {path.name}", flush=True)


def save_confusion_matrix(path: Path, matrix: np.ndarray, normalized: bool) -> None:
    values = matrix.copy()
    if normalized:
        col_sums = values.sum(axis=0, keepdims=True)
        values = np.divide(values, col_sums, out=np.zeros_like(values), where=col_sums > 0)

    labels = ["rock", "background"]
    plt.figure(figsize=(6, 5), facecolor="white")
    plt.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(2), labels)
    plt.yticks(range(2), labels)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Confusion Matrix Normalized" if normalized else "Confusion Matrix")
    for row in range(2):
        for col in range(2):
            text = f"{values[row, col]:.2f}" if normalized else f"{int(values[row, col])}"
            plt.text(col, row, text, ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVE] {path.name}", flush=True)


def save_labels_plot(json_path: Path, path: Path, max_images: int | None = None) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images = sorted(coco["images"], key=lambda item: item["id"])
    if max_images:
        image_ids = {item["id"] for item in images[:max_images]}
    else:
        image_ids = {item["id"] for item in images}
    image_size = {item["id"]: (float(item["width"]), float(item["height"])) for item in images}

    boxes = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") not in image_ids:
            continue
        width, height = image_size[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if width <= 0 or height <= 0:
            continue
        boxes.append([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height])
    boxes_array = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor="white")
    axes[0, 0].bar(CLASS_NAMES, [len(boxes_array)], color=CURVE_COLOR)
    axes[0, 0].set_title("Class Instance Count")
    axes[0, 0].set_ylabel("Instances")

    axes[0, 1].set_title("Normalized BBox Overlay")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(1, 0)
    axes[0, 1].set_aspect("equal")
    for xc, yc, bw, bh in boxes_array[:1000]:
        rect = plt.Rectangle((xc - bw / 2, yc - bh / 2), bw, bh, fill=False, color=CURVE_COLOR, alpha=0.08)
        axes[0, 1].add_patch(rect)

    axes[1, 0].hist2d(boxes_array[:, 0] if len(boxes_array) else [], boxes_array[:, 1] if len(boxes_array) else [], bins=40, range=[[0, 1], [0, 1]], cmap="Blues")
    axes[1, 0].set_title("X-Y Center Density")
    axes[1, 0].set_xlabel("x center")
    axes[1, 0].set_ylabel("y center")

    axes[1, 1].hist2d(boxes_array[:, 2] if len(boxes_array) else [], boxes_array[:, 3] if len(boxes_array) else [], bins=40, range=[[0, 1], [0, 1]], cmap="Blues")
    axes[1, 1].set_title("Width-Height Density")
    axes[1, 1].set_xlabel("width")
    axes[1, 1].set_ylabel("height")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {path.name}", flush=True)


def save_results(output_dir: Path, summary: dict[str, float], checkpoint: dict[str, Any]) -> None:
    epoch = int(checkpoint.get("epoch", 1)) if isinstance(checkpoint, dict) else 1
    row = {
        "epoch": epoch,
        "box_ap50": summary["box_ap50"],
        "mask_ap50": summary["mask_ap50"],
        "box_precision": summary["box_precision_best"],
        "box_recall": summary["box_recall_best"],
        "box_f1": summary["box_f1_best"],
        "mask_precision": summary["mask_precision_best"],
        "mask_recall": summary["mask_recall_best"],
        "mask_f1": summary["mask_f1_best"],
        "inference_ms_per_image": summary["inference_ms_per_image"],
    }

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"[SAVE] {csv_path.name}", flush=True)

    labels = [
        "box AP50",
        "mask AP50",
        "box F1",
        "mask F1",
        "box conf",
        "mask conf",
        "precision",
        "recall",
        "infer ms",
        "total s",
    ]
    values = [
        summary["box_ap50"],
        summary["mask_ap50"],
        summary["box_f1_best"],
        summary["mask_f1_best"],
        summary["box_best_conf"],
        summary["mask_best_conf"],
        summary["box_precision_best"],
        summary["box_recall_best"],
        summary["inference_ms_per_image"],
        summary["total_evaluation_time_s"],
    ]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), facecolor="white")
    for ax, label, value in zip(axes.ravel(), labels, values):
        ax.plot([epoch], [value], marker="o", color=CURVE_COLOR, linewidth=3)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        if "ms" not in label and "total" not in label:
            ax.set_ylim(0, 1)
    fig.tight_layout()
    png_path = output_dir / "results.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {png_path.name}", flush=True)


def compare_yolo_predictions(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"yolo_pred_json={path}", flush=True)
    print(
        "Loaded YOLO predictions for side-by-side comparison. "
        "Expected schema: image_id/file_name with masks or polygons; convert to binary masks before direct IoU.",
        flush=True,
    )
    print(f"yolo_records={len(data) if isinstance(data, list) else len(data.keys())}", flush=True)


def format_seconds(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


if __name__ == "__main__":
    main()
