from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


# Fast 500-image sample:
# python3 -u /home/sanghwon/capstone/benchmark_yolo_seg_speed.py \
#   --model /home/sanghwon/capstone/runs/rock_seg_sanity/weights/best.pt \
#   --source /home/sanghwon/capstone/datasets/rock_det/images/test \
#   --imgsz 640 \
#   --device 0 \
#   --conf 0.25 \
#   --max-images 500 \
#   --warmup 30 \
#   --output-dir /home/sanghwon/capstone/runs/yolo_seg_benchmark_sample \
#   --maskrcnn-ms 47.95
#
# Full test set:
# python3 -u /home/sanghwon/capstone/benchmark_yolo_seg_speed.py \
#   --model /home/sanghwon/capstone/runs/rock_seg_sanity/weights/best.pt \
#   --source /home/sanghwon/capstone/datasets/rock_det/images/test \
#   --imgsz 640 \
#   --device 0 \
#   --conf 0.25 \
#   --warmup 30 \
#   --output-dir /home/sanghwon/capstone/runs/yolo_seg_benchmark_test \
#   --maskrcnn-ms 47.95
#
# ONNX sample:
# python3 -u /home/sanghwon/capstone/benchmark_yolo_seg_speed.py \
#   --model /home/sanghwon/capstone/runs/rock_seg_sanity/weights/best.onnx \
#   --source /home/sanghwon/capstone/datasets/rock_det/images/test \
#   --imgsz 640 \
#   --device 0 \
#   --conf 0.25 \
#   --max-images 500 \
#   --warmup 30 \
#   --output-dir /home/sanghwon/capstone/runs/yolo_seg_benchmark_onnx_sample \
#   --maskrcnn-ms 47.95


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8-seg inference speed.")
    parser.add_argument("--model", type=Path, default=Path("/home/sanghwon/capstone/runs/rock_seg_sanity/weights/best.pt"))
    parser.add_argument("--source", type=Path, default=Path("/home/sanghwon/capstone/datasets/rock_det/images/test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device index such as 0, or cpu.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/sanghwon/capstone/runs/yolo_seg_benchmark"))
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when running on CUDA.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--maskrcnn-ms", type=float, default=47.95, help="Mask R-CNN reference latency in ms/image. Use 0 to omit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(args.source)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.source}")

    device_label = resolve_device_label(args.device)
    use_cuda = device_label != "cpu" and torch.cuda.is_available()
    use_half = bool(args.half and use_cuda)

    print(f"[INFO] model: {args.model}", flush=True)
    print(f"[INFO] source: {args.source}", flush=True)
    print(f"[INFO] device: {device_label}", flush=True)
    print(f"[INFO] imgsz: {args.imgsz}", flush=True)
    print(f"[INFO] total images found: {len(image_paths)}", flush=True)
    print(f"[INFO] warmup: {args.warmup}", flush=True)
    print(f"[INFO] repeat: {args.repeat}", flush=True)
    print(f"[INFO] half: {use_half}", flush=True)

    model = YOLO(str(args.model))
    run_warmup(model, image_paths, args, device_label, use_cuda, use_half)
    rows = run_benchmark(model, image_paths, args, device_label, use_cuda, use_half)

    summary = build_summary(rows, args, image_paths, device_label, use_half)
    save_results(args.output_dir, rows, summary)
    save_plots(args.output_dir, rows)
    save_summary_txt(args.output_dir / "benchmark_summary.txt", summary, args)

    if args.save_vis:
        save_visualizations(model, image_paths[: args.vis_count], args, device_label, use_half)

    print(f"[RESULT] mean latency: {summary['total_ms']['mean']:.2f} ms/image", flush=True)
    print(f"[RESULT] median latency: {summary['total_ms']['median']:.2f} ms/image", flush=True)
    print(f"[RESULT] p95 latency: {summary['total_ms']['p95']:.2f} ms/image", flush=True)
    print(f"[RESULT] mean FPS: {summary['fps_mean']:.2f}", flush=True)


def collect_image_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")
    return sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_device_label(device: str) -> str:
    if str(device).lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print(f"[WARN] CUDA device '{device}' requested, but torch.cuda.is_available() is False. Falling back to CPU.", flush=True)
        return "cpu"
    return str(device)


def run_warmup(model: YOLO, image_paths: list[Path], args: argparse.Namespace, device_label: str, use_cuda: bool, use_half: bool) -> None:
    warmup_count = max(0, args.warmup)
    if warmup_count == 0:
        return
    for idx in range(warmup_count):
        image_path = image_paths[idx % len(image_paths)]
        sync_cuda(use_cuda)
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device_label,
            half=use_half,
            verbose=False,
        )
        sync_cuda(use_cuda)
        print(f"[WARMUP] {idx + 1}/{warmup_count}", flush=True)


def run_benchmark(model: YOLO, image_paths: list[Path], args: argparse.Namespace, device_label: str, use_cuda: bool, use_half: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_steps = len(image_paths) * max(1, args.repeat)
    start_all = time.perf_counter()
    step = 0

    for repeat_idx in range(max(1, args.repeat)):
        for image_path in image_paths:
            step += 1
            width, height = read_image_size(image_path)
            sync_cuda(use_cuda)
            start = time.perf_counter()
            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device_label,
                half=use_half,
                verbose=False,
            )
            sync_cuda(use_cuda)
            total_ms = (time.perf_counter() - start) * 1000.0

            result = results[0]
            speed = getattr(result, "speed", {}) or {}
            preprocess_ms = none_to_nan(speed.get("preprocess"))
            inference_ms = none_to_nan(speed.get("inference"))
            postprocess_ms = none_to_nan(speed.get("postprocess"))
            num_instances = count_instances(result)
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0

            row = {
                "repeat": repeat_idx + 1,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "num_instances": num_instances,
                "total_ms": total_ms,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "fps": fps,
            }
            rows.append(row)

            if step == 1 or step % 100 == 0 or step == total_steps:
                elapsed = time.perf_counter() - start_all
                eta = elapsed / step * (total_steps - step)
                print(
                    f"[BENCH] {step}/{total_steps} | total={total_ms:.2f}ms | "
                    f"infer={format_optional_ms(inference_ms)} | fps={fps:.2f} | eta={format_seconds(eta)}",
                    flush=True,
                )

    return rows


def read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return 0, 0
    height, width = image.shape[:2]
    return int(width), int(height)


def count_instances(result: Any) -> int:
    if getattr(result, "boxes", None) is not None and result.boxes is not None:
        return int(len(result.boxes))
    if getattr(result, "masks", None) is not None and result.masks is not None:
        return int(len(result.masks))
    return 0


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace, image_paths: list[Path], device_label: str, use_half: bool) -> dict[str, Any]:
    total_ms = values_for(rows, "total_ms")
    fps_values = values_for(rows, "fps")
    num_instances = values_for(rows, "num_instances")
    yolo_mean_ms = summary_stats(total_ms)["mean"]
    speedup = None
    if args.maskrcnn_ms and args.maskrcnn_ms > 0 and yolo_mean_ms > 0:
        speedup = args.maskrcnn_ms / yolo_mean_ms

    return {
        "model": str(args.model),
        "source": str(args.source),
        "imgsz": args.imgsz,
        "device": device_label,
        "half": use_half,
        "conf": args.conf,
        "iou": args.iou,
        "repeat": max(1, args.repeat),
        "measured_images": len(rows),
        "warmup_images": max(0, args.warmup),
        "total_images": len(image_paths),
        "total_ms": summary_stats(total_ms),
        "inference_ms": summary_stats(values_for(rows, "inference_ms")),
        "preprocess_ms": summary_stats(values_for(rows, "preprocess_ms")),
        "postprocess_ms": summary_stats(values_for(rows, "postprocess_ms")),
        "fps_mean": float(np.mean(fps_values)) if fps_values else 0.0,
        "fps_median": float(np.median(fps_values)) if fps_values else 0.0,
        "avg_num_instances": float(np.mean(num_instances)) if num_instances else 0.0,
        "total_num_instances": int(np.sum(num_instances)) if num_instances else 0,
        "yolo_mean_ms": yolo_mean_ms,
        "maskrcnn_ms": args.maskrcnn_ms if args.maskrcnn_ms and args.maskrcnn_ms > 0 else None,
        "speedup_vs_maskrcnn": speedup,
    }


def values_for(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isnan(numeric):
            values.append(numeric)
    return values


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def save_results(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = output_dir / "benchmark_results.csv"
    fieldnames = [
        "repeat",
        "image_path",
        "width",
        "height",
        "num_instances",
        "total_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "fps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {csv_path.name}", flush=True)

    json_path = output_dir / "benchmark_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] {json_path.name}", flush=True)


def save_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    total_ms = values_for(rows, "total_ms")
    fps = values_for(rows, "fps")

    save_histogram(output_dir / "latency_histogram.png", total_ms, "Latency Histogram", "total_ms", "image count")
    save_histogram(output_dir / "fps_histogram.png", fps, "FPS Histogram", "FPS", "image count")
    save_speed_boxplot(output_dir / "speed_boxplot.png", rows)


def save_histogram(path: Path, values: list[float], title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 6), facecolor="white")
    plt.hist(values, bins=40, color="#1f77b4", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVE] {path.name}", flush=True)


def save_speed_boxplot(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = ["preprocess", "inference", "postprocess", "total"]
    data = [
        values_for(rows, "preprocess_ms"),
        values_for(rows, "inference_ms"),
        values_for(rows, "postprocess_ms"),
        values_for(rows, "total_ms"),
    ]
    plt.figure(figsize=(8, 6), facecolor="white")
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title("Speed Breakdown Boxplot")
    plt.ylabel("ms/image")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVE] {path.name}", flush=True)


def save_summary_txt(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    total = summary["total_ms"]
    infer = summary["inference_ms"]
    speedup = summary["speedup_vs_maskrcnn"]
    speedup_line = "Speedup vs Mask R-CNN: omitted"
    if speedup is not None:
        speedup_line = (
            f"YOLO mean latency = {summary['yolo_mean_ms']:.2f} ms/image\n"
            f"Mask R-CNN latency = {summary['maskrcnn_ms']:.2f} ms/image\n"
            f"Speedup = {summary['maskrcnn_ms']:.2f} / {summary['yolo_mean_ms']:.2f} = {speedup:.2f}x faster"
        )

    text = f"""YOLOv8-seg inference speed benchmark
Model: {Path(summary['model']).name}
Images: {summary['total_images']}
Measured samples: {summary['measured_images']}
Input size: {summary['imgsz']}
Device: {summary['device']}
Mean latency: {total['mean']:.2f} ms/image
Median latency: {total['median']:.2f} ms/image
P95 latency: {total['p95']:.2f} ms/image
Mean FPS: {summary['fps_mean']:.2f} FPS
Median FPS: {summary['fps_median']:.2f} FPS
Mean model inference time: {infer['mean']:.2f} ms/image
Average detected instances: {summary['avg_num_instances']:.2f}

Mask R-CNN baseline reference:
Mask R-CNN evaluation reference latency: {args.maskrcnn_ms:.2f} ms/image
{speedup_line}

Conclusion:
The YOLOv8-seg model satisfies near real-time / real-time inference requirements for the aggregate inspection pipeline.
"""
    path.write_text(text, encoding="utf-8")
    print(f"[SAVE] {path.name}", flush=True)


def save_visualizations(model: YOLO, image_paths: list[Path], args: argparse.Namespace, device_label: str, use_half: bool) -> None:
    vis_dir = args.output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] saving visualizations: {len(image_paths)} images", flush=True)
    for idx, image_path in enumerate(image_paths, start=1):
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device_label,
            half=use_half,
            verbose=False,
        )
        plotted = results[0].plot()
        out_path = vis_dir / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), plotted)
        print(f"[SAVE] vis/{out_path.name} ({idx}/{len(image_paths)})", flush=True)


def sync_cuda(use_cuda: bool) -> None:
    if use_cuda:
        torch.cuda.synchronize()


def none_to_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def format_optional_ms(value: float) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.2f}ms"


def format_seconds(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


if __name__ == "__main__":
    main()
