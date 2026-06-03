from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation labels to COCO polygons.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/sanghwon/capstone/datasets/rock_det"))
    parser.add_argument("--output-dir", type=Path, default=Path("maskrcnn_pipeline/dataset"))
    parser.add_argument("--labels-dir-name", default="labels")
    parser.add_argument("--bbox-labels-dir-name", default="labels_det")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--simplify-epsilon", type=float, default=1.0, help="Polygon simplification in pixels.")
    parser.add_argument("--max-points", type=int, default=512, help="Maximum polygon vertices per instance.")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        coco = convert_split(
            args.dataset_root,
            args.labels_dir_name,
            args.bbox_labels_dir_name,
            split,
            simplify_epsilon=args.simplify_epsilon,
            max_points=args.max_points,
            progress_every=args.progress_every,
        )
        output_path = args.output_dir / f"{split}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(
            f"Wrote {output_path}: "
            f"{len(coco['images'])} images, {len(coco['annotations'])} annotations"
        )


def convert_split(
    dataset_root: Path,
    labels_dir_name: str,
    bbox_labels_dir_name: str,
    split: str,
    simplify_epsilon: float = 1.0,
    max_points: int = 512,
    progress_every: int = 5000,
) -> dict:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / labels_dir_name / split
    bbox_label_dir = dataset_root / bbox_labels_dir_name / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image split directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label split directory: {label_dir}")

    images = []
    annotations = []
    ann_id = 1

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    for image_id, image_path in enumerate(image_paths, start=1):
        if progress_every > 0 and image_id % progress_every == 0:
            print(f"{split}: processed {image_id}/{len(image_paths)} images", flush=True)
        size = read_image_size(image_path)
        if size is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        width, height = size
        rel_file = image_path.relative_to(dataset_root).as_posix()
        images.append({"id": image_id, "file_name": rel_file, "width": width, "height": height})

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        bbox_lines = []
        bbox_label_path = bbox_label_dir / f"{image_path.stem}.txt"
        if bbox_label_path.exists():
            bbox_lines = bbox_label_path.read_text(encoding="utf-8").splitlines()

        for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            bbox_line = bbox_lines[line_idx] if line_idx < len(bbox_lines) else None
            parsed = yolo_line_to_annotation(line, width, height, simplify_epsilon, max_points, bbox_line)
            if parsed is None:
                continue
            bbox, segmentation, area = parsed
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "rock", "supercategory": "aggregate"}],
    }


def yolo_line_to_annotation(
    line: str,
    width: int,
    height: int,
    simplify_epsilon: float = 1.0,
    max_points: int = 512,
    bbox_line: str | None = None,
):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    if class_id != 0:
        return None

    coords_text = parts[1:]
    coords = decimated_coords(coords_text, max_points)
    if len(coords) == 4:
        polygon = bbox_yolo_to_polygon(coords, width, height)
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        polygon = normalized_polygon_to_pixels(coords, width, height)
    else:
        return None

    polygon = clean_polygon(polygon, width, height, simplify_epsilon)
    if len(polygon) < 6:
        return None

    bbox = bbox_from_yolo_line(bbox_line, width, height) if bbox_line else None
    if bbox is None:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        bbox = [float(x_min), float(y_min), float(max(0.0, x_max - x_min)), float(max(0.0, y_max - y_min))]
    box_w = bbox[2]
    box_h = bbox[3]
    if box_w <= 1.0 or box_h <= 1.0:
        return None

    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    area = float(abs(cv2.contourArea(points)))
    if area <= 1.0:
        area = box_w * box_h
    return bbox, [polygon], area


def decimated_coords(coords_text: list[str], max_points: int) -> list[float]:
    if len(coords_text) <= 4 or max_points <= 0:
        return [float(value) for value in coords_text]
    point_count = len(coords_text) // 2
    if point_count <= max_points:
        return [float(value) for value in coords_text]

    step = max(1, point_count // max_points)
    sampled = []
    for point_idx in range(0, point_count, step):
        sampled.extend([float(coords_text[point_idx * 2]), float(coords_text[point_idx * 2 + 1])])
        if len(sampled) >= max_points * 2:
            break
    return sampled


def bbox_from_yolo_line(line: str | None, width: int, height: int) -> list[float] | None:
    if not line:
        return None
    parts = line.strip().split()
    if len(parts) < 5 or int(float(parts[0])) != 0:
        return None
    x_center, y_center, box_w, box_h = [float(value) for value in parts[1:5]]
    x1 = float(np.clip((x_center - box_w / 2.0) * width, 0, width - 1))
    y1 = float(np.clip((y_center - box_h / 2.0) * height, 0, height - 1))
    x2 = float(np.clip((x_center + box_w / 2.0) * width, 0, width - 1))
    y2 = float(np.clip((y_center + box_h / 2.0) * height, 0, height - 1))
    return [x1, y1, float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def normalized_polygon_to_pixels(coords: list[float], width: int, height: int) -> list[float]:
    polygon = []
    for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
        polygon.extend([x_norm * width, y_norm * height])
    return polygon


def bbox_yolo_to_polygon(coords: list[float], width: int, height: int) -> list[float]:
    x_center, y_center, box_w, box_h = coords
    x1 = (x_center - box_w / 2.0) * width
    y1 = (y_center - box_h / 2.0) * height
    x2 = (x_center + box_w / 2.0) * width
    y2 = (y_center + box_h / 2.0) * height
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def clean_polygon(polygon: list[float], width: int, height: int, simplify_epsilon: float) -> list[float]:
    cleaned_points = []
    for x, y in zip(polygon[0::2], polygon[1::2]):
        cleaned_points.append([float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))])

    points = np.asarray(cleaned_points, dtype=np.float32)
    if simplify_epsilon > 0.0 and len(points) >= 4:
        approx = cv2.approxPolyDP(points.reshape(-1, 1, 2), simplify_epsilon, True)
        points = approx.reshape(-1, 2)
    return points.reshape(-1).astype(float).tolist()


def read_image_size(image_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height


if __name__ == "__main__":
    main()
