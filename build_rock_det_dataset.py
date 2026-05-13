#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build YOLOv8 detection dataset from aggregate rock polygon JSON labels.

Expected source layout:
  /home/capstone/raw_data/images/*.png
  /home/capstone/raw_data/labels_json/*.json

Output layout:
  /home/capstone/datasets/rock_det/
    images/train|val|test
    labels/train|val|test
    rock_det.yaml
    build_summary.json

Class policy:
  - Single class detection dataset
  - class_id 0 = rock
  - JSON rock_type is ignored for now
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create YOLOv8 detection dataset from polygon JSON labels."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/capstone"),
        help="Project root directory. Default: /home/capstone",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Source image directory. Default: <project-root>/raw_data/images",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Source JSON label directory. Default: <project-root>/raw_data/labels_json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output YOLO dataset directory. Default: <project-root>/datasets/rock_det",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-box-px",
        type=float,
        default=2.0,
        help="Skip boxes smaller than this width or height in pixels.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before creating dataset.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep matched images even if zero valid boxes are created.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_images(image_dir: Path) -> tuple[dict[str, Path], list[str]]:
    image_map: dict[str, Path] = {}
    warnings: list[str] = []

    for path in sorted(image_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        key = path.stem
        if key in image_map:
            warnings.append(
                f"duplicate image stem '{key}': '{image_map[key]}' vs '{path}'. "
                f"Keeping first one."
            )
            continue

        image_map[key] = path

    return image_map, warnings


def collect_jsons(json_dir: Path) -> tuple[dict[str, Path], list[str]]:
    json_map: dict[str, Path] = {}
    warnings: list[str] = []

    for path in sorted(json_dir.rglob("*.json")):
        try:
            data = read_json(path)
        except Exception as e:
            warnings.append(f"failed to read json '{path}': {e}")
            continue

        # Prefer object_id from JSON. Fall back to filename stem.
        key = str(data.get("object_id") or path.stem)

        if key in json_map:
            warnings.append(
                f"duplicate json id '{key}': '{json_map[key]}' vs '{path}'. "
                f"Keeping first one."
            )
            continue

        json_map[key] = path

    return json_map, warnings


def polygon_to_yolo_bbox(
    points: list[dict[str, Any]],
    image_width: float,
    image_height: float,
    min_box_px: float,
) -> tuple[float, float, float, float] | None:
    coords: list[tuple[float, float]] = []

    for p in points:
        try:
            x = float(p["x"])
            y = float(p["y"])
        except (KeyError, TypeError, ValueError):
            continue
        coords.append((x, y))

    if len(coords) < 3:
        return None

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    x_min = max(0.0, min(float(image_width), min(xs)))
    y_min = max(0.0, min(float(image_height), min(ys)))
    x_max = max(0.0, min(float(image_width), max(xs)))
    y_max = max(0.0, min(float(image_height), max(ys)))

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w < min_box_px or box_h < min_box_px:
        return None

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    xc = x_center / image_width
    yc = y_center / image_height
    bw = box_w / image_width
    bh = box_h / image_height

    # Final safety clamp
    xc = min(1.0, max(0.0, xc))
    yc = min(1.0, max(0.0, yc))
    bw = min(1.0, max(0.0, bw))
    bh = min(1.0, max(0.0, bh))

    if bw <= 0.0 or bh <= 0.0:
        return None

    return xc, yc, bw, bh


def json_to_yolo_lines(
    json_path: Path,
    min_box_px: float,
) -> tuple[list[str], dict[str, int]]:
    data = read_json(json_path)

    try:
        image_width = float(data["width"])
        image_height = float(data["height"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"missing or invalid width/height in {json_path}: {e}") from e

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size in {json_path}: {image_width}x{image_height}")

    vertices = data.get("vertices", [])
    if not isinstance(vertices, list):
        raise ValueError(f"vertices must be list in {json_path}")

    lines: list[str] = []
    skipped_objects = 0

    for obj in vertices:
        if not isinstance(obj, dict):
            skipped_objects += 1
            continue

        points = obj.get("points", [])
        if not isinstance(points, list):
            skipped_objects += 1
            continue

        bbox = polygon_to_yolo_bbox(
            points=points,
            image_width=image_width,
            image_height=image_height,
            min_box_px=min_box_px,
        )

        if bbox is None:
            skipped_objects += 1
            continue

        # Single class: rock
        class_id = 0
        xc, yc, bw, bh = bbox
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    stats = {
        "total_vertices": len(vertices),
        "valid_boxes": len(lines),
        "skipped_objects": skipped_objects,
    }
    return lines, stats


def split_ids(
    ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must satisfy: train > 0, val >= 0, train + val < 1")

    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def write_yaml(output_dir: Path) -> None:
    yaml_text = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: rock
"""
    (output_dir / "rock_det.yaml").write_text(yaml_text, encoding="utf-8")


def ensure_output_dir(output_dir: Path, clean: bool) -> None:
    if output_dir.exists():
        if clean:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Use --clean if you want to overwrite it."
            )

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    project_root: Path = args.project_root
    image_dir: Path = args.image_dir or (project_root / "raw_data" / "images")
    json_dir: Path = args.json_dir or (project_root / "raw_data" / "labels_json")
    output_dir: Path = args.output_dir or (project_root / "datasets" / "rock_det")

    image_dir = image_dir.expanduser().resolve()
    json_dir = json_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    image_map, image_warnings = collect_images(image_dir)
    json_map, json_warnings = collect_jsons(json_dir)

    matched_ids = sorted(set(image_map.keys()) & set(json_map.keys()))
    image_only = sorted(set(image_map.keys()) - set(json_map.keys()))
    json_only = sorted(set(json_map.keys()) - set(image_map.keys()))

    if not matched_ids:
        raise RuntimeError(
            "No matched image-json pairs found. "
            "Check that PNG/JPG stems match JSON object_id or JSON filename stem."
        )

    ensure_output_dir(output_dir, clean=args.clean)

    splits = split_ids(
        ids=matched_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    summary: dict[str, Any] = {
        "source": {
            "image_dir": image_dir.as_posix(),
            "json_dir": json_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
        },
        "config": {
            "class_mode": "single",
            "class_id": 0,
            "class_name": "rock",
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
            "seed": args.seed,
            "min_box_px": args.min_box_px,
            "keep_empty": args.keep_empty,
        },
        "input_counts": {
            "images": len(image_map),
            "jsons": len(json_map),
            "matched": len(matched_ids),
            "images_without_json": len(image_only),
            "jsons_without_image": len(json_only),
        },
        "split_counts": {},
        "warnings": image_warnings + json_warnings,
        "examples": {
            "images_without_json_first_20": image_only[:20],
            "jsons_without_image_first_20": json_only[:20],
        },
    }

    total_boxes = 0
    total_skipped_objects = 0
    skipped_files: list[str] = []
    split_counter = Counter()

    for split, ids in splits.items():
        copied_images = 0
        written_labels = 0
        split_boxes = 0
        split_skipped_objects = 0

        for file_id in ids:
            src_image = image_map[file_id]
            src_json = json_map[file_id]

            try:
                yolo_lines, obj_stats = json_to_yolo_lines(
                    json_path=src_json,
                    min_box_px=args.min_box_px,
                )
            except Exception as e:
                summary["warnings"].append(f"failed to convert '{src_json}': {e}")
                skipped_files.append(file_id)
                continue

            if not yolo_lines and not args.keep_empty:
                summary["warnings"].append(
                    f"skipped '{file_id}' because no valid boxes were created"
                )
                skipped_files.append(file_id)
                continue

            dst_image = output_dir / "images" / split / src_image.name
            dst_label = output_dir / "labels" / split / f"{src_image.stem}.txt"

            shutil.move(src_image, dst_image)
            dst_label.write_text("\n".join(yolo_lines), encoding="utf-8")

            copied_images += 1
            written_labels += 1
            split_boxes += len(yolo_lines)
            split_skipped_objects += obj_stats["skipped_objects"]

        split_counter[split] = copied_images
        total_boxes += split_boxes
        total_skipped_objects += split_skipped_objects

        summary["split_counts"][split] = {
            "images": copied_images,
            "labels": written_labels,
            "boxes": split_boxes,
            "skipped_objects": split_skipped_objects,
        }

    write_yaml(output_dir)

    summary["output_counts"] = {
        "images": int(sum(split_counter.values())),
        "labels": int(sum(split_counter.values())),
        "boxes": total_boxes,
        "skipped_objects": total_skipped_objects,
        "skipped_files": len(skipped_files),
    }
    summary["examples"]["skipped_files_first_20"] = skipped_files[:20]

    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)

    print("\n[DONE] YOLOv8 detection dataset created")
    print(f"  output_dir : {summary['source']['output_dir']}")
    print(f"  yaml       : {summary['source']['output_dir']}/rock_det.yaml")
    print("\n[INPUT]")
    for k, v in summary["input_counts"].items():
        print(f"  {k:22s}: {v}")

    print("\n[SPLITS]")
    for split in ["train", "val", "test"]:
        s = summary["split_counts"].get(split, {})
        print(
            f"  {split:5s} images={s.get('images', 0):5d} "
            f"labels={s.get('labels', 0):5d} "
            f"boxes={s.get('boxes', 0):6d}"
        )

    print("\n[OUTPUT]")
    for k, v in summary["output_counts"].items():
        print(f"  {k:22s}: {v}")

    if summary["warnings"]:
        print(f"\n[WARNINGS] {len(summary['warnings'])} warning(s).")
        print("  See build_summary.json for details.")
        for warning in summary["warnings"][:10]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
