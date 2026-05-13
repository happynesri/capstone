#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

PROJECT_ROOT = Path("/home/sanghwon/capstone")

DATASET_DIR = PROJECT_ROOT / "datasets" / "rock_det"
IMAGE_SPLIT_DIR = DATASET_DIR / "images"
OUTPUT_LABEL_DIR = DATASET_DIR / "labels_seg"

JSON_DIR = PROJECT_ROOT / "raw_data" / "labels_json"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v, low=0.0, high=1.0):
    return max(low, min(high, v))


def convert_json_to_yolo_seg(json_path, output_txt_path):
    data = load_json(json_path)

    width = float(data["width"])
    height = float(data["height"])

    vertices = data.get("vertices", [])

    lines = []

    for obj in vertices:
        points = obj.get("points", [])

        if len(points) < 3:
            continue

        coords = []

        for p in points:
            try:
                x = float(p["x"])
                y = float(p["y"])
            except Exception:
                continue

            nx = clamp(x / width)
            ny = clamp(y / height)

            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")

        # segmentation은 최소 3점 = x y 3쌍 이상 필요
        if len(coords) < 6:
            continue

        # 단일 클래스 rock = 0
        line = "0 " + " ".join(coords)
        lines.append(line)

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(lines)


def find_json_by_stem(stem):
    json_path = JSON_DIR / f"{stem}.json"
    if json_path.exists():
        return json_path
    return None


def build_labels_seg():
    total_images = 0
    total_labels = 0
    total_objects = 0
    missing_json = []

    for split in ["train", "val", "test"]:
        image_dir = IMAGE_SPLIT_DIR / split
        label_dir = OUTPUT_LABEL_DIR / split

        if not image_dir.exists():
            print(f"[WARN] image split dir not found: {image_dir}")
            continue

        image_paths = []
        for ext in IMAGE_EXTS:
            image_paths.extend(image_dir.glob(f"*{ext}"))

        image_paths = sorted(image_paths)

        print(f"[INFO] split={split}, images={len(image_paths)}")

        for idx, image_path in enumerate(image_paths, start=1):
            stem = image_path.stem
            json_path = find_json_by_stem(stem)

            if json_path is None:
                missing_json.append(stem)
                continue

            output_txt_path = label_dir / f"{stem}.txt"
            object_count = convert_json_to_yolo_seg(json_path, output_txt_path)

            total_images += 1
            total_labels += 1
            total_objects += object_count

            if idx % 1000 == 0:
                print(f"[{split}] {idx}/{len(image_paths)} processed")

    print()
    print("[DONE] YOLO segmentation labels created")
    print(f"labels_seg dir : {OUTPUT_LABEL_DIR}")
    print(f"images matched : {total_images}")
    print(f"labels created : {total_labels}")
    print(f"objects        : {total_objects}")
    print(f"missing json   : {len(missing_json)}")

    if missing_json:
        print("[WARN] first missing json examples:")
        print(missing_json[:20])


if __name__ == "__main__":
    build_labels_seg()