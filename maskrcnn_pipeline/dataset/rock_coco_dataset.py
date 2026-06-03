from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


def collate_fn(batch):
    return tuple(zip(*batch))


class RockCocoDataset(Dataset):
    def __init__(self, json_path: str | Path, image_root: str | Path, normalize: bool = True):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)
        with self.json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = sorted(coco["images"], key=lambda item: item["id"])
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = {
            image["id"]: [] for image in self.images
        }
        for ann in coco.get("annotations", []):
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        if normalize:
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transforms = T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_info = self.images[idx]
        image_path = Path(image_info["file_name"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        anns = self.annotations_by_image.get(image_info["id"], [])
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))
            masks.append(_polygon_to_mask(ann.get("segmentation", []), height, width))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            if masks
            else torch.zeros((0, height, width), dtype=torch.uint8),
            "image_id": torch.tensor([image_info["id"]], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        image = self.transforms(image_rgb)
        return image, target


def _polygon_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask
