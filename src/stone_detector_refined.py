#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate / stone detector for dark tray images.

목표
- 검은 tray 위의 회색 골재를 명확하게 분리
- 그림자 때문에 박스가 커지는 문제 완화
- 내부 음영 때문에 한 골재가 여러 contour로 쪼개지는 문제 완화
- ROI 밖 오검출 제거

실행 예시
python3 aggregate_inspection_cv_complete.py --image /capstone/data_1/101-d-001.png \
  --roi 120,155,1460,1090 --out-dir ./result --debug

현재 사진 기준 추천 시작값
python3 aggregate_inspection_cv_complete.py --image input.png \
  --roi 120,155,1460,1090 \
  --min-delta-l 14 --bg-sigma 55 --morph-close 35 --morph-open 7 \
  --min-area 3500 --min-ring-delta-l 3.5 --merge-distance 45 \
  --out-dir ./result --debug
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class DetectorParams:
    # ROI: x, y, w, h. 카메라 고정이면 tray 내부 영역을 고정하는 것이 가장 중요함.
    roi: tuple[int, int, int, int] | None = None  # 기본값: 130,170,1270,1000 (tray 내부)
    roi_erode_px: int = 10
    roi_margin_px: int = 100  # ROI 경계 근처 reject margin (테두리 artifact 제거)

    # illumination normalization
    median_blur_ksize: int = 5
    bilateral_d: int = 9
    bilateral_sigma_color: float = 55.0
    bilateral_sigma_space: float = 55.0
    bg_sigma: float = 55.0
    min_delta_l: float = 20.0  # delta threshold 강화 (배경 제거)

    # optional absolute lightness helper. 대비 약한 골재 보완용.
    use_otsu_helper: bool = True
    otsu_offset: float = -6.0

    # morphology
    morph_close_px: int = 20  # close kernel 축소 (배경 덩어리 방지)
    morph_open_px: int = 5   # open kernel (노이즈 제거용)
    fill_holes: bool = True

    # contour merge: 내부 그림자로 contour가 찢어진 경우 합침.
    merge_distance: float = 45.0
    merge_max_area_ratio: float = 2.8

    # candidate filtering
    min_area_px: int = 3500
    max_area_frac: float = 0.20
    min_extent: float = 0.22
    min_solidity: float = 0.48
    max_aspect_ratio: float = 4.0
    min_ring_delta_l: float = 3.5
    ring_px: int = 30

    # duplicate suppression
    nms_iou: float = 0.20

    # output
    approx_epsilon_ratio: float = 0.012
    draw_axis: bool = False


def odd_kernel(value: int | float) -> int:
    v = int(round(value))
    if v < 1:
        v = 1
    return v if v % 2 == 1 else v + 1


def ellipse_kernel(size: int | float) -> np.ndarray:
    k = odd_kernel(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None or text.strip().lower() in {"", "none", "full"}:
        return None
    parts = [int(float(p.strip())) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi는 x,y,w,h 형식이어야 합니다. 예: --roi 120,155,1460,1090")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height는 양수여야 합니다.")
    return x, y, w, h


def make_roi_mask(shape: tuple[int, int], roi: tuple[int, int, int, int] | None, erode_px: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if roi is None:
        mask[:, :] = 255
    else:
        x, y, rw, rh = roi
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        mask[y0:y1, x0:x1] = 255

    if erode_px > 0:
        mask = cv2.erode(mask, ellipse_kernel(erode_px * 2 + 1), iterations=1)
    return mask


def fill_binary_holes(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8) * 255
    h, w = binary.shape
    flood = binary.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def local_lightness(image_bgr: np.ndarray, params: DetectorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # median: salt/dust 완화, bilateral: edge 유지하며 표면 노이즈 완화
    L_med = cv2.medianBlur(L, odd_kernel(params.median_blur_ksize))
    L_smooth = cv2.bilateralFilter(
        L_med,
        d=odd_kernel(params.bilateral_d),
        sigmaColor=params.bilateral_sigma_color,
        sigmaSpace=params.bilateral_sigma_space,
    )

    # 큰 스케일 조명 배경 추정
    bg = cv2.GaussianBlur(L_smooth, (0, 0), sigmaX=params.bg_sigma, sigmaY=params.bg_sigma)
    delta = L_smooth.astype(np.int16) - bg.astype(np.int16)
    return L, L_smooth, bg, delta


def otsu_lightness_mask(L_smooth: np.ndarray, roi_mask: np.ndarray, offset: float) -> np.ndarray:
    vals = L_smooth[roi_mask > 0]
    if vals.size < 100:
        return np.zeros_like(L_smooth, np.uint8)

    # ROI 내부 값만으로 Otsu threshold 계산
    _, tmp = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OpenCV가 반환 threshold를 직접 얻기 위해 다시 호출
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(thr) + offset
    mask = ((L_smooth.astype(np.float32) >= threshold) & (roi_mask > 0)).astype(np.uint8) * 255
    return mask


def segment_stones(image_bgr: np.ndarray, params: DetectorParams) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h, w = image_bgr.shape[:2]
    roi_mask = make_roi_mask((h, w), params.roi, params.roi_erode_px)
    L, L_smooth, bg, delta = local_lightness(image_bgr, params)

    # 1차: local background보다 밝은 영역 (delta threshold만 사용 - Otsu helper 제거)
    mask_delta = ((delta >= params.min_delta_l) & (roi_mask > 0)).astype(np.uint8) * 255
    mask_otsu = np.zeros_like(mask_delta)  # Otsu helper 비활성화
    mask = mask_delta

    # morphology: 순서 역순으로 작은 노이즈 먼저 제거
    # 열기로 먼저 작은 노이즈/배경 제거
    if params.morph_open_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(params.morph_open_px), iterations=1)
    # 그 다음 닫기로 골재 내부 그림자/균열 연결
    if params.morph_close_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse_kernel(params.morph_close_px), iterations=1)
    if params.fill_holes:
        mask = fill_binary_holes(mask)

    mask = cv2.bitwise_and(mask, roi_mask)
    
    # 흰색 지그재그 마스크 생성 (debug용)
    white_mask = detect_white_markings(image_bgr, roi_mask)

    debug = {
        "roi_mask": roi_mask,
        "L": L,
        "L_smooth": L_smooth,
        "background_L": bg,
        "delta_L_vis": np.clip(delta + 128, 0, 255).astype(np.uint8),
        "mask_delta": mask_delta,
        "mask_otsu": mask_otsu,
        "mask_final": mask,
        "white_markings": white_mask,
    }
    return mask, debug


def contour_to_mask(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    cv2.drawContours(out, [contour], -1, 255, thickness=-1)
    return out


def contour_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    # 빠른 bbox distance + 실제 점 거리 근사
    x1, y1, w1, h1 = cv2.boundingRect(c1)
    x2, y2, w2, h2 = cv2.boundingRect(c2)
    dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
    dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
    bbox_dist = math.hypot(dx, dy)
    if bbox_dist > 80:
        return bbox_dist

    pts1 = c1.reshape(-1, 2).astype(np.float32)
    pts2 = c2.reshape(-1, 2).astype(np.float32)
    # 너무 많은 점은 샘플링
    if len(pts1) > 80:
        pts1 = pts1[np.linspace(0, len(pts1) - 1, 80).astype(int)]
    if len(pts2) > 80:
        pts2 = pts2[np.linspace(0, len(pts2) - 1, 80).astype(int)]
    diff = pts1[:, None, :] - pts2[None, :, :]
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))


def merge_close_contours(contours: list[np.ndarray], distance_thr: float, max_area_ratio: float) -> list[np.ndarray]:
    if len(contours) <= 1 or distance_thr <= 0:
        return contours

    n = len(contours)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    areas = [max(cv2.contourArea(c), 1.0) for c in contours]

    for i in range(n):
        for j in range(i + 1, n):
            d = contour_distance(contours[i], contours[j])
            if d > distance_thr:
                continue

            combined = np.vstack([contours[i], contours[j]])
            hull = cv2.convexHull(combined)
            hull_area = max(cv2.contourArea(hull), 1.0)
            area_sum = areas[i] + areas[j]

            # 두 contour 사이 빈 공간이 지나치게 크면 별도 골재일 가능성이 높으므로 merge 금지
            if hull_area / area_sum <= max_area_ratio:
                union(i, j)

    groups: dict[int, list[np.ndarray]] = {}
    for idx, c in enumerate(contours):
        groups.setdefault(find(idx), []).append(c)

    merged: list[np.ndarray] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(cv2.convexHull(np.vstack(group)))
    return merged


def rotated_iou(rect_a: tuple, rect_b: tuple) -> float:
    area_a = max(float(rect_a[1][0] * rect_a[1][1]), 1e-6)
    area_b = max(float(rect_b[1][0] * rect_b[1][1]), 1e-6)
    inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if inter_type == cv2.INTERSECT_NONE or inter_pts is None:
        return 0.0
    inter_area = abs(cv2.contourArea(cv2.convexHull(inter_pts)))
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def nms_rotated(candidates: list[dict[str, Any]], iou_thr: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if all(rotated_iou(cand["rect"], prev["rect"]) < iou_thr for prev in kept):
            kept.append(cand)
    return kept


def detect_white_markings(image_bgr: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """흰색 지그재그/테두리 마스크 생성 (현재 비활성화)"""
    # 모든 contour를 제거했으므로 임시로 비활성화
    return np.zeros(image_bgr.shape[:2], dtype=np.uint8)


def extract_candidates(mask: np.ndarray, image_bgr: np.ndarray, params: DetectorParams) -> list[dict[str, Any]]:
    h, w = mask.shape
    roi_mask = make_roi_mask((h, w), params.roi, params.roi_erode_px)
    roi_area = float(np.count_nonzero(roi_mask))
    max_area_px = max(params.min_area_px, params.max_area_frac * roi_area)

    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    white_mask = detect_white_markings(image_bgr, roi_mask)  # 흰색 지그재그 마스크

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= max(100.0, params.min_area_px * 0.25)]
    contours = merge_close_contours(contours, params.merge_distance, params.merge_max_area_ratio)

    candidates: list[dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < params.min_area_px or area > max_area_px:
            continue

        # ROI 경계 근처 contour 제거 (ROI 기준 margin check)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # ROI가 지정된 경우: ROI 좌표 기준으로 margin check
            if params.roi is not None:
                rx, ry, rw, rh = params.roi
                margin = params.roi_margin_px
                if cx < rx + margin or cy < ry + margin or cx > rx + rw - margin or cy > ry + rh - margin:
                    continue
            # ROI가 없는 경우: 전체 이미지 기준 (legacy)
            else:
                margin = 50
                if cx < margin or cy < margin or cx > w - margin or cy > h - margin:
                    continue
        
        # 흰색 지그재그/테두리 오버랩 제거 (선택사항 - 현재 작업 중)
        # contour_mask = contour_to_mask((h, w), contour)
        # white_overlap = np.count_nonzero(cv2.bitwise_and(contour_mask, white_mask))
        # overlap_ratio = white_overlap / max(area, 1.0)
        # if overlap_ratio > 0.05:  # 5% 이상 겹치면 제거
        #     continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw <= 1 or rh <= 1:
            continue

        long_axis = float(max(rw, rh))
        short_axis = float(min(rw, rh))
        aspect = long_axis / max(short_axis, 1e-6)
        if aspect > params.max_aspect_ratio:
            continue

        rect_area = float(rw * rh)
        extent = area / max(rect_area, 1e-6)
        if extent < params.min_extent:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1e-6)
        if solidity < params.min_solidity:
            continue

        obj_mask = contour_to_mask((h, w), contour)
        ring = cv2.dilate(obj_mask, ellipse_kernel(params.ring_px * 2 + 1), iterations=1)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(obj_mask))
        ring = cv2.bitwise_and(ring, roi_mask)

        mean_obj = float(cv2.mean(L, mask=obj_mask)[0])
        mean_ring = float(cv2.mean(L, mask=ring)[0]) if np.count_nonzero(ring) else mean_obj
        ring_delta = mean_obj - mean_ring
        if ring_delta < params.min_ring_delta_l:
            continue

        epsilon = params.approx_epsilon_ratio * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        box = cv2.boxPoints(rect).astype(int)

        score = area * max(ring_delta, 0.01) * max(solidity, 0.01) * max(extent, 0.01)
        candidates.append({
            "rect": rect,
            "box": box,
            "polygon": polygon.reshape(-1, 2),
            "contour": contour,
            "area_px": area,
            "long_axis_px": long_axis,
            "short_axis_px": short_axis,
            "aspect_ratio": aspect,
            "extent": extent,
            "solidity": solidity,
            "mean_l": mean_obj,
            "ring_mean_l": mean_ring,
            "ring_delta_l": ring_delta,
            "score": score,
        })

    kept = nms_rotated(candidates, params.nms_iou)
    return sorted(kept, key=lambda c: (c["rect"][0][1], c["rect"][0][0]))


def classify_pass(long_axis_px: float, min_long_px: float | None, max_long_px: float | None) -> str:
    if min_long_px is not None and long_axis_px < min_long_px:
        return "FAIL_SMALL"
    if max_long_px is not None and long_axis_px > max_long_px:
        return "FAIL_LARGE"
    return "PASS"


def serialize_candidates(candidates: list[dict[str, Any]], min_long_px: float | None, max_long_px: float | None) -> list[dict[str, Any]]:
    stones = []
    for idx, cand in enumerate(candidates, start=1):
        status = classify_pass(cand["long_axis_px"], min_long_px, max_long_px)
        stones.append({
            "id": idx,
            "status": status,
            "area_px": round(float(cand["area_px"]), 2),
            "long_axis_px": round(float(cand["long_axis_px"]), 2),
            "short_axis_px": round(float(cand["short_axis_px"]), 2),
            "aspect_ratio": round(float(cand["aspect_ratio"]), 4),
            "extent": round(float(cand["extent"]), 4),
            "solidity": round(float(cand["solidity"]), 4),
            "mean_l": round(float(cand["mean_l"]), 2),
            "ring_mean_l": round(float(cand["ring_mean_l"]), 2),
            "ring_delta_l": round(float(cand["ring_delta_l"]), 2),
            "box": cand["box"].astype(int).tolist(),
            "polygon": cand["polygon"].astype(int).tolist(),
        })
    return stones


def draw_result(image_bgr: np.ndarray, candidates: list[dict[str, Any]], min_long_px: float | None, max_long_px: float | None) -> np.ndarray:
    out = image_bgr.copy()
    for idx, cand in enumerate(candidates, start=1):
        box = cand["box"].astype(int)
        poly = cand["polygon"].astype(int)
        status = classify_pass(cand["long_axis_px"], min_long_px, max_long_px)

        # BGR colors
        box_color = (255, 0, 0)       # blue
        poly_color = (0, 255, 255)    # yellow
        if status != "PASS":
            poly_color = (0, 0, 255)  # red

        cv2.polylines(out, [box], True, box_color, thickness=2)
        cv2.polylines(out, [poly], True, poly_color, thickness=2)

        center = tuple(np.round(cand["rect"][0]).astype(int))
        label = f"S{idx} L{cand['long_axis_px']:.1f} A{cand['area_px']:.0f}"
        if status != "PASS":
            label += f" {status}"
        cv2.putText(out, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.55, poly_color, 2, cv2.LINE_AA)

        if cand.get("contour") is not None:
            M = cv2.moments(cand["contour"])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)

    return out


def detect(image_bgr: np.ndarray, params: DetectorParams) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    mask, debug = segment_stones(image_bgr, params)
    candidates = extract_candidates(mask, image_bgr, params)
    return candidates, debug


def save_debug_images(out_dir: Path, stem: str, timestamp: str, debug: dict[str, np.ndarray]) -> None:
    for name, arr in debug.items():
        cv2.imwrite(str(out_dir / f"{stem}_debug_{name}_{timestamp}.png"), arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate/stone detector for dark tray images")
    parser.add_argument("--image", type=str, required=True, help="input image path")
    parser.add_argument("--out-dir", type=str, default="./result", help="output directory")
    parser.add_argument("--roi", type=str, default=None, help="x,y,w,h. 예: 120,155,1460,1090")

    parser.add_argument("--min-delta-l", type=float, default=20.0)
    parser.add_argument("--bg-sigma", type=float, default=55.0)
    parser.add_argument("--morph-close", type=int, default=35)
    parser.add_argument("--morph-open", type=int, default=7)
    parser.add_argument("--min-area", type=int, default=3500)
    parser.add_argument("--min-ring-delta-l", type=float, default=3.5)
    parser.add_argument("--max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--merge-distance", type=float, default=45.0)
    parser.add_argument("--no-otsu-helper", action="store_true")

    parser.add_argument("--min-long-px", type=float, default=None, help="PASS 판정 최소 장축 길이(px)")
    parser.add_argument("--max-long-px", type=float, default=None, help="PASS 판정 최대 장축 길이(px)")
    parser.add_argument("--debug", action="store_true", help="save intermediate images")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    params = DetectorParams(
        roi=parse_roi(args.roi),
        min_delta_l=args.min_delta_l,
        bg_sigma=args.bg_sigma,
        morph_close_px=args.morph_close,
        morph_open_px=args.morph_open,
        min_area_px=args.min_area,
        min_ring_delta_l=args.min_ring_delta_l,
        max_aspect_ratio=args.max_aspect_ratio,
        merge_distance=args.merge_distance,
        use_otsu_helper=not args.no_otsu_helper,
    )

    candidates, debug = detect(image, params)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = image_path.stem

    annotated = draw_result(image, candidates, args.min_long_px, args.max_long_px)
    out_img = out_dir / f"{stem}_detected_{timestamp}.png"
    out_json = out_dir / f"{stem}_results_refined_{timestamp}.json"
    cv2.imwrite(str(out_img), annotated)

    stones = serialize_candidates(candidates, args.min_long_px, args.max_long_px)
    pass_count = sum(1 for s in stones if s["status"] == "PASS")
    fail_count = len(stones) - pass_count

    payload = {
        "image": str(image_path),
        "timestamp": timestamp,
        "params": asdict(params),
        "total_stones": len(stones),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "stones": stones,
        "output_image": str(out_img),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.debug:
        save_debug_images(out_dir, stem, timestamp, debug)

    print(f"total_stones={len(stones)} pass={pass_count} fail={fail_count}")
    print(f"image={out_img}")
    print(f"json={out_json}")


if __name__ == "__main__":
    main()