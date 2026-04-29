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
    roi: tuple[int, int, int, int] | None = None
    roi_erode_px: int = 10
    roi_margin_px: int = 40

    # illumination normalization
    median_blur_ksize: int = 5
    bilateral_d: int = 9
    bilateral_sigma_color: float = 55.0
    bilateral_sigma_space: float = 55.0
    bg_sigma: float = 55.0
    min_delta_l: float = 14.0

    # LAB CLAHE: 회색/검정 골재 대비 강화
    use_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: int = 8

    # HSV gray helper: 회색/검정 골재 후보 보완
    use_hsv_gray_helper: bool = True
    hsv_low_sat_max: int = 60
    hsv_v_min: int = 50
    hsv_v_max: int = 180
    # HSV 조건만 쓰면 tray도 잡힐 수 있어 V channel local contrast gate를 추가
    hsv_gray_min_local_std: float = 6.0
    hsv_gray_std_kernel: int = 17

    # optional absolute lightness helper. 기본 비활성화 권장.
    use_otsu_helper: bool = False
    otsu_offset: float = -6.0

    # morphology
    morph_close_px: int = 21
    morph_open_px: int = 7
    fill_holes: bool = True

    # reflection / smooth tray suppression
    reflection_low_sat_max: int = 55
    reflection_min_delta_l: float = 4.0
    reflection_max_gradient: float = 10.0
    reflection_max_local_std: float = 5.0
    min_support_gradient: float = 24.0
    min_support_local_std: float = 20.0
    expansion_kernel_px: int = 5
    expansion_iterations: int = 3

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
    min_edge_density: float = 0.012
    max_seed_expansion_ratio: float = 2.0

    # white zigzag / tray marking reject
    white_l_min: int = 170
    white_ab_tol: int = 12
    white_dilate_px: int = 9
    white_overlap_ratio: float = 0.05
    # white stone까지 reject하지 않도록 white marking은 기본적으로 가장자리 band에서만 사용
    white_marking_band_px: int = 180

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


def make_edge_band_mask(shape: tuple[int, int], roi: tuple[int, int, int, int] | None, band_px: int) -> np.ndarray:
    """흰색 지그재그/테두리 제거용 가장자리 band mask."""
    h, w = shape
    band = np.zeros((h, w), np.uint8)
    if band_px <= 0:
        band[:, :] = 255
        return band

    if roi is None:
        band[:band_px, :] = 255
        band[max(0, h - band_px):, :] = 255
        band[:, :band_px] = 255
        band[:, max(0, w - band_px):] = 255
    else:
        rx, ry, rw, rh = roi
        x0, y0 = max(0, rx), max(0, ry)
        x1, y1 = min(w, rx + rw), min(h, ry + rh)
        band[y0:min(y1, y0 + band_px), x0:x1] = 255
        band[max(y0, y1 - band_px):y1, x0:x1] = 255
        band[y0:y1, x0:min(x1, x0 + band_px)] = 255
        band[y0:y1, max(x0, x1 - band_px):x1] = 255
    return band


def fill_binary_holes(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8) * 255
    h, w = binary.shape
    flood = binary.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def fill_holes_by_component(binary: np.ndarray, max_area_px: float) -> np.ndarray:
    """큰 tray/reflection blob은 그대로 두고, 객체 크기 component에만 hole fill 적용."""
    binary = (binary > 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        component = (labels == idx).astype(np.uint8) * 255
        if area <= max_area_px:
            component = fill_binary_holes(component)
        out = cv2.bitwise_or(out, component)
    return out


def apply_clahe_to_l(L: np.ndarray, params: DetectorParams) -> np.ndarray:
    if not params.use_clahe:
        return L
    tile = max(2, int(params.clahe_tile_grid))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(params.clahe_clip_limit)),
        tileGridSize=(tile, tile),
    )
    return clahe.apply(L)


def local_lightness(image_bgr: np.ndarray, params: DetectorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L_raw = lab[:, :, 0]

    # 수정 1: CLAHE로 회색/검정 골재의 local contrast 강화
    L = apply_clahe_to_l(L_raw, params)

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
    return L_raw, L, L_smooth, bg, delta


def local_std_image(gray: np.ndarray, ksize: int) -> np.ndarray:
    k = odd_kernel(ksize)
    f = gray.astype(np.float32)
    mean = cv2.GaussianBlur(f, (k, k), 0)
    mean_sq = cv2.GaussianBlur(f * f, (k, k), 0)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def normalize_to_u8(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.clip(values * scale, 0, 255).astype(np.uint8)


def filter_components_by_area(
    mask: np.ndarray,
    min_area_px: int | float = 0,
    max_area_px: int | float | None = None,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num <= 1:
        return np.zeros_like(mask)

    keep = np.zeros(num, dtype=np.uint8)
    for idx in range(1, num):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue
        keep[idx] = 255
    return keep[labels]


def hsv_gray_helper_mask(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    L_for_texture: np.ndarray,
    params: DetectorParams,
) -> tuple[np.ndarray, np.ndarray]:
    if not params.use_hsv_gray_helper:
        z = np.zeros_like(roi_mask)
        return z, z

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 요청 조건: S < 60, 50 < V < 180
    hsv_cond = (
        (s < params.hsv_low_sat_max)
        & (v > params.hsv_v_min)
        & (v < params.hsv_v_max)
        & (roi_mask > 0)
    )

    # tray도 저채도/중간밝기라서 전체 배경이 잡힐 수 있음.
    # V 채널의 local contrast가 있는 부분만 helper로 사용.
    v_blur = cv2.GaussianBlur(v, (0, 0), 15)
    local_diff = cv2.absdiff(v, v_blur)
    texture_cond = local_diff > params.hsv_gray_min_local_std

    mask = (hsv_cond & texture_cond).astype(np.uint8) * 255
    roi_area = float(np.count_nonzero(roi_mask))
    max_hsv_component_area = max(float(params.min_area_px), params.max_area_frac * roi_area)
    mask = filter_components_by_area(mask, max_area_px=max_hsv_component_area)
    return mask, local_diff


def otsu_lightness_mask(L_smooth: np.ndarray, roi_mask: np.ndarray, offset: float) -> np.ndarray:
    vals = L_smooth[roi_mask > 0]
    if vals.size < 100:
        return np.zeros_like(L_smooth, np.uint8)

    # ROI 내부 값만으로 Otsu threshold 계산
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(thr) + offset
    mask = ((L_smooth.astype(np.float32) >= threshold) & (roi_mask > 0)).astype(np.uint8) * 255
    return mask


def detect_white_markings(image_bgr: np.ndarray, roi_mask: np.ndarray, params: DetectorParams) -> np.ndarray:
    """밝고 무채색인 흰색 지그재그/테두리 패턴 마스크."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    white_like = (
        (L > params.white_l_min)
        & (np.abs(A.astype(np.int16) - 128) <= params.white_ab_tol)
        & (np.abs(B.astype(np.int16) - 128) <= params.white_ab_tol)
        & (roi_mask > 0)
    )

    # white stone까지 reject되는 것을 막기 위해, white marking은 기본적으로 가장자리 band에서만 사용.
    edge_band = make_edge_band_mask(L.shape, params.roi, params.white_marking_band_px)
    white_like = white_like & (edge_band > 0)

    white_mask = white_like.astype(np.uint8) * 255
    if params.white_dilate_px > 1:
        white_mask = cv2.dilate(white_mask, ellipse_kernel(params.white_dilate_px), iterations=1)
    return white_mask


def constrained_expand(seed_mask: np.ndarray, support_mask: np.ndarray, params: DetectorParams) -> np.ndarray:
    seed = cv2.bitwise_and(seed_mask, support_mask)
    if params.expansion_iterations <= 0:
        return seed

    kernel = ellipse_kernel(params.expansion_kernel_px)
    current = seed.copy()
    for _ in range(int(params.expansion_iterations)):
        grown = cv2.dilate(current, kernel, iterations=1)
        next_mask = cv2.bitwise_and(grown, support_mask)
        if np.array_equal(next_mask, current):
            break
        current = next_mask
    return current


def segment_stones(image_bgr: np.ndarray, params: DetectorParams) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h, w = image_bgr.shape[:2]
    roi_mask = make_roi_mask((h, w), params.roi, params.roi_erode_px)
    L_raw, L_clahe, L_smooth, bg, delta = local_lightness(image_bgr, params)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    local_std = local_std_image(L_raw, params.hsv_gray_std_kernel)
    grad = gradient_magnitude(L_smooth)
    edge_map = cv2.Canny(L_smooth, 50, 150)
    texture_gate = (
        (grad >= params.min_support_gradient)
        | (local_std >= params.min_support_local_std)
        | (edge_map > 0)
    )

    # 1차 seed도 texture/edge가 있는 곳으로 제한해 완만한 반사 영역을 시작점에서 제외한다.
    seed_mask_delta = ((delta >= params.min_delta_l) & texture_gate & (roi_mask > 0)).astype(np.uint8) * 255

    # HSV gray는 최종 영역이 아니라 후보 시작점(seed)으로만 사용한다.
    seed_mask_hsv_gray, local_diff_vis = hsv_gray_helper_mask(image_bgr, roi_mask, L_clahe, params)

    if params.use_otsu_helper:
        mask_otsu = otsu_lightness_mask(L_smooth, roi_mask, params.otsu_offset)
    else:
        mask_otsu = np.zeros_like(seed_mask_delta)

    smooth_bright = (
        (sat < params.reflection_low_sat_max)
        & (delta >= params.reflection_min_delta_l)
        & (grad < params.reflection_max_gradient)
        & (local_std < params.reflection_max_local_std)
        & (roi_mask > 0)
    )
    reflection_mask = smooth_bright.astype(np.uint8) * 255

    support = (
        (
            ((delta >= params.min_delta_l * 0.35) & texture_gate)
            | (edge_map > 0)
        )
        & (reflection_mask == 0)
        & (roi_mask > 0)
    ).astype(np.uint8) * 255

    seed_mask = cv2.bitwise_or(seed_mask_delta, seed_mask_hsv_gray)
    if params.use_otsu_helper:
        seed_mask = cv2.bitwise_or(seed_mask, mask_otsu)
    seed_mask = cv2.bitwise_and(seed_mask, cv2.bitwise_not(reflection_mask))

    mask = constrained_expand(seed_mask, support, params)
    final_mask_before_morphology = mask.copy()

    # close/fill 전에 작은 texture noise를 먼저 정리해야 tray 전체가 큰 blob으로 붙지 않음.
    if params.morph_open_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(params.morph_open_px), iterations=1)
    mask = filter_components_by_area(mask, min_area_px=max(100.0, params.min_area_px * 0.25))

    close_px = min(int(params.morph_close_px), 25)
    if close_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse_kernel(close_px), iterations=1)
    if params.fill_holes:
        roi_area = float(np.count_nonzero(roi_mask))
        max_area_px = max(params.min_area_px, params.max_area_frac * roi_area)
        mask = fill_holes_by_component(mask, max_area_px)

    mask = cv2.bitwise_and(mask, roi_mask)
    final_mask_after_morphology = mask.copy()

    # 수정 3: 흰색 지그재그/테두리 마스크 생성
    white_mask = detect_white_markings(image_bgr, roi_mask, params)

    debug = {
        "roi_mask": roi_mask,
        "L_raw": L_raw,
        "L_clahe": L_clahe,
        "L_smooth": L_smooth,
        "background_L": bg,
        "delta_L_vis": np.clip(delta + 128, 0, 255).astype(np.uint8),
        "local_std_vis": normalize_to_u8(local_std, 12.0),
        "local_gradient_vis": normalize_to_u8(grad, 4.0),
        "local_diff_vis": local_diff_vis,
        "seed_mask_delta": seed_mask_delta,
        "seed_mask_hsv_gray": seed_mask_hsv_gray,
        "mask_delta": seed_mask_delta,
        "mask_hsv_gray": seed_mask_hsv_gray,
        "mask_otsu": mask_otsu,
        "reflection_mask": reflection_mask,
        "edge_map": edge_map,
        "support_mask": support,
        "seed_mask_combined": seed_mask,
        "final_mask_before_morphology": final_mask_before_morphology,
        "final_mask_after_morphology": final_mask_after_morphology,
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


def roi_edge_reject(cx: int, cy: int, shape: tuple[int, int], params: DetectorParams) -> bool:
    h, w = shape
    margin = int(params.roi_margin_px)
    if margin <= 0:
        return False

    if params.roi is not None:
        rx, ry, rw, rh = params.roi
        return bool(
            cx < rx + margin
            or cy < ry + margin
            or cx > rx + rw - margin
            or cy > ry + rh - margin
        )

    return bool(cx < margin or cy < margin or cx > w - margin or cy > h - margin)


def draw_rejected_contours(
    image_bgr: np.ndarray,
    rejected: list[dict[str, Any]],
) -> np.ndarray:
    out = image_bgr.copy()
    for item in rejected:
        contour = item["contour"]
        reason = item["reason"]
        cv2.drawContours(out, [contour], -1, (0, 0, 255), thickness=2)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            x, y, _, _ = cv2.boundingRect(contour)
        else:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        cv2.putText(out, reason[:32], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def extract_candidates(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    params: DetectorParams,
    seed_mask: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    h, w = mask.shape
    roi_mask = make_roi_mask((h, w), params.roi, params.roi_erode_px)
    roi_area = float(np.count_nonzero(roi_mask))
    max_area_px = max(params.min_area_px, params.max_area_frac * roi_area)

    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    white_mask = detect_white_markings(image_bgr, roi_mask, params)
    if seed_mask is None:
        seed_mask = mask
    if edge_map is None:
        edge_map = cv2.Canny(L, 50, 150)
    edge_density_vis = np.zeros_like(mask)
    rejected: list[dict[str, Any]] = []

    def reject(contour: np.ndarray, reason: str) -> None:
        rejected.append({"contour": contour, "reason": reason})

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= max(100.0, params.min_area_px * 0.25)]
    contours = merge_close_contours(contours, params.merge_distance, params.merge_max_area_ratio)

    candidates: list[dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < params.min_area_px or area > max_area_px:
            reject(contour, "area")
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            reject(contour, "moment")
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # ROI가 있으면 ROI 기준, 없으면 전체 이미지 기준으로 가장자리 artifact 제거
        if roi_edge_reject(cx, cy, (h, w), params):
            reject(contour, "roi_edge")
            continue

        obj_mask = contour_to_mask((h, w), contour)
        obj_px = max(int(np.count_nonzero(obj_mask)), 1)

        seed_px = int(np.count_nonzero(cv2.bitwise_and(obj_mask, seed_mask)))
        seed_expansion_ratio = obj_px / max(seed_px, 1)
        if seed_px == 0 or seed_expansion_ratio > params.max_seed_expansion_ratio:
            reject(contour, f"seed_ratio {seed_expansion_ratio:.2f}")
            continue

        edge_px = int(np.count_nonzero(cv2.bitwise_and(edge_map, obj_mask)))
        edge_density = edge_px / obj_px
        edge_density_vis[obj_mask > 0] = np.clip(edge_density * 2550.0, 0, 255)
        if edge_density < params.min_edge_density:
            reject(contour, f"edge_density {edge_density:.3f}")
            continue

        # 수정 3: 흰색 지그재그/테두리와 겹치는 contour 제거
        white_overlap = int(np.count_nonzero(cv2.bitwise_and(obj_mask, white_mask)))
        white_overlap_ratio = white_overlap / obj_px
        if white_overlap_ratio >= params.white_overlap_ratio:
            reject(contour, f"white {white_overlap_ratio:.2f}")
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw <= 1 or rh <= 1:
            reject(contour, "rect")
            continue

        long_axis = float(max(rw, rh))
        short_axis = float(min(rw, rh))
        aspect = long_axis / max(short_axis, 1e-6)
        if aspect > params.max_aspect_ratio:
            reject(contour, f"aspect {aspect:.2f}")
            continue

        rect_area = float(rw * rh)
        extent = area / max(rect_area, 1e-6)
        if extent < params.min_extent:
            reject(contour, f"extent {extent:.2f}")
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1e-6)
        if solidity < params.min_solidity:
            reject(contour, f"solidity {solidity:.2f}")
            continue

        ring = cv2.dilate(obj_mask, ellipse_kernel(params.ring_px * 2 + 1), iterations=1)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(obj_mask))
        ring = cv2.bitwise_and(ring, roi_mask)

        mean_obj = float(cv2.mean(L, mask=obj_mask)[0])
        mean_ring = float(cv2.mean(L, mask=ring)[0]) if np.count_nonzero(ring) else mean_obj
        ring_delta_signed = mean_obj - mean_ring
        contrast_delta = abs(ring_delta_signed)

        # 어두운 골재는 주변보다 어두울 수 있으므로 signed delta가 아니라 abs contrast로 검사
        if contrast_delta < params.min_ring_delta_l:
            reject(contour, f"ring {contrast_delta:.2f}")
            continue

        epsilon = params.approx_epsilon_ratio * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        box = cv2.boxPoints(rect).astype(int)

        score = area * max(contrast_delta, 0.01) * max(edge_density, 0.001) * max(solidity, 0.01) * max(extent, 0.01)
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
            "ring_delta_l": ring_delta_signed,
            "ring_abs_delta_l": contrast_delta,
            "edge_density": edge_density,
            "seed_area_px": seed_px,
            "seed_expansion_ratio": seed_expansion_ratio,
            "white_overlap_ratio": white_overlap_ratio,
            "score": score,
        })

    kept = nms_rotated(candidates, params.nms_iou)
    debug = {
        "edge_density_vis": edge_density_vis,
        "rejected_contours": draw_rejected_contours(image_bgr, rejected),
    }
    return sorted(kept, key=lambda c: (c["rect"][0][1], c["rect"][0][0])), debug


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
            "ring_abs_delta_l": round(float(cand.get("ring_abs_delta_l", abs(cand["ring_delta_l"]))), 2),
            "edge_density": round(float(cand.get("edge_density", 0.0)), 5),
            "seed_area_px": int(cand.get("seed_area_px", 0)),
            "seed_expansion_ratio": round(float(cand.get("seed_expansion_ratio", 0.0)), 4),
            "white_overlap_ratio": round(float(cand.get("white_overlap_ratio", 0.0)), 4),
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
    candidates, candidate_debug = extract_candidates(
        mask,
        image_bgr,
        params,
        seed_mask=debug.get("seed_mask_combined"),
        edge_map=debug.get("edge_map"),
    )
    debug.update(candidate_debug)
    return candidates, debug


def save_debug_images(out_dir: Path, stem: str, timestamp: str, debug: dict[str, np.ndarray]) -> None:
    for name, arr in debug.items():
        cv2.imwrite(str(out_dir / f"{stem}_debug_{name}_{timestamp}.png"), arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate/stone detector for dark tray images")
    parser.add_argument("--image", type=str, required=True, help="input image path")
    parser.add_argument("--out-dir", type=str, default="./result", help="output directory")
    parser.add_argument("--roi", type=str, default=None, help="x,y,w,h. 예: 130,170,1270,1000")
    parser.add_argument("--roi-erode", type=int, default=10)
    parser.add_argument("--roi-margin", type=int, default=40)

    parser.add_argument("--min-delta-l", type=float, default=14.0)
    parser.add_argument("--bg-sigma", type=float, default=55.0)
    parser.add_argument("--morph-close", type=int, default=21)
    parser.add_argument("--morph-open", type=int, default=7)
    parser.add_argument("--min-area", type=int, default=3500)
    parser.add_argument("--min-ring-delta-l", type=float, default=3.5)
    parser.add_argument("--max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--merge-distance", type=float, default=45.0)
    parser.add_argument("--min-edge-density", type=float, default=0.012)
    parser.add_argument("--max-seed-expansion-ratio", type=float, default=2.0)
    parser.add_argument("--reflection-max-gradient", type=float, default=10.0)
    parser.add_argument("--reflection-max-local-std", type=float, default=5.0)
    parser.add_argument("--min-support-gradient", type=float, default=24.0)
    parser.add_argument("--min-support-local-std", type=float, default=20.0)

    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.5)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)

    parser.add_argument("--no-hsv-gray-helper", action="store_true")
    parser.add_argument("--hsv-low-sat-max", "--hsv-s-max", dest="hsv_low_sat_max", type=int, default=60)
    parser.add_argument("--hsv-v-min", type=int, default=50)
    parser.add_argument("--hsv-v-max", type=int, default=180)
    parser.add_argument("--hsv-gray-min-local-std", type=float, default=6.0)
    parser.add_argument("--hsv-gray-std-kernel", type=int, default=17)

    parser.add_argument("--use-otsu-helper", action="store_true")
    parser.add_argument("--no-otsu-helper", action="store_true")  # 기존 커맨드 호환용

    parser.add_argument("--white-l-min", type=int, default=170)
    parser.add_argument("--white-ab-tol", type=int, default=12)
    parser.add_argument("--white-dilate", type=int, default=9)
    parser.add_argument("--white-overlap-ratio", type=float, default=0.05)
    parser.add_argument("--white-marking-band", type=int, default=180)

    parser.add_argument("--min-long-px", type=float, default=None, help="PASS 판정 최소 장축 길이(px)")
    parser.add_argument("--max-long-px", type=float, default=None, help="PASS 판정 최대 장축 길이(px)")
    parser.add_argument("--debug", action="store_true", help="save intermediate images")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"파일 자체가 없습니다: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지 로딩 실패. 경로는 있으나 포맷/권한 문제가 있을 수 있습니다: {image_path}")

    use_otsu = bool(args.use_otsu_helper) and not bool(args.no_otsu_helper)

    params = DetectorParams(
        roi=parse_roi(args.roi),
        roi_erode_px=args.roi_erode,
        roi_margin_px=args.roi_margin,
        min_delta_l=args.min_delta_l,
        bg_sigma=args.bg_sigma,
        morph_close_px=args.morph_close,
        morph_open_px=args.morph_open,
        min_area_px=args.min_area,
        min_ring_delta_l=args.min_ring_delta_l,
        max_aspect_ratio=args.max_aspect_ratio,
        merge_distance=args.merge_distance,
        min_edge_density=args.min_edge_density,
        max_seed_expansion_ratio=args.max_seed_expansion_ratio,
        reflection_max_gradient=args.reflection_max_gradient,
        reflection_max_local_std=args.reflection_max_local_std,
        min_support_gradient=args.min_support_gradient,
        min_support_local_std=args.min_support_local_std,
        use_clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        use_hsv_gray_helper=not args.no_hsv_gray_helper,
        hsv_low_sat_max=args.hsv_low_sat_max,
        hsv_v_min=args.hsv_v_min,
        hsv_v_max=args.hsv_v_max,
        hsv_gray_min_local_std=args.hsv_gray_min_local_std,
        hsv_gray_std_kernel=args.hsv_gray_std_kernel,
        use_otsu_helper=use_otsu,
        white_l_min=args.white_l_min,
        white_ab_tol=args.white_ab_tol,
        white_dilate_px=args.white_dilate,
        white_overlap_ratio=args.white_overlap_ratio,
        white_marking_band_px=args.white_marking_band,
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
