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
from dataclasses import asdict, dataclass, replace
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
    bg_sigma: float = 60.0
    min_delta_l: float = 12.0

    delta_cc_min_area_px: int = 50
    delta_border_cc_min_area_px: int = 300

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
    reflection_max_gradient: float = 6.0
    reflection_max_local_std: float = 5.0
    dark_side_delta_l: float = 8.0
    dark_side_min_gradient: float = 22.0
    side_seed_distance_px: int = 75
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
    min_extent: float = 0.05
    min_solidity: float = 0.08
    max_aspect_ratio: float = 4.0
    min_ring_delta_l: float = 3.5
    ring_px: int = 30
    min_edge_density: float = 0.012
    max_seed_expansion_ratio: float = 2.8
    hard_seed_expansion_ratio: float = 4.0
    refine_seed_ratio: float = 2.0
    max_seed_distance_px: int = 25
    refine_far_seed_px: int = 15
    refine_close_px: int = 9
    reflection_refine_low_sat_max: int = 35
    reflection_refine_v_min: int = 80
    reflection_refine_max_gradient: float = 8.0
    reflection_refine_max_local_std: float = 5.0

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


def component_touches_mask(component: np.ndarray, touch_mask: np.ndarray) -> bool:
    return bool(np.any((component > 0) & (touch_mask > 0)))


def filter_mask_delta_components(
    mask_delta: np.ndarray,
    roi_mask: np.ndarray,
    white_mask: np.ndarray,
    params: DetectorParams,
) -> np.ndarray:
    """mask_delta에서 경계 후보 성분만 남기기 위한 CC 필터.

    - 아주 작은 noise 성분 제거
    - ROI 테두리와 연결된 큰 성분 제거
    - 흰색 지그재그/테두리와 겹치는 성분 제거
    """
    binary = ((mask_delta > 0) & (roi_mask > 0)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num <= 1:
        return np.zeros_like(mask_delta)

    # ROI 경계 ring(1px) 생성: 경계에 닿는 큰 성분 제거에 사용
    roi_eroded = cv2.erode(roi_mask, np.ones((3, 3), np.uint8), iterations=1)
    roi_border = cv2.subtract(roi_mask, roi_eroded)

    out = np.zeros_like(mask_delta)
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(params.delta_cc_min_area_px):
            continue

        comp = (labels == idx).astype(np.uint8) * 255

        if (
            area >= int(params.delta_border_cc_min_area_px)
            and component_touches_mask(comp, roi_border)
        ):
            continue

        comp_px = max(int(np.count_nonzero(comp)), 1)
        white_overlap = int(np.count_nonzero(cv2.bitwise_and(comp, white_mask)))
        white_overlap_ratio = white_overlap / comp_px
        if white_overlap_ratio >= max(0.01, params.white_overlap_ratio * 0.5):
            continue

        out = cv2.bitwise_or(out, comp)

    return cv2.bitwise_and(out, roi_mask)


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

    seed_mask_delta = ((delta >= params.min_delta_l) & texture_gate & (roi_mask > 0)).astype(np.uint8) * 255
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
    for cand in sorted(candidates, key=lambda x: x.get("final_score", x["score"]), reverse=True):
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


def make_reject_record(contour: np.ndarray, reason: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    x, y, w, h = cv2.boundingRect(contour)
    record: dict[str, Any] = {
        "contour": contour,
        "reason": reason,
        "bbox": [int(x), int(y), int(w), int(h)],
    }
    if metrics:
        record.update(metrics)
    return record


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

    def reject(contour: np.ndarray, reason: str, metrics: dict[str, Any] | None = None) -> None:
        rejected.append(make_reject_record(contour, reason, metrics))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= max(100.0, params.min_area_px * 0.25)]
    contours = merge_close_contours(contours, params.merge_distance, params.merge_max_area_ratio)

    candidates: list[dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < params.min_area_px or area > max_area_px:
            reject(contour, "area", {"area_px": round(area, 2)})
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            reject(contour, "moment", {"area_px": round(area, 2)})
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # ROI가 있으면 ROI 기준, 없으면 전체 이미지 기준으로 가장자리 artifact 제거
        if roi_edge_reject(cx, cy, (h, w), params):
            reject(contour, "roi_edge", {"area_px": round(area, 2)})
            continue

        obj_mask = contour_to_mask((h, w), contour)
        obj_px = max(int(np.count_nonzero(obj_mask)), 1)

        seed_px = int(np.count_nonzero(cv2.bitwise_and(obj_mask, seed_mask)))
        seed_expansion_ratio = obj_px / max(seed_px, 1)
        base_metrics: dict[str, Any] = {
            "obj_px": obj_px,
            "seed_px": seed_px,
            "seed_expansion_ratio": round(float(seed_expansion_ratio), 4),
            "seed_expansion_ratio_before_refine": round(float(seed_expansion_ratio), 4),
            "obj_px_before_refine": obj_px,
        }
        if seed_px == 0 or seed_expansion_ratio > params.hard_seed_expansion_ratio:
            reject(contour, f"seed_ratio {seed_expansion_ratio:.2f}", base_metrics)
            continue

        edge_px = int(np.count_nonzero(cv2.bitwise_and(edge_map, obj_mask)))
        edge_density = edge_px / obj_px
        base_metrics["edge_density"] = round(float(edge_density), 5)
        edge_density_vis[obj_mask > 0] = np.clip(edge_density * 2550.0, 0, 255)
        if edge_px == 0 and seed_expansion_ratio > params.max_seed_expansion_ratio:
            reject(contour, f"no_edge_seed_ratio {seed_expansion_ratio:.2f}", base_metrics)
            continue

        # 수정 3: 흰색 지그재그/테두리와 겹치는 contour 제거
        white_overlap = int(np.count_nonzero(cv2.bitwise_and(obj_mask, white_mask)))
        white_overlap_ratio = white_overlap / obj_px
        base_metrics["white_overlap_ratio"] = round(float(white_overlap_ratio), 4)
        if white_overlap_ratio >= params.white_overlap_ratio:
            reject(contour, f"white {white_overlap_ratio:.2f}", base_metrics)
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw <= 1 or rh <= 1:
            reject(contour, "rect", base_metrics)
            continue

        long_axis = float(max(rw, rh))
        short_axis = float(min(rw, rh))
        aspect = long_axis / max(short_axis, 1e-6)
        base_metrics["aspect_ratio"] = round(float(aspect), 4)
        if aspect > params.max_aspect_ratio:
            reject(contour, f"aspect {aspect:.2f}", base_metrics)
            continue

        rect_area = float(rw * rh)
        extent = area / max(rect_area, 1e-6)
        base_metrics["extent"] = round(float(extent), 4)
        if extent < params.min_extent:
            reject(contour, f"extent {extent:.2f}", base_metrics)
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1e-6)
        base_metrics["solidity"] = round(float(solidity), 4)
        if solidity < params.min_solidity:
            reject(contour, f"solidity {solidity:.2f}", base_metrics)
            continue

        ring = cv2.dilate(obj_mask, ellipse_kernel(params.ring_px * 2 + 1), iterations=1)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(obj_mask))
        ring = cv2.bitwise_and(ring, roi_mask)

        mean_obj = float(cv2.mean(L, mask=obj_mask)[0])
        mean_ring = float(cv2.mean(L, mask=ring)[0]) if np.count_nonzero(ring) else mean_obj
        ring_delta_signed = mean_obj - mean_ring
        contrast_delta = abs(ring_delta_signed)
        base_metrics["ring_abs_delta_l"] = round(float(contrast_delta), 2)
        base_metrics["mean_l"] = round(float(mean_obj), 2)
        base_metrics["ring_mean_l"] = round(float(mean_ring), 2)

        if mean_obj < 35.0 and ring_delta_signed < -params.min_ring_delta_l:
            reject(contour, f"dark_hole {ring_delta_signed:.2f}", base_metrics)
            continue

        # 어두운 골재는 주변보다 어두울 수 있으므로 signed delta가 아니라 abs contrast로 검사
        if contrast_delta < params.min_ring_delta_l:
            reject(contour, f"ring {contrast_delta:.2f}", base_metrics)
            continue

        epsilon = params.approx_epsilon_ratio * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        box = cv2.boxPoints(rect).astype(int)

        score = area * max(contrast_delta, 0.01) * max(edge_density, 0.001) * max(solidity, 0.01) * max(extent, 0.01)
        if edge_density < params.min_edge_density:
            score *= max(edge_density / max(params.min_edge_density, 1e-6), 0.2)
        if seed_expansion_ratio > params.max_seed_expansion_ratio:
            score *= params.max_seed_expansion_ratio / seed_expansion_ratio
        candidates.append({
            "rect": rect,
            "box": box,
            "polygon": polygon,
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
            "seed_expansion_ratio_before_refine": seed_expansion_ratio,
            "obj_px_before_refine": obj_px,
            "distance_refined": False,
            "refine_reason": "",
            "removed_reflection_px": 0,
            "white_overlap_ratio": white_overlap_ratio,
            "score": score,
        })

    kept = nms_rotated(candidates, params.nms_iou)
    final_contours = image_bgr.copy()
    for cand in kept:
        cv2.drawContours(final_contours, [cand["contour"]], -1, (0, 255, 0), thickness=2)
        cv2.polylines(final_contours, [cand["box"].astype(int)], True, (255, 0, 0), thickness=2)

    debug = {
        "edge_density_vis": edge_density_vis,
        "rejected_contours": draw_rejected_contours(image_bgr, rejected),
        "final_contours": final_contours,
        "refine_logs": [],
        "rejected_candidates": [
            {k: v for k, v in item.items() if k != "contour"}
            for item in rejected
        ],
    }
    return sorted(kept, key=lambda c: (c["rect"][0][1], c["rect"][0][0])), debug


def classify_pass(long_axis_px: float, min_long_px: float | None, max_long_px: float | None) -> str:
    if min_long_px is not None and long_axis_px < min_long_px:
        return "FAIL_SMALL"
    if max_long_px is not None and long_axis_px > max_long_px:
        return "FAIL_LARGE"
    return "PASS"


def compute_scale_mm_per_px(px_per_cm: float) -> float:
    if px_per_cm is None or px_per_cm <= 0:
        raise ValueError("px_per_cm must be positive.")
    return 10.0 / px_per_cm


def estimate_px_per_cm_from_zigzag(
    image_bgr: np.ndarray,
    debug_dir: Path | str | None = None,
    stem: str = "scale",
) -> float:
    """
    사진 상단/좌측 흰색 지그재그에서 1칸 = 1cm 기준 px_per_cm 추정.
    기본은 상단 지그재그를 우선 사용.
    """
    h, w = image_bgr.shape[:2]

    top_roi = image_bgr[int(h * 0.03):int(h * 0.16), int(w * 0.05):int(w * 0.80)]
    gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, xs = np.where(binary > 0)
    if len(xs) < 100:
        raise RuntimeError("지그재그 흰색 픽셀을 충분히 찾지 못했습니다. --px-per-cm을 직접 입력하세요.")

    projection = np.sum(binary > 0, axis=0).astype(np.float32)
    projection = cv2.GaussianBlur(projection.reshape(1, -1), (1, 15), 0).flatten()

    peaks: list[int] = []
    max_projection = float(np.max(projection))
    for i in range(2, len(projection) - 2):
        if projection[i] > projection[i - 1] and projection[i] > projection[i + 1]:
            if projection[i] > max_projection * 0.35:
                peaks.append(i)

    filtered: list[int] = []
    min_peak_gap = 20
    for p in peaks:
        if not filtered or p - filtered[-1] > min_peak_gap:
            filtered.append(p)
        elif projection[p] > projection[filtered[-1]]:
            filtered[-1] = p

    if len(filtered) < 5:
        raise RuntimeError("지그재그 peak를 충분히 찾지 못했습니다. --px-per-cm을 직접 입력하세요.")

    distances = np.diff(filtered)
    med = np.median(distances)
    valid = distances[(distances > med * 0.6) & (distances < med * 1.4)]

    if len(valid) < 4:
        raise RuntimeError("지그재그 간격 추정이 불안정합니다. --px-per-cm을 직접 입력하세요.")

    px_per_cm = float(np.median(valid))

    if debug_dir is not None:
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for p in filtered:
            cv2.line(debug_img, (p, 0), (p, debug_img.shape[0] - 1), (0, 0, 255), 1)
        cv2.imwrite(str(Path(debug_dir) / f"{stem}_debug_zigzag_scale.png"), debug_img)

    return px_per_cm


def evaluate_stone_quality(
    candidate: dict[str, Any],
    mm_per_px: float,
    size_min_mm: float,
    size_max_mm: float,
    ratio_pass_max: float,
) -> dict[str, Any]:
    rect = candidate["rect"]
    rw, rh = rect[1]
    length_px = float(max(rw, rh))
    breadth_px = float(min(rw, rh))
    length_mm = length_px * mm_per_px
    breadth_mm = breadth_px * mm_per_px
    elongation_ratio = length_mm / max(breadth_mm, 1e-6)

    size_pass = bool(size_min_mm <= breadth_mm <= size_max_mm)
    shape_pass = bool(elongation_ratio < ratio_pass_max)
    fail_reasons: list[str] = []
    if breadth_mm < size_min_mm:
        fail_reasons.append("BREADTH_TOO_SMALL")
    if breadth_mm > size_max_mm:
        fail_reasons.append("BREADTH_TOO_LARGE")
    if not shape_pass:
        fail_reasons.append("ELONGATION_RATIO_OVER_3")

    return {
        "length_px": round(length_px, 2),
        "breadth_px": round(breadth_px, 2),
        "length_mm": round(length_mm, 2),
        "breadth_mm": round(breadth_mm, 2),
        "elongation_ratio": round(float(elongation_ratio), 4),
        "size_pass": size_pass,
        "shape_pass": shape_pass,
        "final_pass": bool(size_pass and shape_pass),
        "fail_reasons": fail_reasons,
    }


def apply_quality_to_candidates(
    candidates: list[dict[str, Any]],
    mm_per_px: float | None,
    size_min_mm: float,
    size_max_mm: float,
    ratio_pass_max: float,
) -> None:
    if mm_per_px is None:
        return
    for candidate in candidates:
        candidate["quality"] = evaluate_stone_quality(
            candidate,
            mm_per_px,
            size_min_mm,
            size_max_mm,
            ratio_pass_max,
        )


def serialize_candidates(candidates: list[dict[str, Any]], min_long_px: float | None, max_long_px: float | None) -> list[dict[str, Any]]:
    stones = []
    for idx, cand in enumerate(candidates, start=1):
        quality = cand.get("quality")
        if quality is not None:
            status = "PASS" if bool(quality["final_pass"]) else "FAIL"
        else:
            status = classify_pass(cand["long_axis_px"], min_long_px, max_long_px)

        stone = {
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
            "seed_expansion_ratio_before_refine": round(float(cand.get("seed_expansion_ratio_before_refine", 0.0)), 4),
            "final_score": round(float(cand.get("final_score", cand.get("score", 0.0))), 4),
            "chosen_preset": str(cand.get("chosen_preset", "")),
            "distance_refined": bool(cand.get("distance_refined", False)),
            "refine_reason": str(cand.get("refine_reason", "")),
            "removed_reflection_px": int(cand.get("removed_reflection_px", 0)),
            "white_overlap_ratio": round(float(cand.get("white_overlap_ratio", 0.0)), 4),
            "box": cand["box"].astype(int).tolist(),
            "polygon": cand["polygon"].astype(int).tolist(),
        }
        if quality is not None:
            stone.update({
                "length_px": quality["length_px"],
                "breadth_px": quality["breadth_px"],
                "length_mm": quality["length_mm"],
                "breadth_mm": quality["breadth_mm"],
                "elongation_ratio": quality["elongation_ratio"],
                "size_pass": bool(quality["size_pass"]),
                "shape_pass": bool(quality["shape_pass"]),
                "final_pass": bool(quality["final_pass"]),
                "fail_reasons": list(quality["fail_reasons"]),
            })
        stones.append(stone)
    return stones


def draw_result(image_bgr: np.ndarray, candidates: list[dict[str, Any]], min_long_px: float | None, max_long_px: float | None) -> np.ndarray:
    out = image_bgr.copy()
    for idx, cand in enumerate(candidates, start=1):
        box = cand["box"].astype(int)
        poly = cand["polygon"].astype(int)
        quality = cand.get("quality")
        if quality is not None:
            status = "PASS" if bool(quality["final_pass"]) else "FAIL"
        else:
            status = classify_pass(cand["long_axis_px"], min_long_px, max_long_px)

        # BGR colors
        box_color = (255, 0, 0)       # blue
        if quality is not None:
            poly_color = (0, 200, 0) if status == "PASS" else (0, 0, 255)
        else:
            poly_color = (0, 255, 255)    # yellow
            if status != "PASS":
                poly_color = (0, 0, 255)  # red

        cv2.polylines(out, [box], True, box_color, thickness=2)
        cv2.polylines(out, [poly], True, poly_color, thickness=2)

        center = tuple(np.round(cand["rect"][0]).astype(int))
        if quality is not None:
            label = (
                f"S{idx} {status} "
                f"B={quality['breadth_mm']:.1f} L={quality['length_mm']:.1f} R={quality['elongation_ratio']:.2f}"
            )
        else:
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


def auto_tune_presets(base: DetectorParams) -> dict[str, DetectorParams]:
    return {
        "conservative": replace(
            base,
            max_seed_distance_px=max(18, min(base.max_seed_distance_px, 24)),
            refine_far_seed_px=max(10, min(base.refine_far_seed_px, 14)),
            reflection_refine_low_sat_max=max(base.reflection_refine_low_sat_max, 45),
            reflection_refine_max_gradient=max(base.reflection_refine_max_gradient, 12.0),
            reflection_refine_max_local_std=max(base.reflection_refine_max_local_std, 8.0),
            max_seed_expansion_ratio=min(base.max_seed_expansion_ratio, 2.6),
        ),
        "balanced": replace(base),
        "permissive": replace(
            base,
            max_seed_distance_px=max(base.max_seed_distance_px, 55),
            refine_far_seed_px=max(base.refine_far_seed_px, 30),
            reflection_refine_low_sat_max=min(base.reflection_refine_low_sat_max, 35),
            reflection_refine_max_gradient=min(base.reflection_refine_max_gradient, 8.0),
            reflection_refine_max_local_std=min(base.reflection_refine_max_local_std, 5.0),
            max_seed_expansion_ratio=max(base.max_seed_expansion_ratio, 3.2),
        ),
        "edge_focused": replace(
            base,
            min_support_gradient=max(12.0, base.min_support_gradient * 0.75),
            min_edge_density=max(0.004, base.min_edge_density * 0.75),
            max_seed_distance_px=max(24, min(base.max_seed_distance_px, 40)),
        ),
        "seed_focused": replace(
            base,
            refine_seed_ratio=1.7,
            max_seed_distance_px=max(20, min(base.max_seed_distance_px, 30)),
            refine_far_seed_px=max(10, min(base.refine_far_seed_px, 18)),
            max_seed_expansion_ratio=min(base.max_seed_expansion_ratio, 2.5),
        ),
    }


def candidate_score_details(
    image_bgr: np.ndarray,
    params: DetectorParams,
    cand: dict[str, Any],
    preset_name: str,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    obj_mask = contour_to_mask((h, w), cand["contour"])
    obj_px = max(int(np.count_nonzero(obj_mask)), 1)

    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    L_refine = cv2.GaussianBlur(L, (0, 0), 3)
    local_std = local_std_image(L_refine, max(params.hsv_gray_std_kernel, 31))
    grad = gradient_magnitude(L_refine)
    reflection_mask = (
        (sat < params.reflection_refine_low_sat_max)
        & (val > params.reflection_refine_v_min)
        & (local_std < params.reflection_refine_max_local_std)
        & (grad < params.reflection_refine_max_gradient)
    ).astype(np.uint8) * 255
    reflection_overlap = int(np.count_nonzero(cv2.bitwise_and(obj_mask, reflection_mask))) / obj_px

    local_std_mean = float(cv2.mean(local_std, mask=obj_mask)[0])
    edge_density = float(cand.get("edge_density", 0.0))
    ring_delta = float(cand.get("ring_abs_delta_l", abs(float(cand.get("ring_delta_l", 0.0)))))
    seed_ratio = float(cand.get("seed_expansion_ratio", 0.0))
    extent = float(cand.get("extent", 0.0))
    solidity = float(cand.get("solidity", 0.0))
    white_overlap = float(cand.get("white_overlap_ratio", 0.0))
    refined_area_ratio = float(cand.get("area_px", 0.0)) / max(float(cand.get("obj_px_before_refine", cand.get("area_px", 1.0))), 1.0)
    if not cand.get("distance_refined", False):
        refined_area_ratio = 1.0

    penalties: list[str] = []
    score = 0.0
    score += min(edge_density / max(params.min_edge_density, 1e-6), 3.0) * 22.0
    score += min(local_std_mean / 8.0, 3.0) * 12.0
    score += min(ring_delta / max(params.min_ring_delta_l, 1e-6), 4.0) * 10.0
    score += min(max(solidity, 0.0), 1.0) * 12.0
    score += min(max(extent, 0.0), 1.0) * 8.0

    if seed_ratio > max(1.8, params.refine_seed_ratio * 0.9):
        penalty = min((seed_ratio - max(1.8, params.refine_seed_ratio * 0.9)) * 20.0, 30.0)
        score -= penalty
        penalties.append(f"seed_ratio_soft {seed_ratio:.2f}")
    if seed_ratio > params.max_seed_expansion_ratio:
        penalty = min((seed_ratio - params.max_seed_expansion_ratio) * 18.0, 35.0)
        score -= penalty
        penalties.append(f"seed_ratio +{seed_ratio - params.max_seed_expansion_ratio:.2f}")
    if reflection_overlap > 0.08:
        score -= min(reflection_overlap * 150.0, 70.0)
        penalties.append(f"reflection_overlap {reflection_overlap:.2f}")
    if white_overlap > 0.0:
        score -= min(white_overlap * 200.0, 50.0)
        penalties.append(f"white_overlap {white_overlap:.2f}")
    if refined_area_ratio < 0.65 and reflection_overlap < 0.18:
        score -= 45.0
        penalties.append(f"over_refined {refined_area_ratio:.2f}")
    if extent < params.min_extent:
        score -= 20.0
        penalties.append(f"low_extent {extent:.2f}")
    if solidity < params.min_solidity:
        score -= 20.0
        penalties.append(f"low_solidity {solidity:.2f}")
    if edge_density < params.min_edge_density:
        score -= 18.0
        penalties.append(f"low_edge {edge_density:.3f}")

    return {
        "area": round(float(cand.get("area_px", 0.0)), 2),
        "edge_density": round(edge_density, 5),
        "local_std": round(local_std_mean, 4),
        "ring_delta": round(ring_delta, 2),
        "seed_expansion_ratio": round(seed_ratio, 4),
        "reflection_overlap": round(float(reflection_overlap), 4),
        "refined_area_ratio": round(refined_area_ratio, 4),
        "extent": round(extent, 4),
        "solidity": round(solidity, 4),
        "final_score": round(float(score), 4),
        "chosen_preset": preset_name,
        "penalty_reason": penalties,
    }


def score_candidate_set(
    image_bgr: np.ndarray,
    params: DetectorParams,
    candidates: list[dict[str, Any]],
    preset_name: str,
    expected_count: int | None,
) -> tuple[float, list[dict[str, Any]]]:
    details = [candidate_score_details(image_bgr, params, cand, preset_name) for cand in candidates]
    total = sum(float(item["final_score"]) for item in details)
    if expected_count is not None:
        total -= abs(len(candidates) - expected_count) * 45.0
    return float(total), details


def auto_tune_detect(
    image_bgr: np.ndarray,
    base_params: DetectorParams,
    expected_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preset_results: dict[str, dict[str, Any]] = {}
    all_candidates: list[dict[str, Any]] = []
    for name, params in auto_tune_presets(base_params).items():
        candidates, debug = detect(image_bgr, params)
        preset_score, contour_scores = score_candidate_set(image_bgr, params, candidates, name, expected_count)
        for idx, cand in enumerate(candidates):
            cand["chosen_preset"] = name
            cand["final_score"] = float(contour_scores[idx]["final_score"])
            cand["score_details"] = contour_scores[idx]
            all_candidates.append(cand)
        preset_results[name] = {
            "params": asdict(params),
            "candidates": candidates,
            "debug": debug,
            "score": preset_score,
            "contour_scores": contour_scores,
        }

    blended = nms_rotated(all_candidates, base_params.nms_iou)
    if expected_count is not None and expected_count > 0 and len(blended) > expected_count:
        blended = sorted(blended, key=lambda c: float(c.get("final_score", c.get("score", 0.0))), reverse=True)[:expected_count]
    blended = sorted(blended, key=lambda c: (c["rect"][0][1], c["rect"][0][0]))
    blended_score, blended_details = score_candidate_set(image_bgr, base_params, blended, "mixed", expected_count)
    for idx, cand in enumerate(blended):
        cand["final_score"] = float(blended_details[idx]["final_score"])
        cand["score_details"] = blended_details[idx] | {"chosen_preset": cand.get("chosen_preset", "mixed")}

    best_preset_name = max(preset_results, key=lambda key: preset_results[key]["score"])
    best_preset = preset_results[best_preset_name]
    if blended_score >= float(best_preset["score"]):
        best_candidates = blended
        best_name = "mixed"
        best_score = blended_score
    else:
        best_candidates = best_preset["candidates"]
        best_name = best_preset_name
        best_score = float(best_preset["score"])

    scores_payload = {
        "best_result": best_name,
        "best_score": round(float(best_score), 4),
        "expected_count": expected_count,
        "presets": {
            name: {
                "score": round(float(result["score"]), 4),
                "count": len(result["candidates"]),
                "contours": result["contour_scores"],
            }
            for name, result in preset_results.items()
        },
        "mixed": {
            "score": round(float(blended_score), 4),
            "count": len(blended),
            "contours": [
                cand.get("score_details", {}) | {"chosen_preset": cand.get("chosen_preset", "mixed")}
                for cand in blended
            ],
        },
    }
    debug = {
        "preset_results": preset_results,
        "scores": scores_payload,
        "best_name": best_name,
        "best_score": best_score,
    }
    return best_candidates, debug


def save_debug_images(out_dir: Path, stem: str, timestamp: str, debug: dict[str, np.ndarray]) -> None:
    for name, arr in debug.items():
        if not isinstance(arr, np.ndarray):
            continue
        cv2.imwrite(str(out_dir / f"{stem}_debug_{name}_{timestamp}.png"), arr)


def save_auto_tune_outputs(
    out_dir: Path,
    image_bgr: np.ndarray,
    preset_results: dict[str, dict[str, Any]],
    best_candidates: list[dict[str, Any]],
    min_long_px: float | None,
    max_long_px: float | None,
    save_presets: bool,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if save_presets:
        for name, result in preset_results.items():
            out_path = out_dir / f"detected_{name}.png"
            cv2.imwrite(str(out_path), draw_result(image_bgr, result["candidates"], min_long_px, max_long_px))
            outputs[f"detected_{name}"] = str(out_path)
    best_path = out_dir / "detected_best.png"
    cv2.imwrite(str(best_path), draw_result(image_bgr, best_candidates, min_long_px, max_long_px))
    outputs["detected_best"] = str(best_path)
    return outputs


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
    parser.add_argument("--min-extent", type=float, default=0.05)
    parser.add_argument("--min-solidity", type=float, default=0.08)
    parser.add_argument("--merge-distance", type=float, default=45.0)
    parser.add_argument("--min-edge-density", type=float, default=0.012)
    parser.add_argument("--max-seed-expansion-ratio", type=float, default=3.0)
    parser.add_argument("--hard-seed-expansion-ratio", type=float, default=4.0)
    parser.add_argument("--refine-seed-ratio", type=float, default=2.0)
    parser.add_argument("--max-seed-distance", type=int, default=35)
    parser.add_argument("--refine-far-seed", type=int, default=15)
    parser.add_argument("--refine-close", type=int, default=9)
    parser.add_argument("--reflection-refine-low-sat-max", type=int, default=35)
    parser.add_argument("--reflection-refine-v-min", type=int, default=80)
    parser.add_argument("--reflection-refine-max-gradient", type=float, default=8.0)
    parser.add_argument("--reflection-refine-max-local-std", type=float, default=5.0)
    parser.add_argument("--reflection-max-gradient", type=float, default=10.0)
    parser.add_argument("--reflection-max-local-std", type=float, default=5.0)
    parser.add_argument("--dark-side-delta-l", type=float, default=8.0)
    parser.add_argument("--dark-side-min-gradient", type=float, default=22.0)
    parser.add_argument("--side-seed-distance", type=int, default=75)
    parser.add_argument("--min-support-gradient", type=float, default=24.0)
    parser.add_argument("--min-support-local-std", type=float, default=20.0)
    parser.add_argument("--delta-cc-min-area", type=int, default=50)
    parser.add_argument("--delta-border-cc-min-area", type=int, default=300)

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
    parser.add_argument("--px-per-cm", type=float, default=None, help="흰색 지그재그 한 칸(1cm)의 픽셀 길이")
    parser.add_argument("--auto-scale", action="store_true", help="상단/좌측 흰색 지그재그 패턴에서 px_per_cm 자동 추정")
    parser.add_argument("--size-min-mm", type=float, default=4.75, help="골재 Breadth PASS 최소값(mm)")
    parser.add_argument("--size-max-mm", type=float, default=25.0, help="골재 Breadth PASS 최대값(mm)")
    parser.add_argument("--ratio-pass-max", type=float, default=3.0, help="Length/Breadth PASS 최대 비율")
    parser.add_argument("--auto-tune", action="store_true", help="run multiple presets and select the best scored result")
    parser.add_argument("--expected-count", type=int, default=None, help="optional expected number of stones for auto tuning")
    parser.add_argument("--save-presets", action="store_true", help="save detected_<preset>.png images when --auto-tune is used")
    parser.add_argument("--score-debug", action="store_true", help="save contour score details to scores.json")
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
        min_extent=args.min_extent,
        min_solidity=args.min_solidity,
        merge_distance=args.merge_distance,
        min_edge_density=args.min_edge_density,
        max_seed_expansion_ratio=args.max_seed_expansion_ratio,
        hard_seed_expansion_ratio=args.hard_seed_expansion_ratio,
        refine_seed_ratio=args.refine_seed_ratio,
        max_seed_distance_px=args.max_seed_distance,
        refine_far_seed_px=args.refine_far_seed,
        refine_close_px=args.refine_close,
        reflection_refine_low_sat_max=args.reflection_refine_low_sat_max,
        reflection_refine_v_min=args.reflection_refine_v_min,
        reflection_refine_max_gradient=args.reflection_refine_max_gradient,
        reflection_refine_max_local_std=args.reflection_refine_max_local_std,
        reflection_max_gradient=args.reflection_max_gradient,
        reflection_max_local_std=args.reflection_max_local_std,
        dark_side_delta_l=args.dark_side_delta_l,
        dark_side_min_gradient=args.dark_side_min_gradient,
        side_seed_distance_px=args.side_seed_distance,
        min_support_gradient=args.min_support_gradient,
        min_support_local_std=args.min_support_local_std,
        delta_cc_min_area_px=args.delta_cc_min_area,
        delta_border_cc_min_area_px=args.delta_border_cc_min_area,
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = image_path.stem

    px_per_cm = args.px_per_cm
    scale_source: str | None = "manual" if px_per_cm is not None else None
    if args.auto_scale:
        try:
            px_per_cm = estimate_px_per_cm_from_zigzag(
                image,
                debug_dir=out_dir if args.debug else None,
                stem=stem,
            )
            scale_source = "auto_zigzag"
            print(f"auto_scale px_per_cm={px_per_cm:.4f}")
        except Exception as e:
            raise RuntimeError(
                f"자동 스케일 추정 실패: {e}\n"
                f"사진의 지그재그 한 칸 픽셀 길이를 직접 재서 --px-per-cm 값으로 입력하세요."
            ) from e

    if px_per_cm is None:
        raise ValueError(
            "품질 판정을 위해 --px-per-cm 또는 --auto-scale이 필요합니다. "
            "예: --px-per-cm 80"
        )

    mm_per_px = compute_scale_mm_per_px(px_per_cm)

    auto_debug: dict[str, Any] | None = None
    if args.auto_tune:
        candidates, auto_debug = auto_tune_detect(image, params, args.expected_count)
        debug = {}
    else:
        candidates, debug = detect(image, params)

    apply_quality_to_candidates(
        candidates,
        mm_per_px,
        args.size_min_mm,
        args.size_max_mm,
        args.ratio_pass_max,
    )
    if auto_debug is not None and mm_per_px is not None:
        for result in auto_debug["preset_results"].values():
            apply_quality_to_candidates(
                result["candidates"],
                mm_per_px,
                args.size_min_mm,
                args.size_max_mm,
                args.ratio_pass_max,
            )

    annotated = draw_result(image, candidates, args.min_long_px, args.max_long_px)
    out_img = out_dir / f"{stem}_detected_{timestamp}.png"
    out_json = out_dir / f"{stem}_results_refined_{timestamp}.json"
    cv2.imwrite(str(out_img), annotated)

    stones = serialize_candidates(candidates, args.min_long_px, args.max_long_px)
    if mm_per_px is not None:
        pass_count = sum(1 for s in stones if bool(s.get("final_pass", False)))
    else:
        pass_count = sum(1 for s in stones if s["status"] == "PASS")
    fail_count = len(stones) - pass_count

    payload = {
        "image": str(image_path),
        "timestamp": timestamp,
        "params": asdict(params),
        "px_per_cm": round(float(px_per_cm), 4) if px_per_cm is not None else None,
        "mm_per_px": round(float(mm_per_px), 6) if mm_per_px is not None else None,
        "scale_source": scale_source,
        "quality_criteria": {
            "size_min_mm": args.size_min_mm,
            "size_max_mm": args.size_max_mm,
            "ratio_pass_max": args.ratio_pass_max,
        },
        "auto_tune": bool(args.auto_tune),
        "expected_count": args.expected_count,
        "total_stones": len(stones),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "stones": stones,
        "output_image": str(out_img),
    }
    if args.auto_tune and auto_debug is not None:
        auto_outputs = save_auto_tune_outputs(
            out_dir,
            image,
            auto_debug["preset_results"],
            candidates,
            args.min_long_px,
            args.max_long_px,
            args.save_presets,
        )
        payload["auto_tune_best"] = auto_debug["best_name"]
        payload["auto_tune_best_score"] = round(float(auto_debug["best_score"]), 4)
        payload["auto_tune_outputs"] = auto_outputs
        if args.score_debug:
            scores_path = out_dir / "scores.json"
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(auto_debug["scores"], f, ensure_ascii=False, indent=2)
            payload["scores_json"] = str(scores_path)
    else:
        payload["rejected_candidates"] = debug.get("rejected_candidates", [])
        payload["refine_logs"] = debug.get("refine_logs", [])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.debug and not args.auto_tune:
        save_debug_images(out_dir, stem, timestamp, debug)

    print(f"total_stones={len(stones)} pass={pass_count} fail={fail_count}")
    print(f"image={out_img}")
    print(f"json={out_json}")


if __name__ == "__main__":
    main()
