#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import json
import os
from pathlib import Path
from datetime import datetime


def nothing(x):
    pass


class AggregateInspector:

    def _build_stone_mask(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        L_eq = clahe.apply(L)

        bg_sigma = max(25, int(self.blur_kernel * 7))
        bg = cv2.GaussianBlur(L_eq, (0, 0), bg_sigma)
        detail = cv2.subtract(L_eq, bg)
        detail_norm = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        inv_sat = 255 - hsv[:, :, 1]

        mix = cv2.addWeighted(detail_norm, 0.8, inv_sat, 0.2, 0)
        mix = cv2.GaussianBlur(mix, (5, 5), 0)

        _, mask = cv2.threshold(mix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_k = max(21, self.morph_kernel * 4 + 1)
        if close_k % 2 == 0:
            close_k += 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        return mask

    def _refine_binary_mask(self, binary, min_area, border_margin=0):
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mk = max(3, self.morph_kernel)
        if mk % 2 == 0:
            mk += 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))

        mask = cv2.medianBlur(binary, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        refined = np.zeros_like(mask)
        h, w = mask.shape[:2]

        for label_idx in range(1, num_labels):
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            ww = stats[label_idx, cv2.CC_STAT_WIDTH]
            hh = stats[label_idx, cv2.CC_STAT_HEIGHT]
            area = stats[label_idx, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            if border_margin > 0:
                if x <= border_margin or y <= border_margin:
                    continue
                if (x + ww) >= (w - border_margin) or (y + hh) >= (h - border_margin):
                    continue

            refined[labels == label_idx] = 255

        return refined

    def _smooth_contour(self, contour):
        if contour is None or len(contour) < 5:
            return contour

        peri = cv2.arcLength(contour, True)
        epsilon = max(2.0, 0.02 * peri)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx is None or len(approx) < 3:
            return contour
        return cv2.convexHull(approx)

    
    def detect_stone_edges(self, frame):
        original = frame.copy()

        if self.process_scale < 0.999:
            proc = cv2.resize(
                frame,
                None,
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            proc = frame.copy()

        edges = self._build_stone_mask(proc)

        roi_mask = self._build_roi_mask(edges.shape[0], edges.shape[1])
        edges = cv2.bitwise_and(edges, roi_mask)

        edges = self._refine_binary_mask(
            edges,
            min_area=max(250, int(self.min_contour_area * 0.3)),
            border_margin=3,
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_edge = original.copy()
        clean_edge_map = np.zeros_like(edges)
        scale = self.process_scale if self.process_scale > 0 else 1.0

        edge_stones = []
        stone_id = 0

        candidates = []
        for contour in contours:
            contour_smooth = self._smooth_contour(contour)
            if contour_smooth is None:
                contour_smooth = contour

            area_proc = cv2.contourArea(contour_smooth)
            if area_proc < self.min_contour_area or area_proc > self.max_contour_area:
                continue

            rect = cv2.minAreaRect(contour_smooth)
            (cx_proc, cy_proc), (w_proc, h_proc), angle = rect

            if w_proc < 1 or h_proc < 1:
                continue

            long_axis_proc = max(w_proc, h_proc)
            short_axis_proc = min(w_proc, h_proc)
            aspect_ratio = long_axis_proc / max(short_axis_proc, 1e-6)

            hull = cv2.convexHull(contour_smooth)
            hull_area = cv2.contourArea(hull)
            solidity = area_proc / max(hull_area, 1e-6)

            if short_axis_proc < self.edge_short_axis_min:
                continue
            if aspect_ratio > self.edge_aspect_ratio_max:
                continue
            if solidity < self.edge_min_solidity:
                continue

            candidates.append((
                area_proc,
                contour_smooth,
                rect,
                aspect_ratio,
                angle,
                cx_proc,
                cy_proc,
                long_axis_proc,
                short_axis_proc,
            ))

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:self.edge_keep_top_k]

        for (
            area_proc,
            contour_draw,
            rect,
            aspect_ratio,
            angle,
            cx_proc,
            cy_proc,
            long_axis_proc,
            short_axis_proc,
        ) in candidates:

            area_px = area_proc / (scale * scale)
            cx = cx_proc / scale
            cy = cy_proc / scale
            long_axis = long_axis_proc / scale
            short_axis = short_axis_proc / scale

            stone_id += 1

            cv2.drawContours(clean_edge_map, [contour_draw], -1, 255, 2)

            contour_draw = contour_draw.astype(np.float32)
            contour_draw[:, 0, 0] /= scale
            contour_draw[:, 0, 1] /= scale
            contour_draw = contour_draw.astype(np.int32)

            box = cv2.boxPoints(rect)
            box[:, 0] /= scale
            box[:, 1] /= scale
            box = box.astype(np.int32)

            # 노란색 외곽선 + 파란 박스
            cv2.drawContours(debug_edge, [contour_draw], -1, (0, 255, 255), 2)
            cv2.polylines(debug_edge, [box], True, (255, 0, 0), 2)

            label = f'EDGE ID:{stone_id} L:{long_axis:.1f} S:{short_axis:.1f}'
            tx = max(0, int(cx) - 70)
            ty = max(20, int(cy) - 10)

            cv2.putText(
                debug_edge, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
            )

            edge_stones.append({
                "id": stone_id,
                "center_x": float(cx),
                "center_y": float(cy),
                "area_px": float(area_px),
                "long_axis_px": float(long_axis),
                "short_axis_px": float(short_axis),
                "aspect_ratio": float(aspect_ratio),
                "angle_deg": float(angle),
            })

        return {
            "edges": clean_edge_map,
            "debug_edge": debug_edge,
            "edge_stones": edge_stones
        }
    
    def __init__(
        self,
        min_contour_area=1000,
        max_contour_area=1000000,
        long_axis_min=30,
        long_axis_max=300,
        short_axis_min=20,
        short_axis_max=200,
        aspect_ratio_max=4.0,
        threshold_mode=0,   # 0: THRESH_BINARY, 1: THRESH_BINARY_INV
        use_otsu=True,
        binary_thresh=100,
        morph_kernel=5,
        blur_kernel=5,
        edge_short_axis_min=28,
        edge_aspect_ratio_max=6.0,
        edge_min_solidity=0.82,
        edge_keep_top_k=8,
        process_scale=1.0,
        roi_top=0,
        roi_bottom=0,
        roi_left=0,
        roi_right=0,
        min_delta_l=18.0,
        min_extent=0.40,
        min_solidity_filter=0.60,
        ring_px=20,
        margin_px=40,
    ):
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.long_axis_min = long_axis_min
        self.long_axis_max = long_axis_max
        self.short_axis_min = short_axis_min
        self.short_axis_max = short_axis_max
        self.aspect_ratio_max = aspect_ratio_max
        self.threshold_mode = threshold_mode
        self.use_otsu = use_otsu
        self.binary_thresh = binary_thresh
        self.morph_kernel = morph_kernel
        self.blur_kernel = blur_kernel
        self.edge_short_axis_min = edge_short_axis_min
        self.edge_aspect_ratio_max = edge_aspect_ratio_max
        self.edge_min_solidity = edge_min_solidity
        self.edge_keep_top_k = edge_keep_top_k
        self.process_scale = process_scale
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_left = roi_left
        self.roi_right = roi_right
        self.min_delta_l = min_delta_l
        self.min_extent = min_extent
        self.min_solidity_filter = min_solidity_filter
        self.ring_px = ring_px
        self.margin_px = margin_px

    def _build_roi_mask(self, height, width):
        top = max(0, int(self.roi_top))
        bottom = max(0, int(self.roi_bottom))
        left = max(0, int(self.roi_left))
        right = max(0, int(self.roi_right))

        y1 = top
        y2 = max(y1 + 1, height - bottom)
        x1 = left
        x2 = max(x1 + 1, width - right)

        y2 = min(y2, height)
        x2 = min(x2, width)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    def preprocess(self, frame):
        if self.process_scale < 0.999:
            proc = cv2.resize(
                frame,
                None,
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            proc = frame.copy()

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        bk = max(3, self.blur_kernel)
        if bk % 2 == 0:
            bk += 1
        gray = cv2.GaussianBlur(gray, (bk, bk), 0)

        if self.use_otsu:
            if self.threshold_mode == 0:
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
        else:
            if self.threshold_mode == 0:
                _, binary = cv2.threshold(
                    gray, self.binary_thresh, 255, cv2.THRESH_BINARY
                )
            else:
                _, binary = cv2.threshold(
                    gray, self.binary_thresh, 255, cv2.THRESH_BINARY_INV
                )

        mk = max(3, self.morph_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        return proc, gray, binary

    def inspect(self, frame):
        original = frame.copy()
        proc, gray, binary = self.preprocess(frame)

        roi_mask = self._build_roi_mask(binary.shape[0], binary.shape[1])
        binary = cv2.bitwise_and(binary, roi_mask)

        binary = self._refine_binary_mask(
            binary,
            min_area=max(250, int(self.min_contour_area * 0.3)),
            border_margin=3,
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        debug = original.copy()
        debug_rejected = original.copy()
        scale = self.process_scale if self.process_scale > 0 else 1.0

        # LAB 색공간에서 L 채널 추출
        lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
        L_channel = lab[:, :, 1].astype(np.float32)

        stones = []
        rejected_stones = []
        stone_id = 0
        rejected_id = 0

        # ROI 내부 영역 계산
        top = max(0, int(self.roi_top))
        bottom = max(0, int(self.roi_bottom))
        left = max(0, int(self.roi_left))
        right = max(0, int(self.roi_right))
        roi_h = binary.shape[0] - top - bottom
        roi_w = binary.shape[1] - left - right
        roi_x1, roi_y1 = left, top
        roi_x2, roi_y2 = left + roi_w, top + roi_h

        for contour in contours:
            area_proc = cv2.contourArea(contour)

            if area_proc < self.min_contour_area or area_proc > self.max_contour_area:
                continue

            rect = cv2.minAreaRect(contour)
            (cx_proc, cy_proc), (w_proc, h_proc), angle = rect

            if w_proc < 1 or h_proc < 1:
                continue

            contour_draw = self._smooth_contour(contour)
            if contour_draw is None:
                contour_draw = contour

            long_axis_proc = max(w_proc, h_proc)
            short_axis_proc = min(w_proc, h_proc)
            aspect_ratio = long_axis_proc / max(short_axis_proc, 1e-6)

            # 원래 해상도로 환산
            area_px = area_proc / (scale * scale)
            cx = cx_proc / scale
            cy = cy_proc / scale
            long_axis = long_axis_proc / scale
            short_axis = short_axis_proc / scale

            # ===== 후처리 필터 1: ROI 가장자리 제거 =====
            margin_scaled = self.margin_px / scale
            if (cx < roi_x1 + margin_scaled or cy < roi_y1 + margin_scaled or
                cx > roi_x2 - margin_scaled or cy > roi_y2 - margin_scaled):
                rejected_id += 1
                rejected_stones.append(({
                    "id": rejected_id,
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "reason": "margin"
                }))
                continue

            # ===== 후처리 필터 2: extent (contour area / minAreaRect area) =====
            rect_area = w_proc * h_proc
            extent = area_proc / max(rect_area, 1e-6)
            if extent < self.min_extent:
                rejected_id += 1
                rejected_stones.append({
                    "id": rejected_id,
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "reason": "extent"
                })
                continue

            # ===== 후처리 필터 3: solidity (contour area / convex hull area) =====
            hull = cv2.convexHull(contour_draw)
            hull_area = cv2.contourArea(hull)
            solidity = area_proc / max(hull_area, 1e-6)
            if solidity < self.min_solidity_filter:
                rejected_id += 1
                rejected_stones.append({
                    "id": rejected_id,
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "reason": "solidity"
                })
                continue

            # ===== 후처리 필터 4: 내부 vs ring 밝기 차이 =====
            # contour 내부 평균 L값 계산
            mask_contour = np.zeros_like(binary)
            cv2.drawContours(mask_contour, [contour_draw], -1, 255, -1)
            mean_inside_L = cv2.mean(L_channel, mask=mask_contour)[0]

            # ring 영역 (contour 외부 ring_px 거리) 평균 L값 계산
            kernel_dilate_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ring_px * 2 + 1, self.ring_px * 2 + 1))
            mask_dilated_ring = cv2.dilate(mask_contour, kernel_dilate_ring, iterations=1)
            mask_ring = cv2.subtract(mask_dilated_ring, mask_contour)
            
            delta_L = 0
            if cv2.countNonZero(mask_ring) > 0:
                mean_ring_L = cv2.mean(L_channel, mask=mask_ring)[0]
                delta_L = mean_inside_L - mean_ring_L
                # delta_L 검사는 선택사항 (너무 엄격하면 주석 처리)
                # if delta_L < self.min_delta_l:
                #     rejected_id += 1
                #     rejected_stones.append({
                #         "id": rejected_id,
                #         "center_x": float(cx),
                #         "center_y": float(cy),
                #         "reason": "delta_L"
                #     })
                #     continue

            # ===== 기본 조건 검사 =====
            is_pass = (
                self.long_axis_min <= long_axis <= self.long_axis_max and
                self.short_axis_min <= short_axis <= self.short_axis_max and
                aspect_ratio <= self.aspect_ratio_max
            )

            stone_id += 1

            # contour draw를 위해 원래 스케일로 복원
            contour_draw = contour_draw.astype(np.float32)
            contour_draw[:, 0, 0] /= scale
            contour_draw[:, 0, 1] /= scale
            contour_draw = contour_draw.astype(np.int32)

            box = cv2.boxPoints(rect)
            box[:, 0] /= scale
            box[:, 1] /= scale
            box = box.astype(np.int32)

            color = (0, 255, 0) if is_pass else (0, 0, 255)

            cv2.drawContours(debug, [contour_draw], -1, color, 2)
            cv2.polylines(debug, [box], True, color, 2)

            label = (
                f'ID:{stone_id} '
                f'{"PASS" if is_pass else "FAIL"} '
                f'L:{long_axis:.1f} '
                f'S:{short_axis:.1f}'
            )

            tx = max(0, int(cx) - 60)
            ty = max(20, int(cy) - 10)

            cv2.putText(
                debug, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
            )

            stones.append({
                "id": stone_id,
                "center_x": float(cx),
                "center_y": float(cy),
                "area_px": float(area_px),
                "long_axis_px": float(long_axis),
                "short_axis_px": float(short_axis),
                "aspect_ratio": float(aspect_ratio),
                "angle_deg": float(angle),
                "pass": bool(is_pass),
                "extent": float(extent),
                "solidity": float(solidity),
                "delta_L": float(delta_L),
            })

        # 거부된 후보들을 debug_rejected에 빨간색으로 표시
        for rej in rejected_stones:
            cv2.circle(debug_rejected, (int(rej["center_x"]), int(rej["center_y"])), 8, (0, 0, 255), 2)
            cv2.putText(
                debug_rejected, f"REJ:{rej['reason']}", 
                (int(rej["center_x"]) - 40, int(rej["center_y"]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
            )

        return {
            "debug": debug,
            "debug_rejected": debug_rejected,
            "gray": gray,
            "binary": binary,
            "stones": stones,
            "rejected": rejected_stones
        }


def create_trackbars(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("use_otsu", window_name, 1, 1, nothing)
    cv2.createTrackbar("th_mode(0/1)", window_name, 1, 1, nothing)
    cv2.createTrackbar("bin_thresh", window_name, 100, 255, nothing)
    cv2.createTrackbar("blur", window_name, 5, 31, nothing)
    cv2.createTrackbar("morph", window_name, 5, 31, nothing)
    cv2.createTrackbar("min_area", window_name, 1000, 50000, nothing)
    cv2.createTrackbar("long_min", window_name, 30, 1000, nothing)
    cv2.createTrackbar("long_max", window_name, 300, 2000, nothing)
    cv2.createTrackbar("short_min", window_name, 20, 1000, nothing)
    cv2.createTrackbar("short_max", window_name, 200, 2000, nothing)
    cv2.createTrackbar("aspect_x10", window_name, 40, 100, nothing)
    cv2.createTrackbar("scale_x100", window_name, 100, 100, nothing)


def read_trackbars(window_name, inspector):
    inspector.use_otsu = bool(cv2.getTrackbarPos("use_otsu", window_name))
    inspector.threshold_mode = cv2.getTrackbarPos("th_mode(0/1)", window_name)
    inspector.binary_thresh = cv2.getTrackbarPos("bin_thresh", window_name)

    blur = cv2.getTrackbarPos("blur", window_name)
    morph = cv2.getTrackbarPos("morph", window_name)
    inspector.blur_kernel = max(3, blur)
    inspector.morph_kernel = max(3, morph)

    inspector.min_contour_area = max(1, cv2.getTrackbarPos("min_area", window_name))
    inspector.long_axis_min = cv2.getTrackbarPos("long_min", window_name)
    inspector.long_axis_max = cv2.getTrackbarPos("long_max", window_name)
    inspector.short_axis_min = cv2.getTrackbarPos("short_min", window_name)
    inspector.short_axis_max = cv2.getTrackbarPos("short_max", window_name)
    inspector.aspect_ratio_max = max(0.1, cv2.getTrackbarPos("aspect_x10", window_name) / 10.0)

    scale = cv2.getTrackbarPos("scale_x100", window_name) / 100.0
    inspector.process_scale = max(0.1, scale)


def print_stone_info(stones):
    if not stones:
        print("검출된 골재 없음")
        return

    print("=" * 80)
    for s in stones:
        print(
            f'ID={s["id"]} | '
            f'PASS={s["pass"]} | '
            f'center=({s["center_x"]:.1f}, {s["center_y"]:.1f}) | '
            f'area={s["area_px"]:.1f} | '
            f'long={s["long_axis_px"]:.1f} | '
            f'short={s["short_axis_px"]:.1f} | '
            f'AR={s["aspect_ratio"]:.2f} | '
            f'angle={s["angle_deg"]:.1f}'
        )


def save_results(image_path, result, output_dir="../result", suffix=""):
    """
    이미지 처리 결과를 result 폴더에 저장합니다.
    
    Args:
        image_path: 입력 이미지 경로
        result: inspect() 함수의 반환값 dictionary
        output_dir: 결과 저장 폴더 (기본값: ../result)
        suffix: 출력 파일 이름에 추가할 접미사
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{suffix}" if suffix else ""
    debug_file = os.path.join(output_dir, f"{image_name}_debug{suffix}_{timestamp}.png")
    debug_rejected_file = os.path.join(output_dir, f"{image_name}_debug_rejected{suffix}_{timestamp}.png")
    binary_file = os.path.join(output_dir, f"{image_name}_binary{suffix}_{timestamp}.png")
    gray_file = os.path.join(output_dir, f"{image_name}_gray{suffix}_{timestamp}.png")
    json_file = os.path.join(output_dir, f"{image_name}_results{suffix}_{timestamp}.json")
    
    cv2.imwrite(debug_file, result["debug"])
    if "debug_rejected" in result:
        cv2.imwrite(debug_rejected_file, result["debug_rejected"])
        print(f"✓ 이미지 저장 완료: {debug_rejected_file}")
    cv2.imwrite(binary_file, result["binary"])
    cv2.imwrite(gray_file, result["gray"])
    print(f"✓ 이미지 저장 완료: {debug_file}")
    print(f"✓ 이미지 저장 완료: {binary_file}")
    print(f"✓ 이미지 저장 완료: {gray_file}")
    
    output_data = {
        "image_path": str(image_path),
        "timestamp": timestamp,
        "total_stones": len(result["stones"]),
        "pass_count": sum(1 for s in result["stones"] if s["pass"]),
        "fail_count": sum(1 for s in result["stones"] if not s["pass"]),
        "accepted_count": len(result["stones"]),
        "rejected_count": len(result.get("rejected", [])),
        "stones": result["stones"],
        "rejected_candidates": result.get("rejected", [])
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 결과 저장 완료: {json_file}")


DEFAULT_INSPECTOR_PRESETS = {
    "aggressive": {
        "min_contour_area": 180,
        "max_contour_area": 1000000,
        "long_axis_min": 8,
        "long_axis_max": 600,
        "short_axis_min": 4,
        "short_axis_max": 400,
        "aspect_ratio_max": 10.0,
        "threshold_mode": 1,
        "use_otsu": True,
        "binary_thresh": 100,
        "morph_kernel": 1,
        "blur_kernel": 1,
        "process_scale": 1.0,
    },
    "baseline": {
        "min_contour_area": 400,
        "max_contour_area": 20000,
        "long_axis_min": 20,
        "long_axis_max": 500,
        "short_axis_min": 10,
        "short_axis_max": 300,
        "aspect_ratio_max": 5.5,
        "threshold_mode": 1,
        "use_otsu": True,
        "binary_thresh": 100,
        "morph_kernel": 2,
        "blur_kernel": 3,
        "process_scale": 1.0,
        "min_delta_l": 8.0,
        "min_extent": 0.25,
        "min_solidity_filter": 0.45,
        "ring_px": 25,
        "margin_px": 50,
    },
    "wide": {
        "min_contour_area": 700,
        "max_contour_area": 1000000,
        "long_axis_min": 20,
        "long_axis_max": 400,
        "short_axis_min": 12,
        "short_axis_max": 220,
        "aspect_ratio_max": 5.5,
        "threshold_mode": 1,
        "use_otsu": True,
        "binary_thresh": 100,
        "morph_kernel": 3,
        "blur_kernel": 3,
        "process_scale": 1.0,
    },
    "tight": {
        "min_contour_area": 1200,
        "max_contour_area": 1000000,
        "long_axis_min": 40,
        "long_axis_max": 300,
        "short_axis_min": 25,
        "short_axis_max": 200,
        "aspect_ratio_max": 3.5,
        "threshold_mode": 1,
        "use_otsu": True,
        "binary_thresh": 100,
        "morph_kernel": 7,
        "blur_kernel": 7,
        "process_scale": 1.0,
    },
    "binary_low": {
        "min_contour_area": 800,
        "max_contour_area": 1000000,
        "long_axis_min": 25,
        "long_axis_max": 350,
        "short_axis_min": 16,
        "short_axis_max": 220,
        "aspect_ratio_max": 5.0,
        "threshold_mode": 1,
        "use_otsu": False,
        "binary_thresh": 90,
        "morph_kernel": 5,
        "blur_kernel": 5,
        "process_scale": 1.0,
    },
    "binary_high": {
        "min_contour_area": 1200,
        "max_contour_area": 1000000,
        "long_axis_min": 35,
        "long_axis_max": 320,
        "short_axis_min": 20,
        "short_axis_max": 200,
        "aspect_ratio_max": 4.5,
        "threshold_mode": 1,
        "use_otsu": False,
        "binary_thresh": 130,
        "morph_kernel": 7,
        "blur_kernel": 5,
        "process_scale": 1.0,
    },
}


def build_inspector_from_preset(preset_name, roi_top=0, roi_bottom=0, roi_left=0, roi_right=0):
    config = DEFAULT_INSPECTOR_PRESETS.get(preset_name)
    if config is None:
        raise ValueError(f"알 수 없는 preset: {preset_name}")
    return AggregateInspector(
        roi_top=roi_top,
        roi_bottom=roi_bottom,
        roi_left=roi_left,
        roi_right=roi_right,
        **config,
    )


def run_on_image_with_presets(image_path, preset_names, use_gui=False):
    image_path = resolve_image_path(image_path)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    for preset_name in preset_names:
        print(f"\n=== Preset: {preset_name} ===")
        inspector = build_inspector_from_preset(preset_name)
        result = inspector.inspect(frame)
        edge_result = inspector.detect_stone_edges(frame)

        print_stone_info(result["stones"])
        save_results(image_path, result, output_dir="../result", suffix=preset_name)

        edge_output_dir = "../result"
        Path(edge_output_dir).mkdir(parents=True, exist_ok=True)
        image_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        edge_file = os.path.join(edge_output_dir, f"{image_name}_edge_{preset_name}_{timestamp}.png")
        edge_debug_file = os.path.join(edge_output_dir, f"{image_name}_edge_debug_{preset_name}_{timestamp}.png")
        cv2.imwrite(edge_file, edge_result["edges"])
        cv2.imwrite(edge_debug_file, edge_result["debug_edge"])
        print(f"✓ edge 저장 완료: {edge_file}")
        print(f"✓ edge debug 저장 완료: {edge_debug_file}")

        if use_gui:
            cv2.imshow(f"binary_{preset_name}", result["binary"])
            cv2.imshow(f"debug_{preset_name}", result["debug"])
            cv2.waitKey(1)


def resolve_image_path(image_path):
    """경로를 여러 방식으로 시도하여 파일을 찾습니다."""
    import os
    from pathlib import Path
    
    # 1. 입력 경로 그대로 시도
    if os.path.exists(image_path):
        return image_path
    
    # 2. ~ 확장 경로 시도
    expanded = os.path.expanduser(image_path)
    if os.path.exists(expanded):
        return expanded
    
    # 3. 상대 경로 시도 (현재 디렉터리)
    if os.path.exists(image_path):
        return image_path
    
    # 4. 상대 경로 시도 (부모 디렉터리의 data_1)
    base_name = os.path.basename(image_path)
    relative_paths = [
        os.path.join("../data_1", base_name),
        os.path.join("../../data_1", base_name),
        os.path.join("data_1", base_name),
    ]
    for path in relative_paths:
        if os.path.exists(path):
            return path
    
    # 5. /capstone -> /home/sanghwon/capstone 변환 시도
    if image_path.startswith("/capstone/"):
        home_path = f"/home/sanghwon{image_path}"
        if os.path.exists(home_path):
            return home_path
    
    return image_path  # 찾지 못하면 원래 경로 반환


def run_on_image(image_path, inspector, use_gui=True):
    image_path = resolve_image_path(image_path)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    result = inspector.inspect(frame)
    edge_result = inspector.detect_stone_edges(frame)

    print_stone_info(result["stones"])

    save_results(image_path, result, output_dir="../result")

    # edge 결과도 저장
    edge_output_dir = "../result"
    Path(edge_output_dir).mkdir(parents=True, exist_ok=True)
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    edge_file = os.path.join(edge_output_dir, f"{image_name}_edge_{timestamp}.png")
    edge_debug_file = os.path.join(edge_output_dir, f"{image_name}_edge_debug_{timestamp}.png")

    cv2.imwrite(edge_file, edge_result["edges"])
    cv2.imwrite(edge_debug_file, edge_result["debug_edge"])

    print(f"✓ edge 저장 완료: {edge_file}")
    print(f"✓ edge debug 저장 완료: {edge_debug_file}")

    if use_gui:
        cv2.imshow("original", frame)
        cv2.imshow("gray", result["gray"])
        cv2.imshow("binary", result["binary"])
        cv2.imshow("debug", result["debug"])
        cv2.imshow("edges", edge_result["edges"])
        cv2.imshow("edge_debug", edge_result["debug_edge"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(source, inspector, use_gui=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"비디오 소스를 열 수 없습니다: {source}")
        return

    if use_gui:
        create_trackbars("controls")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_gui:
            read_trackbars("controls", inspector)

        result = inspector.inspect(frame)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        debug = result["debug"].copy()
        cv2.putText(
            debug,
            f'FPS: {fps:.1f} | Stones: {len(result["stones"])}',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        if use_gui:
            cv2.imshow("binary", result["binary"])
            cv2.imshow("debug", debug)

        # 콘솔 출력이 너무 많아지지 않도록 q 키 누를 때만 자세히 보거나,
        # 아래 줄 주석 해제해서 매 프레임 출력 가능
        # print_stone_info(result["stones"])

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            print_stone_info(result["stones"])
        elif key == ord('s'):
            cv2.imwrite("debug_snapshot.png", debug)
            cv2.imwrite("binary_snapshot.png", result["binary"])
            print("스냅샷 저장 완료")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="입력 이미지 경로")
    parser.add_argument("--video", type=str, default=None, help="입력 영상 경로")
    parser.add_argument("--camera", type=int, default=None, help="카메라 인덱스 (예: 0)")
    parser.add_argument("--nogui", action="store_true", help="GUI 없이 실행")
    parser.add_argument("--roi-top", type=int, default=0, help="검출 제외 상단 픽셀")
    parser.add_argument("--roi-bottom", type=int, default=0, help="검출 제외 하단 픽셀")
    parser.add_argument("--roi-left", type=int, default=0, help="검출 제외 좌측 픽셀")
    parser.add_argument("--roi-right", type=int, default=0, help="검출 제외 우측 픽셀")
    parser.add_argument(
        "--preset",
        type=str,
        default="baseline",
        choices=list(DEFAULT_INSPECTOR_PRESETS.keys()) + ["all"],
        help="사용할 기본 파라미터 preset"
    )
    parser.add_argument(
        "--batch-default",
        action="store_true",
        help="모든 기본 preset을 순차 실행하여 결과를 비교"
    )
    args = parser.parse_args()

    use_gui = not args.nogui

    if args.batch_default and args.image is None:
        print("--batch-default는 --image와 함께 사용해야 합니다.")
        return

    if args.batch_default or args.preset == "all":
        preset_names = list(DEFAULT_INSPECTOR_PRESETS.keys())
        run_on_image_with_presets(args.image, preset_names, use_gui=False)
        return

    inspector = build_inspector_from_preset(
        args.preset,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
        roi_left=args.roi_left,
        roi_right=args.roi_right,
    )

    if args.image is not None:
        run_on_image(args.image, inspector, use_gui=use_gui)
    elif args.video is not None:
        run_on_video(args.video, inspector, use_gui=use_gui)
    elif args.camera is not None:
        run_on_video(args.camera, inspector, use_gui=use_gui)
    else:
        print("사용 예시:")
        print("  python3 aggregate_inspection_cv.py --image /path/to/image.png")
        print("  python3 aggregate_inspection_cv.py --video /path/to/video.mp4")
        print("  python3 aggregate_inspection_cv.py --camera 0")
        print("  python3 aggregate_inspection_cv.py --image /path/to/image.png --batch-default")
        print("  python3 aggregate_inspection_cv.py --image /path/to/image.png --preset wide")


if __name__ == "__main__":
    main()