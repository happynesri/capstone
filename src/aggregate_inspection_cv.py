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

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        bk = max(3, self.blur_kernel)
        if bk % 2 == 0:
            bk += 1

        # 경계 보존을 위해 Gaussian 대신 bilateral 사용
        smooth = cv2.bilateralFilter(gray, 9, 50, 50)

        # canny edge
        edges = cv2.Canny(smooth, 40, 120)

        mk = max(3, self.morph_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))

        # 끊긴 edge를 조금 연결
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        roi_mask = self._build_roi_mask(edges.shape[0], edges.shape[1])
        edges = cv2.bitwise_and(edges, roi_mask)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_edge = original.copy()
        scale = self.process_scale if self.process_scale > 0 else 1.0

        edge_stones = []
        stone_id = 0

        for contour in contours:
            area_proc = cv2.contourArea(contour)

            if area_proc < self.min_contour_area or area_proc > self.max_contour_area:
                continue

            rect = cv2.minAreaRect(contour)
            (cx_proc, cy_proc), (w_proc, h_proc), angle = rect

            if w_proc < 1 or h_proc < 1:
                continue

            long_axis_proc = max(w_proc, h_proc)
            short_axis_proc = min(w_proc, h_proc)
            aspect_ratio = long_axis_proc / max(short_axis_proc, 1e-6)

            area_px = area_proc / (scale * scale)
            cx = cx_proc / scale
            cy = cy_proc / scale
            long_axis = long_axis_proc / scale
            short_axis = short_axis_proc / scale

            stone_id += 1

            contour_draw = contour.astype(np.float32)
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
            "edges": edges,
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
        process_scale=1.0,
        roi_top=0,
        roi_bottom=0,
        roi_left=0,
        roi_right=0,
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
        self.process_scale = process_scale
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_left = roi_left
        self.roi_right = roi_right

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

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        debug = original.copy()
        scale = self.process_scale if self.process_scale > 0 else 1.0

        stones = []
        stone_id = 0

        for contour in contours:
            area_proc = cv2.contourArea(contour)

            if area_proc < self.min_contour_area or area_proc > self.max_contour_area:
                continue

            rect = cv2.minAreaRect(contour)
            (cx_proc, cy_proc), (w_proc, h_proc), angle = rect

            if w_proc < 1 or h_proc < 1:
                continue

            long_axis_proc = max(w_proc, h_proc)
            short_axis_proc = min(w_proc, h_proc)
            aspect_ratio = long_axis_proc / max(short_axis_proc, 1e-6)

            # 원래 해상도로 환산
            area_px = area_proc / (scale * scale)
            cx = cx_proc / scale
            cy = cy_proc / scale
            long_axis = long_axis_proc / scale
            short_axis = short_axis_proc / scale

            is_pass = (
                self.long_axis_min <= long_axis <= self.long_axis_max and
                self.short_axis_min <= short_axis <= self.short_axis_max and
                aspect_ratio <= self.aspect_ratio_max
            )

            stone_id += 1

            # contour draw를 위해 원래 스케일로 복원
            contour_draw = contour.astype(np.float32)
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
            })

        return {
            "debug": debug,
            "gray": gray,
            "binary": binary,
            "stones": stones
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


def save_results(image_path, result, output_dir="../result"):
    """
    이미지 처리 결과를 result 폴더에 저장합니다.
    
    Args:
        image_path: 입력 이미지 경로
        result: inspect() 함수의 반환값 dictionary
        output_dir: 결과 저장 폴더 (기본값: ../result)
    """
    # 결과 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 입력 파일명 추출 (확장자 제거)
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 파일 이름들
    debug_file = os.path.join(output_dir, f"{image_name}_debug_{timestamp}.png")
    binary_file = os.path.join(output_dir, f"{image_name}_binary_{timestamp}.png")
    gray_file = os.path.join(output_dir, f"{image_name}_gray_{timestamp}.png")
    json_file = os.path.join(output_dir, f"{image_name}_results_{timestamp}.json")
    
    # 이미지 저장
    cv2.imwrite(debug_file, result["debug"])
    cv2.imwrite(binary_file, result["binary"])
    cv2.imwrite(gray_file, result["gray"])
    print(f"✓ 이미지 저장 완료: {debug_file}")
    print(f"✓ 이미지 저장 완료: {binary_file}")
    print(f"✓ 이미지 저장 완료: {gray_file}")
    
    # 결과 JSON 저장
    output_data = {
        "image_path": str(image_path),
        "timestamp": timestamp,
        "total_stones": len(result["stones"]),
        "pass_count": sum(1 for s in result["stones"] if s["pass"]),
        "fail_count": sum(1 for s in result["stones"] if not s["pass"]),
        "stones": result["stones"]
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 결과 저장 완료: {json_file}")


def run_on_image(image_path, inspector, use_gui=True):
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
    args = parser.parse_args()

    inspector = AggregateInspector(
        min_contour_area=1000,
        max_contour_area=1000000,
        long_axis_min=30,
        long_axis_max=300,
        short_axis_min=20,
        short_axis_max=200,
        aspect_ratio_max=4.0,
        threshold_mode=1,
        use_otsu=True,
        binary_thresh=100,
        morph_kernel=5,
        blur_kernel=5,
        process_scale=1.0,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
        roi_left=args.roi_left,
        roi_right=args.roi_right,
    )

    use_gui = not args.nogui

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


if __name__ == "__main__":
    main()