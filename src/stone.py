import cv2
import numpy as np
import os

# =========================
# 설정값
# =========================
IMAGE_PATH = "stone.jpg"   # 원본 이미지 경로로 바꿔줘
SAVE_DIR = "output_stone"
CROP_BORDER = 20           # 바깥 흰/검은 여백 제거용, 필요 없으면 0
BG_SIGMA = 51              # 배경 조명 추정용 blur 크기
DIFF_THRESH = 15           # 돌이 덜 잡히면 13~14, 노이즈 많으면 17~18
MIN_AREA = 10000           # 너무 작은 잡영 제거
TEXT_SCALE = 0.8

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 이미지 읽기
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없음: {IMAGE_PATH}")

orig = img.copy()

# 바깥 테두리 제거
if CROP_BORDER > 0:
    img = img[CROP_BORDER:-CROP_BORDER, CROP_BORDER:-CROP_BORDER].copy()

h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 1) 조명 보정
# =========================
# 큰 blur를 배경 조명으로 보고, 원본에서 빼서 돌을 강조
bg = cv2.GaussianBlur(gray, (0, 0), BG_SIGMA)
detail = cv2.subtract(gray, bg)

# 약간 부드럽게
detail = cv2.GaussianBlur(detail, (5, 5), 0)

# =========================
# 2) threshold
# =========================
_, mask_raw = cv2.threshold(detail, DIFF_THRESH, 255, cv2.THRESH_BINARY)

# =========================
# 3) 같은 돌 내부 끊김 연결
# =========================
mask = cv2.medianBlur(mask_raw, 5)

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

mask = cv2.dilate(mask, kernel_dilate, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# =========================
# 4) 연결요소 분석
# =========================
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

clean_mask = np.zeros_like(mask)
result = img.copy()

stone_count = 0
stone_info = []

for i in range(1, num_labels):
    x, y, ww, hh, area = stats[i]

    # 작은 잡영 제거
    if area < MIN_AREA:
        continue

    # 가장자리 붙은 영역 제거
    # (오른쪽 위 밝은 코너 같은 것 제거용)
    margin = 3
    if x <= margin or y <= margin or (x + ww) >= (w - margin) or (y + hh) >= (h - margin):
        continue

    comp = np.zeros_like(mask)
    comp[labels == i] = 255

    # 외곽 contour 하나만 사용
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    cnt = max(contours, key=cv2.contourArea)

    # 너무 찌꺼기면 제거
    if cv2.contourArea(cnt) < MIN_AREA:
        continue

    # 마스크에 채우기
    cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 길이/폭 계산
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    long_side = max(rw, rh)
    short_side = min(rw, rh)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    stone_count += 1
    stone_info.append((stone_count, long_side, short_side, int(cx), int(cy)))

    # 시각화
    cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)   # 초록 외곽
    cv2.polylines(result, [box], True, (255, 0, 0), 2)    # 파란 최소회전박스

    label = f"ID {stone_count} L={long_side:.1f} S={short_side:.1f}"
    tx = max(int(cx) - 100, 10)
    ty = max(int(cy), 25)
    cv2.putText(result, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 255), 2, cv2.LINE_AA)

# =========================
# 결과 저장
# =========================
cv2.imwrite(os.path.join(SAVE_DIR, "01_gray.png"), gray)
cv2.imwrite(os.path.join(SAVE_DIR, "02_background.png"), bg)
cv2.imwrite(os.path.join(SAVE_DIR, "03_detail.png"), detail)
cv2.imwrite(os.path.join(SAVE_DIR, "04_mask_raw.png"), mask_raw)
cv2.imwrite(os.path.join(SAVE_DIR, "05_mask_connected.png"), mask)
cv2.imwrite(os.path.join(SAVE_DIR, "06_mask_clean.png"), clean_mask)
cv2.imwrite(os.path.join(SAVE_DIR, "07_result.png"), result)

print(f"[완료] 검출된 골재 개수: {stone_count}")
for sid, L, S, cx, cy in stone_info:
    print(f"ID {sid}: Long={L:.1f}px, Short={S:.1f}px, Center=({cx}, {cy})")