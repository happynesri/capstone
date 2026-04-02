import cv2
import numpy as np

# =========================
# 설정
# =========================
IMAGE_PATH = "stone.png"   # 원본 이미지 파일명으로 수정
MIN_AREA = 10000           # 너무 작은 잡영 제거
BORDER_MARGIN = 5          # 이미지 가장자리 붙은 영역 제거
OPEN_K = 5
CLOSE_K = 41               # 5개가 따로 안 잡히면 33~41 사이 조절
FINAL_CLOSE_K = 21
BG_SIGMA = 35
TEXT_SCALE = 0.8

# =========================
# 이미지 읽기
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

result = img.copy()
h, w = img.shape[:2]

# =========================
# 1. 밝기 보정 + 로컬 대비 강조
# =========================
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
L_eq = clahe.apply(L)

# 큰 blur로 배경 조명을 추정
bg = cv2.GaussianBlur(L_eq, (0, 0), BG_SIGMA)

# 배경을 빼서 돌이 있는 부분만 상대적으로 강조
detail = cv2.subtract(L_eq, bg)
detail_norm = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)

# =========================
# 2. 채도 정보도 약하게 반영
#    돌은 상대적으로 저채도 회색 계열이라서
#    (255 - S)를 섞으면 배경 분리에 도움됨
# =========================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)
inv_sat = 255 - S

mix = cv2.addWeighted(detail_norm, 0.8, inv_sat, 0.2, 0)
mix = cv2.GaussianBlur(mix, (5, 5), 0)

# =========================
# 3. threshold
# =========================
_, raw_mask = cv2.threshold(mix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# =========================
# 4. morphology
# =========================
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FINAL_CLOSE_K, FINAL_CLOSE_K))

mask_open = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
mask_morph = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

# =========================
# 5. 연결 요소 분석
#    - 작은 잡영 제거
#    - 가장자리 붙은 영역 제거
# =========================
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_morph, connectivity=8)

clean_mask = np.zeros_like(mask_morph)

for i in range(1, num_labels):
    x, y, ww, hh, area = stats[i]

    if area < MIN_AREA:
        continue

    # 이미지 가장자리 붙은 큰 번짐 제거
    if x <= BORDER_MARGIN or y <= BORDER_MARGIN or (x + ww) >= (w - BORDER_MARGIN) or (y + hh) >= (h - BORDER_MARGIN):
        continue

    clean_mask[labels == i] = 255

# 한 번 더 닫아줘서 끊긴 경계 보정
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_final)

# =========================
# 6. contour 추출 및 결과 표시
# =========================
contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 중심 기준으로 위->아래, 좌->우 정렬
items = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    items.append((cy, cx, cnt, rect))

items.sort(key=lambda x: (x[0], x[1]))

final_mask = np.zeros_like(clean_mask)

stone_count = 0
for _, _, cnt, rect in items:
    stone_count += 1

    cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    (cx, cy), (rw, rh), angle = rect
    long_side = max(rw, rh)
    short_side = min(rw, rh)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # 초록 외곽선
    cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
    # 파란 최소 회전 박스
    cv2.polylines(result, [box], True, (255, 0, 0), 2)

    label = f"ID {stone_count} L={long_side:.1f} S={short_side:.1f}"
    tx = max(int(cx) - 100, 10)
    ty = max(int(cy), 30)
    cv2.putText(result, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 255), 2, cv2.LINE_AA)

    print(f"ID {stone_count}: Long={long_side:.1f}px, Short={short_side:.1f}px")

print(f"\n검출된 골재 개수: {stone_count}")

# =========================
# 7. 디버그 이미지 저장
# =========================
cv2.imwrite("debug_low_mask.png", raw_mask)     # threshold 직후
cv2.imwrite("debug_high_mask.png", mask_morph)  # morphology 후
cv2.imwrite("debug_mask.png", final_mask)       # 최종 mask
cv2.imwrite("stone_result.png", result)         # 최종 결과

print("저장 완료:")
print("- debug_low_mask.png")
print("- debug_high_mask.png")
print("- debug_mask.png")
print("- stone_result.png")