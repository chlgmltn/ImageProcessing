import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform

def is_clean_background(image, edge_thresh=25000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_count = cv2.countNonZero(edges)
    return edge_count < edge_thresh

# 원본 이미지 로딩
original_image = cv2.imread("recipt_image/recipt1.jpg")
image = original_image.copy()
image = imutils.resize(image, width=500)

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 배경 상태에 따라 흐림과 이진화 순서 선택
if is_clean_background(image):
    print("✅ 깨끗한 배경 감지됨: 이진화 → 흐림")
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    blurred = cv2.GaussianBlur(thresh, (7, 7), 0)
    before_edged = blurred
else:
    print("⚠️ 배경이 복잡함: 흐림 → 이진화")
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    before_edged = thresh

cv2.imshow("Threshold", thresh)

# 엣지 검출
edged = cv2.Canny(before_edged, 50, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 외곽선 추출 및 면적 기준 정렬
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# 사각형 윤곽선 찾기
receiptCnt = None
min_area = image.shape[0] * image.shape[1] * 0.2
max_area = image.shape[0] * image.shape[1] * 0.95

for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000 or area < min_area or area > max_area:
        continue

    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        if 0.4 < aspect_ratio < 2.5:
            receiptCnt = approx
            break

# 윤곽선 없으면 minAreaRect로 보정
if receiptCnt is None:
    print("📦 4-point 윤곽선 없음: minAreaRect 사용")
    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    receiptCnt = np.intp(box)

# 윤곽선 시각화
output = image.copy()
cv2.drawContours(output, [receiptCnt], -1, (255, 0, 0), 3)
cv2.imshow("Receipt Outline", output)

# 원본 이미지 기준으로 투시 변환
ratio = original_image.shape[1] / float(image.shape[1])
receipt = four_point_transform(original_image, receiptCnt.reshape(4, 2) * ratio)
cv2.imshow("Receipt", receipt)
cv2.waitKey(0)
cv2.destroyAllWindows()
