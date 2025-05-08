import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform

original_image =  cv2.imread("recipt_image/recipt7.jpg")

image = original_image.copy()
image = imutils.resize(image, width=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 밝기 대비 향상을 위해 adaptive threshold 적용
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# 이진화 결과 확인
cv2.imshow("Threshold", thresh)

blurred = cv2.GaussianBlur(thresh, (7,7,), 0)
edged = cv2.Canny(blurred, 50, 150)

# 디버깅 1
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("thresh checksum:", np.sum(thresh))
print("blurred checksum:", np.sum(blurred))

# 외곽선 추출 및 면적 기준 정렬
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# 사각형 윤곽선 찾기
receiptCnt = None
min_area = image.shape[0] * image.shape[1] * 0.2  # 20% 이상만
max_area = image.shape[0] * image.shape[1] * 0.95  # 너무 큰 것도 제외

for c in cnts:
    if cv2.contourArea(c) > 1000:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            if area > max_area and 0.5 < aspect_ratio < 2.5:  # 세로로 긴 경우
                max_area = area
                receiptCnt = approx
            if 0.4 < aspect_ratio < 0.8 or 0.8 < aspect_ratio < 1.2 or aspect_ratio > 1.5:  # 다양한 가로/세로 비율 허용
                receiptCnt = approx
                break

# 윤곽선 없으면 minAreaRect로 보정 시도
if receiptCnt is None:
    print("Could not find 4-point contour. Using minAreaRect fallback.")
    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    receiptCnt = np.intp(box)

# 윤곽선 시각화
output = image.copy()
cv2.drawContours(output, [receiptCnt], -1, (255, 0, 0), 3)
cv2.imshow("Receipt Outline", output)

# 원본 이미지 비율로 다시 투시 변환
ratio = original_image.shape[1] / float(image.shape[1])
receipt = four_point_transform(original_image, receiptCnt.reshape(4, 2) * ratio)
cv2.imshow("Receipt", receipt)
cv2.waitKey(0)
cv2.destroyAllWindows()