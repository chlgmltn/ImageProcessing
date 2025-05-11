import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from imutils.perspective import order_points

def central_crop(image, crop_ratio=0.85):
    h, w = image.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    start_y = (h - ch) // 2
    start_x = (w - cw) // 2
    return image[start_y:start_y+ch, start_x:start_x+cw]

original_image =  cv2.imread("recipt_image/recipt9.jpg")

image = original_image.copy()
image = imutils.resize(image, width=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 그림자 제거
dilated = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
bg = cv2.medianBlur(dilated, 21)
shadowless = cv2.absdiff(gray, bg)
shadowless = cv2.normalize(shadowless, None, 0, 255, cv2.NORM_MINMAX)

# 히스토그램 평활화
shadowless_eq = cv2.equalizeHist(shadowless)

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

# 외곽선 추출 및 면적 기준 정렬
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# 사각형 윤곽선 찾기
receiptCnt = None
min_area = image.shape[0] * image.shape[1] * 0.2  # 20% 이상만
max_area = image.shape[0] * image.shape[1] * 0.95  # 너무 큰 것도 제외

for c in cnts:
    if cv2.contourArea(c) < 10000: continue
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

# 실패 시 방법1 - minAreaRect
if receiptCnt is None:
    print("1. Could not find 4-point contour. Using minAreaRect fallback.")
    blurred = cv2.GaussianBlur(thresh, (5,5,), 0)
    edged = cv2.Canny(blurred, 50, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    largest = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area or area > max_area:
        print("❌ minAreaRect contour size out of range")
    else:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = order_points(box)
        w, h = rect[1]
        angle = abs(rect[2])
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        if angle > 30 or aspect_ratio > 4:
            print("⚠️ minAreaRect shape invalid → ignored")
        else:
            receiptCnt = np.intp(box)

        
# 실패 시 방법2
if receiptCnt is None:
    print("2. Could not find 4-point contour. Using minAreaRect fallback.")
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_blurred = cv2.GaussianBlur(otsu, (7, 7), 0)
    otsu_edged = cv2.Canny(otsu_blurred, 50, 150)
    cnts = cv2.findContours(otsu_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receiptCnt = approx
            break

if receiptCnt is None:
    print("⚠️ 윤곽선 검출 실패 → 중앙 크롭 사용")
    receipt = central_crop(original_image)
else:
    print("✅ 윤곽선 검출 성공 → 투시 변환")

# 윤곽선 시각화 (윤곽선이 있을 경우만)
output = image.copy()
if receiptCnt is not None:
    cv2.drawContours(output, [receiptCnt], -1, (255, 0, 0), 3)
    cv2.imshow("Receipt Outline", output)
    ratio = original_image.shape[1] / float(image.shape[1])
    receipt = four_point_transform(original_image, receiptCnt.reshape(4, 2) * ratio)

cv2.imshow("Receipt", receipt)
cv2.waitKey(0)
cv2.destroyAllWindows()
