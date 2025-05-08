import cv2
import numpy as np
import os
from imutils.perspective import four_point_transform

def central_crop(image, crop_ratio=0.85):
    h, w = image.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    start_y = (h - ch) // 2
    start_x = (w - cw) // 2
    return image[start_y:start_y+ch, start_x:start_x+cw]

def clean_background_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

os.makedirs("output", exist_ok=True)

for i in range(1, 10):
    path = f"recipt_image/recipt{i}.jpg"
    original_image = cv2.imread(path)

    if original_image is None:
        print(f"⚠️ 이미지 {path} 를 불러올 수 없습니다.")
        continue

    h, w = original_image.shape[:2]
    image = cv2.resize(original_image.copy(), (500, int(h * (500 / w))))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    blurred = cv2.GaussianBlur(thresh, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    receiptCnt = None
    min_area = image.shape[0] * image.shape[1] * 0.3
    max_area = image.shape[0] * image.shape[1] * 0.95

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10000 or area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receiptCnt = approx
            break

    # 실패 시 Otsu로 재시도
    if receiptCnt is None:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_blurred = cv2.GaussianBlur(otsu, (7, 7), 0)
        otsu_edged = cv2.Canny(otsu_blurred, 50, 150)
        cnts, _ = cv2.findContours(otsu_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1000 or area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                receiptCnt = approx
                break

    output = image.copy()

    if receiptCnt is not None:
        cv2.drawContours(output, [receiptCnt], -1, (255, 0, 0), 3)
        ratio = original_image.shape[1] / float(image.shape[1])
        receipt = four_point_transform(original_image, receiptCnt.reshape(4, 2) * ratio)
        result = receipt
        print(f"✅ recipt{i}: 윤곽선 검출 성공")
    else:
        print(f"⚠️ recipt{i}: 윤곽선 검출 실패 → 중앙 크롭 + 배경 제거")
        receipt = central_crop(original_image)
        mask = clean_background_mask(receipt)
        result = cv2.bitwise_and(receipt, receipt, mask=mask)

    cv2.imwrite(f"result/receipt{i}_outline.jpg", output)
    cv2.imwrite(f"result/receipt{i}_result.jpg", result)
