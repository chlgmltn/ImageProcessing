import cv2
import imutils
from imutils.perspective import four_point_transform

original_image =  cv2.imread("recipt_image/recipt2.jpg")

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

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
edged_cleaned = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# 디버깅 1
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

receiptCnt = None
for c in cnts:
    if cv2.contourArea(c) > 1000:  # 너무 작으면 제외
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) == 4:
        receiptCnt = approx
        break

if receiptCnt is None:
    raise Exception(("Could not find outline"))

output = image.copy()
cv2.drawContours(output, [receiptCnt], -1, (255, 0, 0), 3)
cv2.imshow("Receipt Outline", output)

ratio = original_image.shape[1] / float(image.shape[1])
receipt = four_point_transform(original_image, receiptCnt.reshape(4, 2) * ratio)
cv2.imshow("Receipt", receipt)
cv2.waitKey(0)
cv2.destroyAllWindows()