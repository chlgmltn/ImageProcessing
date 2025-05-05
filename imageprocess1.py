import cv2
import imutils
from imutils.perspective import four_point_transform

original_image =  cv2.imread("recipt_image/recipt2.jpg")

image = original_image.copy()
image = imutils.resize(image, width=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5,), 0)
edged = cv2.Canny(blurred, 75, 200)

# 디버깅 1
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

receiptCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

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

