import cv2
import numpy as np

img = cv2.imread('test_image/test1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = cv2.blur(hsv, (5, 5))
mask = cv2.inRange(hsv, (89, 124, 73), (255, 255, 255))
lower_blue = np.array([124, 123, 138])


upper_blue = np.array([150, 150, 150])

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
counturs = sorted(contours, key=cv2.contourArea, reverse=True)
for contour in counturs:

    cv2.drawContours(img, counturs[0], -1, (255, 0, 255), 3)
    while True:
        cv2.imshow("Counturs", img)  # рисует рокно с конурами
        key = cv2.waitKey(1)
        if key == 27:
            break
cv2.destroyAllWindows()
