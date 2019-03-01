import cv2
import numpy
def LibFunc():
    img = cv2.imread('test.jpg')
    cv2.imshow("Image",img)
    cv2.waitKey(500)


