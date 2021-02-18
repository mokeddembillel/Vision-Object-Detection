from backend.utils import *

img = cv2.imread('detect.jpg')

detect(img)

cv2.imshow('sdf', img)
cv2.waitKey(-1)

cv2.imshow('sdf2', gradient(img, kernel_size=3))
cv2.waitKey(-1)
