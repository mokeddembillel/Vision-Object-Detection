from backend.utils import *

img = cv2.imread('j.png')

cv2.imshow('sdf', img)
cv2.waitKey(-1)

cv2.imshow('sdf', mean_filter(img, kernel_size=3))
cv2.waitKey(-1)