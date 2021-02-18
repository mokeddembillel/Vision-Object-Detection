from backend.utils import *

img = cv2.imread('j.png')

cv2.imshow('sdf', img)
cv2.waitKey(-1)

cv2.imshow('sdf2', laplacian(img, kernel_size=5))
cv2.waitKey(-1)
