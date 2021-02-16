import cv2

from backend.utils import detect

img = cv2.imread('test.png')
print(detect(img))


