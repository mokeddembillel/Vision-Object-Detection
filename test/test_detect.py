import cv2

from backend.utils import detect

img = cv2.imread('detect.jpg')
print(detect(img))


