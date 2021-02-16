import cv2

from backend.scene import Scene

scene = Scene(shape=(500, 800))

cv2.imshow('sdf', scene.img)
cv2.waitKey(-1)
