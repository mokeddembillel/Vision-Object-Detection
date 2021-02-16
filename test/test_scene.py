import cv2

from backend.scene import Scene

scene = Scene(shape=(500, 800))

fps = 1/100
while 1:
    # Zid frame function
    scene.render()
    cv2.imshow('sdf', scene.img)
    cv2.waitKey(int(fps*1000))
