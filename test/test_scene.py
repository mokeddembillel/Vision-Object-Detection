import cv2
import numpy as np
from backend.scene import Scene

fps = 1/100
scene = Scene(shape=(500, 800), empty=True)
scene.generate_circle((250, 50))
scene.generate_circle((250, 500))

scene.objects[0].velocity = np.array([3, 0])
scene.objects[1].velocity = np.array([3, 0])
while 1:
    scene.render()
    scene.frame()
    cv2.imshow('h', scene.img)
    cv2.waitKey(int(fps*1000))