import cv2
from backend.scene import Scene, Shape
from backend.video import write_info_img
import numpy as np

fps = 1/100
scene = Scene(shape=(500, 800), empty=False, shape_types=[Shape.CIRCLE, Shape.TRIANGLE, Shape.RECTANGLE])

# scene.generate_triangle((450, 250))
# scene.generate_triangle((250, 250))
#
# scene.objects[0].velocity = np.array([0, 4])
# scene.objects[1].velocity = np.array([0, -4])

while 1:
    scene.render()
    scene.frame()
    scene.img = write_info_img(scene.img)
    cv2.imshow('h', scene.img)
    cv2.waitKey(int(fps*1000))