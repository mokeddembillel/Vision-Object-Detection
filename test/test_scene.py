import cv2
from backend.scene import Scene, Shape
from backend.video import write_info_img
import numpy as np
from backend.colors import rgb2color

fps = 1/100
scene = Scene(shape=(500, 800), empty=True, shape_types=[Shape.CIRCLE, Shape.TRIANGLE, Shape.RECTANGLE])


color2rgb = {v:k for k,v in rgb2color.items()}
color = color2rgb['RED']
scene.generate_triangle((10, 10), color=color)
# scene.generate_triangle((250, 250))

scene.objects[0].velocity = np.array([0, 4])
# scene.objects[1].velocity = np.array([0, -4])

while 1:
    scene.render()
    scene.frame()
    scene.img = write_info_img(scene.img)
    cv2.imshow('h', scene.img)
    cv2.waitKey(int(fps*1000))