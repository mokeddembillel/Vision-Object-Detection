import cv2

from backend.scene import Scene, UPDATE_NOISE_DELAY

scene = Scene(shape=(500, 800))
fps = 1/10
i = 0
while 1:
    if i % UPDATE_NOISE_DELAY == 0:
        scene.noises = []
        scene.generate(noise=True)
        scene.render()
    cv2.imshow('sdf', scene.img)
    cv2.waitKey(int(1000*fps))
    i += 1
