import cv2

from backend.scene import Scene

scene = Scene(shape=(500, 800), empty=False)

cv2.imshow('org', scene.img)
cv2.waitKey(-1)
scene.rectify_boundary_collisions()
scene.render()
cv2.imshow('bound', scene.img)
cv2.waitKey(-1)
scene.rectify_object_collisions()
scene.render()
cv2.imshow('obj', scene.img)
cv2.waitKey(-1)