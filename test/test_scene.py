import cv2
from backend.scene import Scene, Shape
import numpy as np

fps = 1/100
scene = Scene(shape=(500, 800), empty=False, shape_types=[Shape.CIRCLE, Shape.TRIANGLE])

# scene.generate_triangle((450, 250))
# scene.generate_triangle((250, 250))
#
# scene.objects[0].velocity = np.array([0, 4])
# scene.objects[1].velocity = np.array([0, -4])

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


while 1:
    scene.render()
    scene.frame()
    cv2.imshow('h', scene.img)
    cv2.waitKey(int(fps*1000))