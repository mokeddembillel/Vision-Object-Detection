import cv2

from backend.scene import Scene, Shape
from backend.utils import detect


def generate_video(resolution=(500, 800), time=10, fps=100, num_objects=5, num_noises=20, shape_types=list(Shape),
                   delay_change_noise=1):
    # time and delay in seconds
    delay_change_noise = int(delay_change_noise * fps)
    fps = 1 / fps
    scene = Scene(shape=resolution, num_objects=num_objects, num_noise=num_noises, empty=True, shape_types=shape_types)
    scene.generate(noise=True)
    scene.generate(noise=False)
    video = []
    for i in range(int(time / fps)):
        if i % delay_change_noise == 0:
            scene.noises = []
            scene.generate(noise=True)
        scene.render()
        scene.frame()
        video.append(scene.img.copy())
    return video


def write_video(path, video, fps):
    # path ends with .avi
    h, w, _ = video[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    for i in range(len(video)):
        out.write(video[i])
    out.release()


def read_video(path):
    cap = cv2.VideoCapture(path)
    video = []
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame)
        # frame = write_info_img(frame)
        # cv2.imshow('cv', frame)
        # cv2.waitKey(10)
    return video


def write_info_img(img):
    result = detect(img)
    for cog, shape, color in result:
        text = str(shape) + ' : ' + str(color)
        cv2.putText(img, text, tuple(cog), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0))
    return img
