from backend.video import generate_video, write_video, read_video

video = generate_video()
write_video('test.avi', video, 100)
read_video('test.avi')