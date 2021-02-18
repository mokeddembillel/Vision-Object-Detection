from backend.video import read_video, generate_video, write_video

# video = generate_video()
# write_video('test.avi', video, 100)
video = read_video('test.avi')
write_video('test_annotated.avi', video, 100)
