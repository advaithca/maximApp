import cv2
from cap_from_youtube import cap_from_youtube
from GetInference import infer, imshow
from PIL import Image

youtube_url = 'https://youtu.be/4jaGyv0KRPw'
cap = cap_from_youtube(youtube_url)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height

frame = (width, height)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'),60,frame)

count = 0

print("\n\nEntering...\n")
# cv2.namedWindow('video', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    count += 1
    print(f"Frame {count}/{length}", end="\n\n")
    print(type(frame))
    frameDat = infer(frame)
    frameDat = imshow(frameDat)
    if not ret:
        break
    img = Image.fromarray(frameDat)
    img.save(f"{count}.jpg")


out.write(frameDat)