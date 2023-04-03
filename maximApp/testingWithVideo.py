import cv2
from cap_from_youtube import cap_from_youtube
from GetInference import infer, imshow
from PIL import Image
from matplotlib import pyplot as plt
import glob

youtube_url = 'https://youtu.be/-ECyW8r3_uw'
cap = cap_from_youtube(youtube_url, "480p")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height

frame = (width, height)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'),60,frame)

count = 0

print("\n\nEntering...\n")
# cv2.namedWindow('video', cv2.WINDOW_NORMAL)
while True:
    plt.figure(figsize=(15,15))
    ret, frame = cap.read()
    count += 1
    print(f"Frame {count}/{length}", end="\n\n")
    frameDat = infer(frame)
    frameDat = imshow(frameDat)
    if not ret:
        break
    # plt.subplot(1,2,1)
    # frame = Image.fromarray(frame)
    # plt.imshow(frame)

    # plt.subplot(1,2,2)
    plt.imshow(frameDat)
    plt.axis('off')
    plt.savefig(f"images/{count}.jpg", bbox_inches='tight')

for image in glob.glob("images/*.jpg"):
    img = cv2.imread(image)
    out.write(img)

out.release()