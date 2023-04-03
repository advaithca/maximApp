from flask import Flask, render_template, request
import cv2
from cap_from_youtube import cap_from_youtube
from GetInference import infer, imshow
from PIL import Image
from matplotlib import pyplot as plt
import glob

app = Flask(__name__)

optionDict = {'1': "https://tfhub.dev/sayakpaul/maxim_s-2_deraining_raindrop/1", '2': "https://tfhub.dev/sayakpaul/maxim_s-2_deraining_rain13k/1"}

def doVideo(option, videoName):
    cap = cv2.VideoCapture(videoName)

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
        frameDat = infer(frame, option)
        frameDat = imshow(frameDat)
        if not ret:
            break

        plt.imshow(frameDat)
        plt.axis('off')
        plt.savefig(f"images/{count}.jpg", bbox_inches='tight')

    for image in glob.glob("images/*.jpg"):
        img = cv2.imread(image)
        out.write(img)

    out.release()

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET','POST'])
def process():
    global optionDict
    things = request.form
    things2 = request.files['video']
    with open(f'video.{things2.filename.split(".")[1]}','+wb') as f:
        things2.save(f)
    doVideo(optionDict[things['option']], f'video.{things2.filename.split(".")[1]}')
    return render_template('index.html')
if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True)