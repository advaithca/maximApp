from flask import Flask, render_template, request

app = Flask(__name__)

optionDict = {1: "https://tfhub.dev/sayakpaul/maxim_s-2_deraining_raindrop/1", 2: "https://tfhub.dev/sayakpaul/maxim_s-2_deraining_rain13k/1"}

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET','POST'])
def process():
    things = request.form
    things2 = request.files['video']
    with open(f'video.{things2.filename.split(".")[0]}') as f:
        things2.save(f)
    return render_template('index.html')
if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True)