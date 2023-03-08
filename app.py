from flask import Flask, render_template, Response, url_for, request
import cv2
import os
import json
from camera import gen_frames, save_frame
from weftwarp import weft_warp
from pwmLed import ledBrightness

app = Flask(__name__)

IMG_PATH = "/static/cap.jpg"
OS_IMG_SAVE_PATH = app.root_path + IMG_PATH

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/vid')
def vid():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/compute')
def compute():
	save_frame(OS_IMG_SAVE_PATH)
	ret = weft_warp(OS_IMG_SAVE_PATH)
	ret = json.loads(ret)
	print(ret)
	return render_template('results.html', path=IMG_PATH, weft=ret['weft'], warp=ret['warp'])

@app.route('/led')
def led():
	duty = int(request.args.get('duty'))
	ledBrightness(duty)
	return "ok"

app.run(host='0.0.0.0', port=8000)
