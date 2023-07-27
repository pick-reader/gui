from flask import Flask, render_template, Response, url_for, request
import cv2
import os
import json
from camera import gen_frames, save_frame
from weftwarp import weft_warp #, cloth_type
from pwmLed import ledBrightness

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

IMG_PATH = "/static/cap.jpg"
OS_IMG_SAVE_PATH = app.root_path + IMG_PATH

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r



@app.route('/')
def index():
	return render_template('index.html')

@app.route('/vid')
def vid():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/compute')
def compute():
	img=save_frame(OS_IMG_SAVE_PATH)
	#ret = weft_warp(IMG_SAVE_PATH)
	##ret=weft_warp(img)
	cloth = "lol" #cloth_type(OS_IMG_SAVE_PATH)
	##ret = json.loads(ret)
	##print(ret)
#	return render_template('results.html', path=IMG_PATH, weft=ret['weft'], warp=ret['warp'], cloth=cloth)
	return render_template('results.html', path=IMG_PATH, weft="5", warp="5", cloth="5")

@app.route('/led')
def led():
	duty = int(request.args.get('duty'))
	pin = int(request.args.get('pin'))
	ledBrightness(pin, duty)
	return "ok"

app.run(host='0.0.0.0', port=8000)
