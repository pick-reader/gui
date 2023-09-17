from flask import Flask, render_template, Response, url_for, request, send_file
import cv2
import os
import json
from camera import gen_frames, save_frame
from weftwarp import weft_warp #, cloth_type
from pwmLed import ledBrightness

weft_img = None
warp_img = None

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
	global weft_img, warp_img
	img=save_frame(OS_IMG_SAVE_PATH)
	ret, weft_img, warp_img = weft_warp('.' + IMG_PATH)
#	ret=weft_warp(img)
	print(ret)
	cloth = "lol" #cloth_type(OS_IMG_SAVE_PATH)
	ret = json.loads(ret)
	##print(ret)
#	return render_template('results.html', path=IMG_PATH, weft=ret['weft'], warp=ret['warp'], cloth=cloth)
	return render_template('results.html', weft=ret['weft'], warp=ret['warp'], cloth="5")

@app.route('/led')
def led():
	duty = int(request.args.get('duty'))
	pin = int(request.args.get('pin'))
	ledBrightness(pin, duty)
	return "ok"

@app.route('/weft_img')
def weft_img():
	global weft_img
#	a = cv2.imread('.' + IMG_PATH)
	img = cv2.cvtColor(weft_img, cv2.COLOR_BGR2RGB)
	ret, buffer = cv2.imencode('.jpg', img)
	buffer = buffer.tobytes()
#	yield(b'--frame\r\n Content-Type: image\r\n\r\n' + buffer + b'\r\n')
	return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/warp_img')
def warp_img():
	global warp_img
	img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB)
	ret, buffer = cv2.imencode('.jpg', img)
	buffer = buffer.tobytes()
	return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=8000)
