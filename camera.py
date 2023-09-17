from picamera2 import Picamera2

from libcamera import controls
import cv2
import os
picam2 = Picamera2()

#for full res
x = [505, 1380]
y = [385, 820]


#x = [0, -1]
#y = [0, -1]

camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)
#picam2.set_controls({"Sharpness":1.5, "Contrast": 1})
picam2.start()
try:
	picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
except:
	pass
#picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 10.0})


#picam2.capture_array()

def gen_frames():
	while(True):
		img = picam2.capture_array()[y[0]:y[1], x[0]:x[1]]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		ret, buffer = cv2.imencode('.jpg', img)
		img = buffer.tobytes()
		yield(b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

def save_frame(path):
		img = picam2.capture_array()[y[0]:y[1], x[0]:x[1]]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		cv2.imwrite(path, img)
		return img
