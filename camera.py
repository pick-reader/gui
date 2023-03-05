from picamera2 import Picamera2
import cv2
import os
picam2 = Picamera2()


x = [400, 1540]
y = [400, 940]


camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.start()


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
