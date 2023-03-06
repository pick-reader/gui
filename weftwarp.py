from skimage.transform import (hough_line, hough_line_peaks,rotate)
from skimage.color import rgb2gray
from skimage import filters,io, transform
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tflite_runtime.interpreter import Interpreter
from PIL import Image

import json
#from PIL.Image import Resampling
def count(image):
        amount = 250
        radius = 75
        blurred = cv2.GaussianBlur(image, (radius*2+1, radius*2+1), 0)
        unsharp = cv2.addWeighted(image, 1.0 + amount/100.0, blurred, -amount/100.0, 0)
        imgGray = rgb2gray(unsharp)
        image = preprocessing.Binarizer(threshold=0.5).transform(imgGray)
        tested_angles = np.linspace(89.2/180*np.pi, 90.25/180*np.pi, 1)

        hspace, theta, dist = hough_line(image, tested_angles)  
        h, q, d = hough_line_peaks(hspace, theta, dist)
        angle_list=[]  
        origin = np.array((0, image.shape[1])) 
        return len(h)
    
def weft_warp(path: str):
    img_weft = io.imread(path)
#    img_weft = img_weft[0:200, 0:200]
    img_warp=cv2.rotate(img_weft, cv2.ROTATE_90_CLOCKWISE)
    weft=count(img_weft)
    warp=count(img_warp)
    #cloth=cloth_type(path)
    img_ret = cv2.resize(img_weft, (int(300*0.9), int(200*0.9)))
    return json.dumps({
        'weft': weft,
        'warp': warp
    })
def cloth_type(image):
        interpreter = Interpreter(model_path=r"/home/pi/lo/warp-weft-counting/linear1.tflite")
        interpreter.allocate_tensors()
        input_index=interpreter.get_input_details()
        img=Image.open(image).resize((300,300))
#        Image.Resampling.LANCZOS
        im=np.array(img,dtype=np.float32)
        imarr=im[np.newaxis, ...]
        input_index=interpreter.get_input_details()[0]["index"]
        interpreter.set_tensor(input_index,imarr)
        interpreter.invoke()
        output_details=interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = np.squeeze(output_data)
        class_names=["Cotton","Denim","Nylon","Polyester","Silk","Wool"]
        prediction = np.argmax(pred)
        return(class_names[prediction])