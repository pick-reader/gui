from skimage.transform import (hough_line, hough_line_peaks,rotate)
from skimage.color import rgb2gray
from skimage import filters,io, transform
import numpy as np
import cv2 as cv
import math
import cv2
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tflite_runtime.interpreter import Interpreter
from PIL import Image

import json
#from PIL.Image import Resampling
import scipy.ndimage as inter

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )
def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
   
    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]
        
def correct_skew_weft(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv.BORDER_REPLICATE)

    return best_angle, corrected
        
def correct_skew_warp(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=0, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv.BORDER_REPLICATE)

    return best_angle, corrected
def power_spectrum(image_arr):
    original_height, original_width = image_arr.shape
    zp_factor = 1  
    pimg = np.pad(image_arr, ((0, original_height * (zp_factor - 1)), (0, original_width * (zp_factor - 1))), mode='constant')
    ft=np.fft.ifftshift(pimg)
    ft=np.fft.fft2(ft)
    ft=ft/(pimg.shape[0]*pimg.shape[1])

    ft=np.fft.fftshift(ft)
    ftconj=np.conj(ft)
    mag=np.abs(ft*ftconj)
    ps=mag
    ps=np.log(1+ps)

    dc_index = np.unravel_index(np.argmax(ps), ps.shape)
    centroid_x = dc_index[1]
    centroid_y = dc_index[0]
#ps[dc_index[0]-1][dc_index[1]] = 0
#ps[dc_index[0]+1][dc_index[1]] = 0
    ps[dc_index] = 0
    ps[centroid_y-8:centroid_y+8, centroid_x-8:centroid_x+8]=0

    return ps, centroid_x, centroid_y
def grouper(iterable):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= 2:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group
def weft_warp(img):
#    img_weft = img #io.imread(path)
    image = io.imread(img)
#    image = img
#    img_weft = img_weft[0:200, 0:200]
#    img_warp=cv2.rotate(img_weft, cv2.ROTATE_90_CLOCKWISE)
    image_height, image_width = image.shape[0:2]
    rotimh, rotimw=image.shape[0:2]
    angle1, cimweft=correct_skew_weft(image, 1, 5)
    angle2, cimwarp=correct_skew_warp(image, 1, 5)
    irc_weft = crop_around_center(
            cimweft,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(angle1)
            )
        )
    irc_warp = crop_around_center(
            cimwarp,
            *largest_rotated_rect(
                rotimw,
                rotimh,
                math.radians(angle2)
            )
        )

    try:
        grayimg1=cv.cvtColor(irc_weft, cv.COLOR_BGR2GRAY)
        grayimg2=cv.cvtColor(irc_warp, cv.COLOR_BGR2GRAY)

        image_array1 = np.array(grayimg1, dtype=np.float64)
        image_array2 = np.array(grayimg2, dtype=np.float64)
        ps1, centroid_x1, centroid_y1=power_spectrum(image_array1)
        ps, centroid_x, centroid_y=power_spectrum(image_array2)
        horizontal=[]
    #weft
        horizontal_index=[]
        vertical=[]
    #warp
        vertical_index=[]
        for i in range(0,ps.shape[0]):
            for j in range(0,ps.shape[1]):
            #if j==centroid_x and i<centroid_y and ps[i][j]>1:
                #horizontal.append(ps[i,j])
                #horizontal_index.append(i)
                if i==centroid_y and j<centroid_x and ps[i][j]>1:
                    vertical_index.append(j)
                    vertical.append(ps[i,j])
        for i in range(0,ps1.shape[0]):
            for j in range(0,ps1.shape[1]):
                if j==centroid_x1 and i<centroid_y1 and ps1[i][j]>1:
                    horizontal.append(ps1[i,j])
                    horizontal_index.append(i)
            #if i==centroid_y and j<centroid_x1 and ps1[i][j]>1:
                #vertical_index.append(j)
                #vertical.append(ps1[i,j])
            
        tdlist1=[]
        for i in grouper(horizontal_index):
            tdlist1.append(i)
        for i in tdlist1:
            for j in i:
                if(j>=(centroid_y1-9)):
                    tdlist1.remove(i)
                        
        tdlist2=[]
        for i in grouper(vertical_index):
            tdlist2.append(i)
        for i in tdlist2:
            for j in i:
                if(j>=(centroid_x-9)):
                    tdlist2.remove(i)
                        
        warp_value_list=max(tdlist2, key=len)
        for i in tdlist2:
            if len(i)==len(warp_value_list) and i != warp_value_list :
                if(i[0]>warp_value_list[0]):
                    warp_value_list=i
                        
        weft_value_list=max(tdlist1, key=len)
        for i in tdlist1:
            if len(i)==len(weft_value_list) and i != weft_value_list :
                if(i[0]>weft_value_list[0]):
                    weft_value_list=i
                        
        max_weft,max_bright=0,0
        for i in weft_value_list:
            if(ps1[i][centroid_x1]>max_bright):       
                max_bright=ps1[i][centroid_x1]
                max_weft=i
        max_warp,max_bright=0,0
        for i in warp_value_list:
            if(ps[centroid_y][i]>max_bright):       
                max_bright=ps[centroid_y][i]
                max_warp=i

        distance_y = max_warp - centroid_x
        distance_x = max_weft - centroid_y1
            
        return json.dumps({
            'weft': int(str(abs(distance_x))),
            'warp': int(str(abs(distance_y)))
        }), irc_weft, irc_warp
    except ValueError:
        return json.dumps({
            'weft': 0,
            'warp': 0
        }), irc_weft, irc_warp

def cloth_type(image):
        interpreter = Interpreter(model_path=r"/home/pi/flask_cam/linear1.tflite")
        interpreter.allocate_tensors()
        input_index=interpreter.get_input_details()
#        img2 = image
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
