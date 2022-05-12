from http.client import HTTPResponse
from json import dumps
from collections import deque
import cv2
import base64
import numpy as np
import time
import io
import glob 
from matplotlib import pyplot as plt
from detect import predict

index = 0
def app1(request):
    return render(request, 'app1.html', {})
    
def image_1():
    path='/code/media/img.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img,(400,300))
    _, image_data = cv2.imencode('.png', img)
    image_data = bytearray(image_data)
    image_data = base64.b64encode(image_data).decode('utf-8')
    return image_data

stream = []
frame_ind = 0
vid = cv2.VideoCapture('/code/media/video.mp4')

def upload_image(request):
    global stream, vid
    if request.method == 'POST':
        if 'image' in list(request.FILES.keys()):
            with open('media/img.jpg', 'wb+') as file:
                file.write( request.FILES['image'].read())
            return render(request, 'dash-board.html')
        
        if 'video' in list(request.FILES.keys()):
            with open('media/video.mp4', 'wb+') as file:
                #for chunk in request.FILES['filename'].chunks():
                file.write( request.FILES['video'].read())
            stream=[]
            frame_ind=0
            vid = cv2.VideoCapture('/code/media/video.mp4')
            return render(request, 'dash-board.html')
    

def image_2():
    global stream, frame_ind, vid
    
    ret, frame = vid.read()
    if ret == True:
        img = cv2.resize(frame,(256,256))
        img = predict(img)
        _, image_data = cv2.imencode('.jpg', img)
        image_data = bytearray(image_data)
        image_data = base64.b64encode(image_data).decode('utf-8')
        stream.append(image_data)
         
    frame = stream[frame_ind]
    if frame_ind <=  len(stream) : frame_ind +=1
    return frame
       
        
