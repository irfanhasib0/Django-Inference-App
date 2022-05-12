import django
django.setup()
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from monitor.models import DataIO, ImageIO, UploadImage
from monitor.forms import UserImage
from json import dumps
from collections import deque
import cv2
import base64
import numpy as np
import time
import io
import glob 
from matplotlib import pyplot as plt

index = 0
def app1(request):
    return render(request, 'app1.html', {})

def dashboard(request):
    return render(request, 'dash-board.html')

def data(request):
    global index
    data_1 = []; data_2 =[]; data_3 =[]; data_4 =[]
    for _ in range(10):
        data = DataIO.objects.get(index=index)
        index +=1
        if index == 10:
            index=0
        data_1.append(data.col_1)
        data_2.append(data.col_2)
        data_3.append(data.col_3)
        data_4.append(data.col_4)
    data = {'1':data_1,'2':data_2,'3':data_3,'4':data_4}
    #return  HttpResponse(dumps(data), content_type="application/json")
    #time.sleep(0.5)
    return JsonResponse(data,safe=False)
    
def image_1(request):
    path='/code/media/img.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img,(400,300))
    _, image_data = cv2.imencode('.png', img)
    image_data = bytearray(image_data)
    image_data = base64.b64encode(image_data).decode('utf-8')
    return HttpResponse(image_data,content_type='image/png')





def _image_2(request):
    path='/code/media/img.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img,(400,300))
    _, image_data = cv2.imencode('.png', img)
    image_data = bytearray(image_data)
    image_data = base64.b64encode(image_data).decode('utf-8')
    return HttpResponse(image_data,content_type='image/png')

data_1 =  deque(maxlen = 10); 
data_2 =  deque(maxlen = 10); 

def image_3(request):
    global data_1,data_2,data_3,data_4
    data = DataIO.objects.get(index=index)

    data_1.append(data.col_1)
    data_2.append(data.col_2)

    fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(5,3))
    plt.subplots_adjust(wspace=0.5)
    my_stringIObytes = io.BytesIO()
    axes[0].plot(data_1,color='b');axes[0].grid()
    axes[1].plot(data_2,color='r');axes[1].grid()
    fig.savefig(my_stringIObytes, format='png',dpi=100)
    my_stringIObytes.seek(0)
    plt.cla()
    image_data = base64.b64encode(my_stringIObytes.read()).decode('utf-8')
    return HttpResponse(image_data,content_type='image/png')

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
    

def image_2(request):
    global stream, frame_ind, vid
    
    ret, frame = vid.read()
    if ret == True:
        img = cv2.resize(frame,(400,300))
        _, image_data = cv2.imencode('.png', img)
        image_data = bytearray(image_data)
        image_data = base64.b64encode(image_data).decode('utf-8')
        stream.append(image_data)
         
    frame = stream[frame_ind]
    if frame_ind <=  len(stream) : frame_ind +=1
    return HttpResponse(frame,content_type='image/png')
       
        
