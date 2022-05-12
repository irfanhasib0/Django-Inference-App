import django
django.setup()
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from json import dumps
from .models import DataIO, ImageIO

from collections import deque
import cv2
import base64
import numpy as np
import time
import io
from matplotlib import pyplot as plt

index = 0
def app1(request):
    return render(request, 'app1.html', {})

def dashboard(request):
    global index
    data = DataIO.objects.get(index=index)
    index +=1
    if index == 100:
        index=0
    data_json = dumps({'1':data.col_1,'2':data.col_2,'3':data.col_3,'4':data.col_4})
    return render(request, 'dash-board.html', {'data_dict': data_json})

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
    global index
    data_1 = []; data_2 =[];
    for _ in range(10):
        data = ImageIO.objects.get(index=index)
        index +=1
        if index == 10:
            index=0
        data_1.append(data.col_1)
        data_2.append(data.col_2)
    data = {'1':data_1,'2':data_2}
    img = cv2.imread(data_1[0])
    img = cv2.resize(img,(480,320))
    _, image_data = cv2.imencode('.png', img)
    image_data = bytearray(image_data)
    image_data = base64.b64encode(image_data).decode('utf-8')
        
    #with open(data_1[0], "rb") as image_file:
     #   image_data = base64.b64encode(image_file.read()).decode('utf-8')
    #time.sleep(1)
    return HttpResponse(image_data,content_type='image/png')

data_1 =  deque(maxlen = 10); 
data_2 =  deque(maxlen = 10); 
data_3 =  deque(maxlen = 10); 
data_4 =  deque(maxlen = 10)

def image_2(request):
    global data_1,data_2,data_3,data_4
    data = DataIO.objects.get(index=index)

    data_1.append(data.col_1)
    data_2.append(data.col_2)
    data_3.append(data.col_3)
    data_4.append(data.col_4)

    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,5))
    plt.subplots_adjust(wspace=0.5)
    my_stringIObytes = io.BytesIO()
    axes[0,0].plot(data_1,color='b');axes[0,0].grid()
    axes[0,1].plot(data_2,color='r');axes[0,1].grid()
    axes[1,0].plot(data_3,color='g');axes[1,0].grid()
    axes[1,1].plot(data_4,color='c');axes[1,1].grid()
    fig.savefig(my_stringIObytes, format='png',dpi=100)
    my_stringIObytes.seek(0)
    plt.cla()
    image_data = base64.b64encode(my_stringIObytes.read()).decode('utf-8')
    return HttpResponse(image_data,content_type='image/png')

