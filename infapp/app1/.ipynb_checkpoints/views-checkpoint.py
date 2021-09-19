from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from json import dumps
import django
django.setup()
from .models import DataIO
from collections import deque

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
        if index == 100:
            index=0
        data_1.append(data.col_1)
        data_2.append(data.col_2)
        data_3.append(data.col_3)
        data_4.append(data.col_4)
    data = {'1':data_1,'2':data_2,'3':data_3,'4':data_4}
    #return  HttpResponse(dumps(data), content_type="application/json")
    return JsonResponse(data)