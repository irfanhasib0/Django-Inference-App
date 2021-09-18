from django.shortcuts import render

def app1(request):
    return render(request, 'app1.html', {})