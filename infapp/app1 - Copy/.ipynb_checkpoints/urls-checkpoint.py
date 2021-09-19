from django.urls import path
from app1 import views

urlpatterns = [
    #path('', views.app1, name='app1'),
    path('', views.dashboard, name='dashboard'),
    path('data', views.data, name='data'),
]
