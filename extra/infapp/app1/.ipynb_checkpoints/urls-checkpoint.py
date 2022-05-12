from django.urls import path
from app1 import views

urlpatterns = [
    #path('', views.app1, name='app1'),
    path('', views.dashboard, name='dashboard'),
    path('data', views.data, name='data'),
    path('image_1', views.image_1, name='image_1'),
    path('image_2', views.image_2, name='image_2')
]
