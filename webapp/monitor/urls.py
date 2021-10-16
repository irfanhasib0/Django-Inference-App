from django.urls import path
from monitor.views import image_1, image_2, dashboard

urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('image_1', image_1, name='image_1'),
    path('image_2', image_2, name='image_2')
]
