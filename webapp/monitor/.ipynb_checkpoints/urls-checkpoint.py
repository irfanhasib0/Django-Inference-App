from django.urls import path
from django.conf.urls.static import static
from monitor.views import image_1, image_2, dashboard, upload_image
from webapp import settings

urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('image_1', image_1, name='image_1'),
    path('image_2', image_2, name='image_2'),
    path('upload_image', upload_image, name='upload_image')
]

urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
