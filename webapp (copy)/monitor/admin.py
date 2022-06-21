from django.contrib import admin
from monitor.models import DataIO, ImageIO, UploadImage
# Register your models here.
admin.site.register(DataIO)
admin.site.register(ImageIO)
admin.site.register(UploadImage)
