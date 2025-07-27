from django.contrib import admin

from .models import UploadedImages, ImageSubmitted

admin.site.register(UploadedImages)
admin.site.register(ImageSubmitted)
