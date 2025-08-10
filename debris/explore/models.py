from django.db import models


class UploadedImages(models.Model):
    encoded_image = models.CharField(max_length=500)
    embeddings = models.JSONField(default=list, blank=True)
    node = models.IntegerField(default=-1)
