from django.db import models


class ImageSubmitted(models.Model):
    submission = models.CharField(max_length=500)
    node = models.IntegerField(default=0)


class ImageRetrieved(models.Model):
    encoded_image = models.CharField(max_length=500)
    node = models.IntegerField(default=0)
