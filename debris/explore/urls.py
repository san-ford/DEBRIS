from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("create_db/", views.create_db, name="create_db"),
    path("upload/", views.upload, name="upload"),
    path("result/", views.result, name="result"),
]