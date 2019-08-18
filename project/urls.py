from django.conf.urls import url
from django.contrib import admin
from django.urls import include, path

from project import views

urlpatterns = [
    path('', views.index),
    path("type/",  views.changeChart),

]