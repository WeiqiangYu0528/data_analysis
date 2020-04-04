from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from project import views

urlpatterns = [
    path('', views.index),
    path('document/<type>-<int:id>', views.document),
    path('index/', views.index),
    path("type/",  views.changeChart),
    path("graph/",  views.changed3),
    path("apollo/", views.change_apollo),
    path("doc",views.doc),
    path("download/<filename>",views.download,name='download'),
    path("zipdownload/",views.zipdownload,name='zipdownload'),
    path("prediction/",views.prediction,name='prediction'),
    path("extra/",views.extra,name='extra'),
    path("upload/",views.upload,name='upload'),

    # path("error/",views.error)
]

