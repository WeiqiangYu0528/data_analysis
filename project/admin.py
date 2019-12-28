from django.contrib import admin

# Register your models here.
from .models import Datainfo,SubData,GraphInfo

admin.site.register(Datainfo)
admin.site.register(SubData)
admin.site.register(GraphInfo)