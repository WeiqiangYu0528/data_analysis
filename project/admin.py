from django.contrib import admin

# Register your models here.
from .models import Datainfo,SubData,GraphInfo,otherData

admin.site.register(Datainfo)
admin.site.register(SubData)
admin.site.register(GraphInfo)
admin.site.register(otherData)