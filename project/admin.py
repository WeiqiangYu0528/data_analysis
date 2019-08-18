from django.contrib import admin

# Register your models here.
from .models import Datainfo,SubData

admin.site.register(Datainfo)
admin.site.register(SubData)