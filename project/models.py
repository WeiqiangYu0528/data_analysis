from django.db import models

# Create your models here.
from django.db import models


class Datainfo(models.Model):
    id = models.IntegerField(default=0, primary_key=True)
    Name = models.CharField(max_length=200)
    Type = models.CharField(max_length=200)
    Nodes= models.IntegerField(default=0)
    Temporal_Edges= models.IntegerField(default=0)
    Static_Edges = models.IntegerField(default=0)
    Description=models.CharField(max_length=200)

    def return_id(self):
        return self.id

    def return_name(self):
        return self.Name

    def return_type(self):
        return self.Type

    def return_nodes(self):
        return self.Nodes

    def return_temporal_edges(self):
        return self.Temporal_Edges

    def return_static_edges(self):
        return self.Static_Edges

    def return_description(self):
        return self.Description

class SubData(models.Model):
        SuperData= models.ForeignKey(Datainfo, on_delete="PROTECT")
        Name = models.CharField(max_length=200)
        TimeSpan=models.IntegerField(default=0)
        Nodes = models.IntegerField(default=0)
        Temporal_Edges = models.IntegerField(default=0)
        Static_Edges = models.IntegerField(default=0)
        id=models.IntegerField(default=0, primary_key=True)

        def return_id(self):
            return self.id

        def return_name(self):
            return self.Name

        def return_timespan(self):
            return self.TimeSpan

        def return_super(self):
            return self.SuperData

        def return_nodes(self):
            return self.Nodes

        def return_temporal_edges(self):
            return self.Temporal_Edges

        def return_static_edges(self):
            return self.Static_Edges








