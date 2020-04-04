
# Create your models here.

from django.db import models
from django.utils import timezone


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
        heatmap=models.TextField()
        dataspace=models.TextField()
        total_communities=models.TextField()
        force_nodes=models.TextField()
        force_links=models.TextField()
        communities=models.TextField()
        size=models.IntegerField(default=0)
        distribution=models.TextField()
        max_degree=models.IntegerField()
        min_degree=models.IntegerField()
        average_degree=models.IntegerField()
        max_date=models.DateTimeField(default = timezone.now)
        min_date=models.DateTimeField(default = timezone.now)
        prediction = models.TextField(default="")
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

        def return_heatmap(self):
                return self.heatmap

        def return_dataspace(self):
                return self.dataspace

        def return_total_communities(self):
            return self.total_communities

        def return_force_nodes(self):
            return self.force_nodes

        def return_force_links(self):
            return self.force_links

        def return_communities(self):
            return self.communities

        def return_size(self):
            return self.size

        def return_distribution(self):
            return self.distribution

        def return_maxdate(self):
            return self.max_date

        def return_mindate(self):
            return self.min_date

        def return_maxdegree(self):
            return self.max_degree

        def return_mindegree(self):
            return self.min_degree

        def return_averagedegree(self):
            return self.average_degree

        def return_prediction(self):
            return self.prediction


class GraphInfo(models.Model):
        id = models.IntegerField(default=0, primary_key=True)
        heatmap = models.TextField()
        dataspace = models.TextField()

        def return_id(self):
            return self.id

        def return_heatmap(self):
                return self.heatmap

        def return_dataspace(self):
                return self.dataspace


class otherData(models.Model):
    id = models.AutoField(primary_key=True)
    Name = models.CharField(max_length=200)
    Nodes = models.IntegerField(default=0)
    Temporal_Edges = models.IntegerField(default=0)
    Static_Edges = models.IntegerField(default=0)
    TimeSpan = models.CharField(max_length=200)
    min_date = models.DateTimeField(default=timezone.now)
    max_date = models.DateTimeField(default=timezone.now)
    max_degree = models.IntegerField()
    min_degree = models.IntegerField()
    average_degree = models.IntegerField()
    heatmap = models.TextField()
    dataspace = models.TextField()
    total_communities = models.TextField()
    force_nodes = models.TextField()
    force_links = models.TextField()
    communities = models.TextField()
    size = models.IntegerField(default=0)
    distribution = models.TextField()
    prediction=models.TextField(default="")
    types=models.CharField(max_length=200,default="")
    description=models.CharField(max_length=500,default="")

    def return_id(self):
        return self.id

    def return_name(self):
        return self.Name

    def return_timespan(self):
        return self.TimeSpan

    def return_nodes(self):
        return self.Nodes

    def return_temporal_edges(self):
        return self.Temporal_Edges

    def return_static_edges(self):
        return self.Static_Edges

    def return_heatmap(self):
        return self.heatmap

    def return_dataspace(self):
        return self.dataspace

    def return_total_communities(self):
        return self.total_communities

    def return_force_nodes(self):
        return self.force_nodes

    def return_force_links(self):
        return self.force_links

    def return_communities(self):
        return self.communities

    def return_size(self):
        return self.size

    def return_distribution(self):
        return self.distribution

    def return_maxdate(self):
        return self.max_date

    def return_mindate(self):
        return self.min_date

    def return_maxdegree(self):
        return self.max_degree

    def return_mindegree(self):
        return self.min_degree

    def return_averagedegree(self):
        return self.average_degree

    def return_prediction(self):
        return self.prediction

    def return_description(self):
        return self.description

    def return_types(self):
        return self.types











