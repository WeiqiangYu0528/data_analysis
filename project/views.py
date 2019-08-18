from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.shortcuts import render, get_object_or_404

# Create your views here.
from django.http import HttpResponse, JsonResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

from django.views.generic import CreateView

from .models import Datainfo, SubData
import plotly.graph_objects as go
from plotly.offline import plot
import json


def index(request):
    files=os.listdir("./project/files")
    files.remove(".DS_Store")
    files.sort()
    filename = request.GET.get('fn', "sx-mathoverflow-c2q.txt")
    data=readfile("./project/files/" + filename)
    details = preprocess(data)
    data_info = []
    infos=readwholedata()
    id=14
    for inx,item in enumerate(files):
        if(item==""+filename):
            id=inx
    info=readspecificdata(id)
    if(filename=="soc-sign-bitcoinalpha.csv"):
        details.append("-10 to +10")
        details.append("93%")
    if (filename == "soc-sign-bitcoinotc.csv"):
        details.append("-10 to +10")
        details.append("89%")
    graph_type= request.GET.get('tg', 'A')
    for index, row in data.head(10).iterrows():
        temp = []
        for item in row:
            temp.append(item)
        data_info.append(temp)
    df=createdataframe(data)
    fig=generateImg(df,graph_type,filename)
    graph=dftojson(data)
    context={"data_info":data_info,
             "filename":filename,
             "details":details,
             "type":graph_type,
             "files":files,
             "info":info,
             "fig":fig,
             "graph":graph,
             }
    return render(request, 'pages/index.html', context)


def createdataframe(data):
    df = pd.DataFrame(data['UNIXTS'].sort_values(ascending=True).apply(timestamp2datetime))
    df['UNIXTS'] = pd.to_datetime(df['UNIXTS'])  # 将UNIXTS字段转化为日期类型
    df = df.set_index('UNIXTS')  # 将UNIXTS字段设置为索引
    return df

def readfile(filename):
    if(filename[-3:]=='csv'):
        data=pd.read_csv(filename)
    #txt
    else:
        data = pd.read_csv(filename,sep=" ")
    return data


def preprocess(data):
    mini = data['UNIXTS'].min()
    maxi = data['UNIXTS'].max()
    mini_date = timestamp2datetime(mini)
    maxi_date = timestamp2datetime(maxi)
    delta = calDelta(maxi_date, mini_date)
    return [mini_date,maxi_date,delta]


def timestamp2datetime(timestamp):
    ts = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return ts

def strtodatetime(time):
    return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')


def calDelta(t1, t2):
    delta = strtodatetime(t1) - strtodatetime(t2)
    return delta.total_seconds()

def calfrequency(arr,df):
    fre = []
    for item in arr:
        fre.append(len(df[item]))
    return fre

def indextolist(period, h=False):
    temps = []
    temp = list(period.index)
    if h == False:
        for item in temp:
            temps.append(str(item))
    else:
        for item in temp:
            temps.append(str(item)[:13])
    return temps



def readwholedata():
    data= Datainfo.objects.order_by('Name')
    info=[]
    for item in data:
        temp= []
        temp.append(item.return_type())
        temp.append(item.return_nodes())
        temp.append(item.return_temporal_edges())
        temp.append(item.return_static_edges())
        temp.append(item.return_description())
        info.append(temp)
    return info

def readspecificdata(id):
    item = get_object_or_404(SubData, pk=id)
    temp= []
    super=item.return_super()
    # temp.append(item.return_name())
    temp.append(item.return_nodes())
    temp.append(item.return_temporal_edges())
    temp.append(item.return_static_edges())
    temp.append(item.return_timespan())
    temp.append(super.return_type())
    temp.append(super.return_description())
    print(temp)
    return temp


def generateImg(df,type,fn):
    df_time = df.resample(type).sum().to_period(type)  # 按年进行统计加和
    index=indextolist(df_time)
    df_time['amount'] = np.array(calfrequency(index,df))
    fig = go.Figure(
    data=[go.Scatter(x=index,y=df_time['amount'])],
    layout_title_text=""+fn[:-4]
    )
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div




def changeChart(request):
    tg = request.GET.get("tg", None)
    fn = request.GET.get("fn", None)
    data = readfile("./project/files/" + fn)
    df=createdataframe(data)
    result=generateImg(df,tg,fn)
    data = {
        'fig': result
    }
    return JsonResponse(data)

def dftojson(df):
    nodes1 = df['SRC'][:250].unique()
    nodes2 = df['DST'][:250].unique()
    nodes = pd.DataFrame(np.unique(np.append(nodes1, nodes2)), columns=["id"])
    part1 = nodes.to_json(orient='records')
    part2 = df[:250].to_json(orient='records')
    jsondata = '{"nodes":' + part1 + ',"links":' + part2 + '}'
    return jsondata.replace("SRC", 'source').replace('DST', 'target')




