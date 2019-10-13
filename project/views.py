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
from collections import Counter
from django.views.generic import CreateView
from scipy.spatial import distance
from sklearn.manifold import MDS

from .models import Datainfo, SubData
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import json

import networkx as nx
from itertools import combinations
from networkx.algorithms.community import k_clique_communities
import time



def index(request):
    files=os.listdir("./project/files")
    # files.remove(".DS_Store")
    files.sort()
    filename = request.GET.get('fn', "soc-sign-bitcoinalpha.csv")
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
    mini=details[0][:4]
    maxi=details[1][:4]
    time_range=[]
    mini_val=int(mini)
    maxi_val=int(maxi)
    for year in range(mini_val,maxi_val+1):
        time_range.append(year)
    unixts=createunixts(data)
    fig=generateImg(unixts,graph_type,filename,mini,maxi)
    df = createdf(data, unixts)
    communities=calolcpm(df,3,mini)
    graph = dftojson(df, mini,communities)
    heatmap,dataspace=apollo(unixts,df)

    context={"data_info":data_info,
             "filename":filename,
             "details":details,
             "type":graph_type,
             "files":files,
             "info":info,
             "fig":fig,
             "graph":graph,
             "time":time_range,
             "community":communities,
             "heatmap":heatmap,
             "dataspace":dataspace
             }
    return render(request, 'pages/index.html', context)


def createunixts(data):
    df = pd.DataFrame(data['UNIXTS'].apply(timestamp2datetime))
    df['UNIXTS'] = pd.to_datetime(df['UNIXTS'])  # 将UNIXTS字段转化为日期类型
    df = df.set_index('UNIXTS')  # 将UNIXTS字段设置为索引
    return df

def createdf(data,unixts):
    df = data[['SRC', 'DST']]
    df.index = unixts.index
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
    values1 = data['SRC'].values.tolist()
    values2 = data['DST'].values.tolist()
    values = Counter(values1) + Counter(values2)
    max_degree= values.most_common()[0][1]
    min_degree= values.most_common()[-1][1]
    avg_degree= round(len(data)*2/len(values))
    # delta = calDelta(maxi_date, mini_date)
    return [mini_date,maxi_date,max_degree,min_degree,avg_degree]


def timestamp2datetime(timestamp):
    ts = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return ts

def strtodatetime(time):
    return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')


def calDelta(t1, t2):
    delta = strtodatetime(t1) - strtodatetime(t2)
    return delta.days

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
    temp.append(item.return_nodes())
    temp.append(item.return_temporal_edges())
    temp.append(item.return_static_edges())
    temp.append(item.return_timespan())
    temp.append(super.return_type())
    temp.append(super.return_description())
    return temp


def ocpm(K, G, SE):
    for item in SE:
        len_item = len(item)
        if len_item == 2:
            if item[1] == "+":
                G.add_node(item[0])
            elif item[1] == "-":
                G.remove_node(item[0])
        elif len_item == 3:
            if item[2] == "+":
                G.add_edge(item[0], item[1])
            elif item[2] == "-":
                G.remove_edge(item[0], item[1])
    return k_clique_communities(G, K)

def olcpm(K,g,SE):
    result=ocpm(K,g,SE)
#     print(result)
    existed_nodes=set()
    communities=[]
    for items in result:
        communities.append(items)
        for item in items:
            existed_nodes.add(item)
#     print(communities)
    if len(communities)!=0:
        node=set(g.nodes())
        others=node-existed_nodes
        temps=[]
        for inx,val in enumerate(communities):
            endpoints=set([*val])
            other=set(node-(existed_nodes-endpoints))
            newgraph=g.subgraph(other)
            temps.append(findmini(inx,endpoints,newgraph,others))
        for point in others:
            mini=None
            for inx,temp in enumerate(temps):
                for value in temp:
                    if point==value[1]:
                        if mini is None:
                            mini=value[2]
                            community=[inx]
                        elif mini>value[2]:
                            mini=value[2]
                            community=[inx]
                        elif mini==value[2]:
                            community.append(inx)
            if mini is not None:
#                 print(point,community,mini)
                for index in community:
                    communities[index]=communities[index]|set([point])
        return communities

def findmini(num,ends,graph,other):
    results=[]
    for node in other:
            for inx,end in enumerate(ends):
                if nx.has_path(graph, node, end):
                    flag=False
                    distance=nx.shortest_path_length(graph,source=node,target=end)
                    if inx==0:
                        temp=distance
                    elif distance<temp:
                        temp=distance
                else:
                    flag=True
                    break
            if not flag:
                results.append([num,node,temp])
    return results



def generateImg(df,type,fn,fr,to):
    temp=df[fr:to]
    df_time = temp.resample(type).sum().to_period(type)  # 按年进行统计加和
    index=indextolist(df_time)
    df_time['amount'] = np.array(calfrequency(index,temp))
    fig = go.Figure(
    data=[go.Scatter(x=index,y=df_time['amount'])],
    layout_title_text=""+fn[:-4]
    )
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

def calolcpm(df,k,duration):
    g = nx.Graph()
    g.clear()
    events = []
    for index, row in df[duration].iloc[:500].iterrows():
        events.append((row['SRC'], row['DST'], '+'))
    return olcpm(k, g, events)

def changeChart(request):
    tg = request.GET.get("tg", None)
    fn = request.GET.get("fn", None)
    fr=request.GET.get('from',None)
    to=request.GET.get('to',None)
    data = readfile("./project/files/" + fn)
    df=createunixts(data)
    result=generateImg(df,tg,fn,fr,to)
    data = {
        'fig': result
    }
    return JsonResponse(data)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def changed3(request):
    k = int(request.GET.get("k", None))
    fn = request.GET.get("fn", None)
    year=request.GET.get('year',None)
    month=request.GET.get('month',None)
    if month!="0":
        duration=year+"-"+month
    else:
        duration = year
    data = readfile("./project/files/" + fn)
    unixts=createunixts(data)
    df=createdf(data,unixts)
    temp=calolcpm(df,k,duration)
    communities = []
    comm_page=[]
    if temp is not None:
        for val in temp:
            communities.append((list(val)))
    for com in communities:
        comm_page.append("<p>{}</p>".format(com))
    d3 = dftojson(df, duration,communities)
    data = {
        'd3js': d3,
        'olcpm':comm_page
    }
    return JsonResponse(data,encoder=NpEncoder)


def dftojson(df,duration,communities):
    nodes1 = df[duration]['SRC'][:500].unique()
    nodes2 = df[duration]['DST'][:500].unique()
    nodes = pd.DataFrame(np.unique(np.append(nodes1, nodes2)), columns=["id"])
    nodes['group'] = 0
    if communities is not None:
        for index, coms in enumerate(communities):
            for node in coms:
                inx = nodes[nodes['id'] == node].index[0]
                nodes.iloc[inx]['group'] = index
    part1 = nodes.to_json(orient='records')
    part2 = df[duration][:500].to_json(orient='records')
    jsondata = '{"nodes":' + part1 + ',"links":' + part2 + '}'
    return jsondata.replace("SRC", 'source').replace('DST', 'target')


def degree_distribution(data):
    values1=data['SRC'].values.tolist()
    values2=data['DST'].values.tolist()
    values=Counter(values1)+Counter(values2)
    max_degree=values.most_common()[0][1]
    degrees=Counter(list(values.values())).most_common()
    degree=pd.DataFrame(degrees,columns=['degree','count'])
    degree['count']=degree['count']/(degree['count'].sum())
    degree_index=list(degree['degree'])
    non_index=[i for i in range(1,max_degree+1) if i not in degree_index]
    degree_2=pd.DataFrame(non_index,columns=['degree'])
    degree_2['count']=0.0
    degree=pd.concat([degree,degree_2])
    degree=degree.set_index('degree')
    return degree.sort_values('degree')

def js_distance(p, q):
    p_len=len(p)
    q_len=len(q)
    if(p_len<q_len):
        nums=[i+1 for i in range(p_len,q_len)]
        r=pd.DataFrame(index=nums,data=0,columns=['count'],)
        p=pd.concat([p,r])
    else:
        nums=[i+1 for i in range(q_len,p_len)]
        r=pd.DataFrame(index=nums,data=0,columns=['count'],)
        q=pd.concat([q,r])
    return distance.jensenshannon(p['count'],q['count'])

def draw_headmap(times,df):
    heatmap=[]
    for row in times:
            heat=[]
            row_=df[row]
            dis1=degree_distribution(row_)
            for col in times:
                col_=df[col]
                dis2=degree_distribution(col_)
                heat.append(js_distance(dis1,dis2))
            heatmap.append(heat)
    return heatmap


def apollo(unixts,df):
    df_time = unixts.resample('M').sum().to_period('M')  # 按月度进行统计加和
    times = indextolist(df_time, h=False)
    empty = []
    for row in times:
        if len(df[row]) <= 0:
            empty.append(row)
    times = [time for time in times if time not in empty]
    heatmap=draw_headmap(times,df)
    fig1 = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=times,
        y=times,
        colorscale='RdYlBu'
    ))
    embedding = MDS(n_components=3)
    X_transformed = embedding.fit_transform(heatmap)
    dataspace=pd.DataFrame(X_transformed,columns=['x','y','z'])
    dataspace['text']=times
    dataspace['category']=dataspace['text'].apply(lambda x: x[:4])

    fig2 = px.scatter_3d(dataspace, x='x', y='y', z='z',text='text',
                         color='category'
                       )

    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    return plot_div1,plot_div2



