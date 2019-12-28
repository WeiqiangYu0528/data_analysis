import math
import random

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

from .models import Datainfo, SubData,GraphInfo
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
    files.remove(".DS_Store")
    files.sort()
    filename = request.GET.get('fn')
    if filename is None:
        filename="soc-sign-bitcoinalpha.csv"
        path="pages/index.html"
    else:
        path = "pages/document.html"
    # filename = request.GET.get('fn', "sx-superuser.txt")
    data=readfile(filename)
    details = preprocess(data)
    data_info = []
    infos=readwholedata()
    id=0
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
    for index, row in data.head(5).iterrows():
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
    dataspace = info.pop(-1)
    heatmap=info.pop(-1)
    # heatmap,dataspace=apollo(unixts,df)
    heatmap_whole,dataspace_whole=return_datasetsimilarities()
    context={
            "data_info":data_info,
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
             "dataspace":dataspace,
             "wholeheatmap": heatmap_whole,
             "wholedataspace": dataspace_whole,
             }
    path="pages/index1.html"
    return render(request,path , context)


def return_datasetsimilarities(id=1):
    data = get_object_or_404(GraphInfo, pk=id)
    temp=[]
    temp.append(data.return_heatmap())
    temp.append(data.return_dataspace())
    return temp



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
        data=pd.read_csv("./project/files/" +filename)
    #txt
    else:
        data = pd.read_csv("./project/files/" +filename,sep=" ")
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
    temp.append(item.return_heatmap())
    temp.append(item.return_dataspace())
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
    print(plot_div)
    return plot_div


def changeChart(request):
    tg = request.GET.get("tg", None)
    fn = request.GET.get("fn", None)
    fr=request.GET.get('from',None)
    to=request.GET.get('to',None)
    data = readfile(fn)
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
    data = readfile(fn)
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



def bhattacharyya(p, q):
    """ Bhattacharyya distance between distributions (lists of floats). """
    p_len=len(p)
    q_len=len(q)
    if(p_len<q_len):
        nums=[i+1 for i in range(p_len,q_len)]
        r=pd.DataFrame(index=nums,data=0,columns=['count'],)
        p=pd.concat([p,r])
    elif(p_len>q_len):
        nums=[i+1 for i in range(q_len,p_len)]
        r=pd.DataFrame(index=nums,data=0,columns=['count'],)
        q=pd.concat([q,r])
    return sum((math.sqrt(u * w) for u, w in zip(p['count'], q['count'])))



def degree_similarity(times,df,measure):
    heatmap=[]
    for row in times:
            heat=[]
            row_=df[row]
            dis1=degree_distribution(row_)
            for col in times:
                col_=df[col]
                dis2=degree_distribution(col_)
                if measure == 'js':
                    score=1-js_distance(dis1,dis2)
                if measure == 'bc':
                    score=bhattacharyya(dis1,dis2)
                heat.append(score)
            heatmap.append(heat)
    return heatmap

def vertexcount(times,df):
    heatmap=[]
    for row in times:
            heat=[]
            row_=df[row].count()[0]
            for col in times:
                col_=df[col].count()[0]
                if row_ < col_:
                    heat.append(row_/col_)
                else:
                    heat.append(col_/row_)
            heatmap.append(heat)
    return heatmap



def apollo(unixts,df,measure):
    df_time = unixts.resample('M').sum().to_period('M')  # 按月度进行统计加和
    times = indextolist(df_time, h=False)
    empty = []
    for row in times:
        if len(df[row]) <= 0:
            empty.append(row)
    times = [time for time in times if time not in empty]
    if measure=="vc":
        heatmap = vertexcount(times, df)
    else:
        heatmap = degree_similarity(times, df,measure)
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
             color='category',color_discrete_sequence=px.colors.qualitative.Set3
                       )

    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    return plot_div1,plot_div2

def changed_apollo(request):
    sm=request.GET.get("sm",None)
    fn = request.GET.get("fn", None)
    begin=request.GET.get('begin',None)
    end=request.GET.get('end',None)
    data = readfile(fn)
    unixts=createunixts(data)
    df=createdf(data,unixts)
    g1,g2=apollo(unixts[begin:end],df[begin:end],sm)
    data = {
        'heatmap': g1,
        'space':g2
    }
    return JsonResponse(data,encoder=NpEncoder)




# def total_communities(id):
#     item = get_object_or_404(SubData, pk=id)
#     return item.return_communities()


def calolcpm(df,k,duration):
    g = nx.Graph()
    g.clear()
    events = []
    for index, row in df[duration].iloc[:500].iterrows():
        events.append((row['SRC'], row['DST'], '+'))
    return olcpm(k, g, events)

def model(p,months,df,s,g,operator):
    p=p
    length=len(months)
    ceilval=math.ceil(length*p)
    if ceilval < 10:
        ceilval=10
    t={}
    indices=[]
    for i in range(ceilval):
        index=random.randint(0,length-1)
        key,value=caloperator(df,months[index],g,operator)
        t[key]=value
        if index not in indices:
            indices.append(index)
#     print(indices)
    for inx,month in enumerate(months):
        counts=0
        sum_wi=0
        sum_value=0
        if inx not in indices:
            x=np.array(s)[inx]
            for i in np.argsort(x)[::-1]:
                if i in indices:
                    m=months[i]
                    if counts >=3:
                        if operator == "d":
                            t[month] = int(sum_value / sum_wi)
                        if operator == "s":
                            t[month]=sum_value/sum_wi
                        indices.append(inx)
                        break
                    else:
                        operator_value=t[m]
#                         print(m,x[i],diameter)
                        counts=counts+1
                        sum_value=sum_value+x[i]*operator_value
                        sum_wi=x[i]+sum_wi
#             print("-----",indices)
#     print(diameters)
    return { k:t[k] for k in sorted(t.keys())}

def caloperator(df, month, g, operator):
    g.clear()
    for index, row in df[month].iloc[:].iterrows():
        g.add_edge(row['SRC'], row['DST'])
    if nx.is_connected(g):
        if operator == "d":
            output = nx.diameter(g)
        if operator == "s":
            output = nx.average_shortest_path_length(g)
    #             print("# Diameter:" + str(diameter))
    else:
        G = max(nx.connected_component_subgraphs(g), key=len)
        if operator == "d":
            output = nx.diameter(G)
        if operator == "s":
            output = nx.average_shortest_path_length(G)
    #             print("# Diameter:" + str(diameter))
    return month, output


