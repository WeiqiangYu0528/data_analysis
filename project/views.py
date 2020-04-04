import math
import random

from django.shortcuts import render, get_object_or_404


# Create your views here.
from django.http import JsonResponse, FileResponse
import numpy as np
import pandas as pd
import datetime
import os
from collections import Counter


from django.views.decorators.csrf import csrf_exempt
from scipy.spatial import distance
from sklearn.manifold import MDS
from itertools import groupby
from .models import Datainfo, SubData,GraphInfo,otherData
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import json
import community
import time

import networkx as nx
from networkx.algorithms.community import k_clique_communities

datasets=[]
for item in otherData.objects.all():
    dataset=[]
    dataset.append(item.return_id())
    dataset.append(item.return_name())
    datasets.append(dataset)
print(datasets)


def index(request):
    global datasets
    path="pages/index.html"
    infos=readwholedata()
    heatmap_whole,dataspace_whole=return_datasetsimilarities()
    context={
            "info":infos,
             "wholeheatmap": heatmap_whole,
             "wholedataspace": dataspace_whole,
             "datasets": datasets,
             }
    return render(request,path,context)

def document(request,type,id):
    global datasets
    if(type=='s'):
        item = get_object_or_404(SubData, pk=id)
        filename = item.return_name()
        data=readfile(filename)[['SRC','DST','UNIXTS']]
        obj=item.return_super()
        description=obj.return_description()
        types=obj.return_type()
    elif(type=='o'):
        item = get_object_or_404(otherData, pk=id)
        filename = item.return_name()
        data=readotherfile(filename)[['SRC','DST','UNIXTS']]
        description=item.return_description()
        types=item.return_types()
    else:
        return render(request,'pages/404.html')
    path = "pages/document.html"
    min_date=item.return_mindate()
    max_date=item.return_maxdate()
    details=[min_date,max_date,item.return_maxdegree(),item.return_mindegree(),item.return_averagedegree(),description]
    data_info = [[item for item in row ] for index, row in data.head(12).iterrows()]
    info = [types,item.return_nodes(), item.return_temporal_edges(), item.return_static_edges(), item.return_timespan()]
    # if(filename=="soc-sign-bitcoinalpha.csv"):
    #     details.append("-10 to +10")
    #     details.append("93%")
    # if (filename == "soc-sign-bitcoinotc.csv"):
    #     details.append("-10 to +10")
    #     details.append("89%")
    dates=[min_date.month,min_date.year,max_date.month,max_date.year]
    context={
            "datasets":datasets,
            "data_info":data_info,
             "filename":filename,
             "details":details,
             "info":info,
             "fig":item.return_distribution(),
             "nodes":item.return_force_nodes(),
            "links":item.return_force_links(),
             "dates":dates,
             "community":json.loads(item.return_communities()),
             "heatmap":item.return_heatmap(),
             "dataspace":item.return_dataspace(),
              "size":item.return_size(),
             "total":item.return_total_communities(),
              "prediction":item.return_prediction()
             }
    return render(request,path, context)

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

def return_df(filename):
    data=readfile(filename)
    unixts=createunixts(data)
    df = data[['SRC', 'DST']]
    df.index = unixts.index
    return df,unixts

def return_time(df,unixts):
    df_time = unixts.resample('M').sum().to_period('M')  # 按月度进行统计加和
    times=indextolist(df_time)
    empty=[row for row in times if len(df[row])<=0]
    times=[time for time in times if time not in empty]
    return times


def readfile(filename):
    if(filename[-3:]=='csv'):
        data=pd.read_csv("static/files/" +filename)
    else:
        data = pd.read_csv("static/files/" +filename,sep=" ")
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

def indextolist(period):
    temp = list(period.index)
    temps=[str(item) for item in temp]
    return temps

def readwholedata():
    data= Datainfo.objects.order_by('Name')
    info=[]
    for item in data:
        temp= [item.return_id(),item.return_name(),item.return_type(),item.return_nodes(),item.return_temporal_edges(),item.return_static_edges(),item.return_description()]
        info.append(temp)
    return info

def ocpm(df,K):
    G=createGraphs(df[:500])
    return G,k_clique_communities(G, K)

def olcpm(df,K):
    g,result=ocpm(df,K)
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

def community_detection(df,method,duration,k):
    try:
        if method == "l":
            communities, size = louvain(df[duration])
        else:
            temp = olcpm(df[duration], k)
            communities = []
            if temp is not None:
                for val in temp:
                    communities.append((list(val)))
            size = len(communities)
        # comm_page = []
        # for com in communities:
        #     comm_page.append("<p>{}</p>".format(com))
        nodes, links = dftojson(df, duration, communities)
        data = {
            'nodes': nodes,
            'links': links,
            'coms': communities,
            "size": size,
            "method": 'cd'
        }
    except (ValueError, KeyError):
        data = {
            "error": True,
            "message": "No data in the time span of "+duration
        }
    return data

def upperBound(G):
    v=len(G.nodes)
    e=len(G.edges)
    return  math.floor((1+math.sqrt(9+8*(e-v)))/2)

def candidateGeneration(graph, v, k):
    c = [v]
    d = {}
    q = list(graph[v])
    d[1] = q.copy()
    i = 0
    maxi = 1
    while len(q) != 0:
        # remove node
        node = q.pop(0)
        d[maxi].remove(node)
        if len(d[maxi]) == 0:
            d.pop(maxi)
            if len(d) != 0:
                maxi = sorted(d.keys())[-1]
        # add to the c
        c.append(node)
        # return c or not
        H = graph.subgraph(c)
        minimum_degree = sorted([du for n, du in H.degree()])[0]
        if minimum_degree >= k:
            return True, c

        for n in graph[node]:
            # whether the node in community
            if n not in c:
                if n in q:
                    # update dict
                    for key, val in list(d.items()):
                        if n in val:
                            d[key].remove(n)
                            if len(d[maxi]) == 0:
                                d.pop(maxi)
                                if len(d) != 0:
                                    maxi = sorted(d.keys())[-1]

                            newk = key + 1
                            if newk not in d.keys():
                                d[newk] = [n]
                                if newk > maxi:
                                    maxi = newk
                            else:
                                d[newk].append(n)
                            break
                # add to the dict!!!
                elif graph.degree[n] >= k:
                    count = 0
                    for w in graph[n]:
                        if w in c:
                            count = count + 1
                    if count not in d.keys():
                        d[count] = [n]
                        if count > maxi:
                            maxi = count
                    else:
                        d[count].append(n)

        temp = list(d[k] for k in sorted(d.keys(), reverse=True))
        q = []
        for item in temp:
            for vertex in item:
                q.append(vertex)
        #         if len(d)!=0:
        #             maxi=sorted(d.keys())[-1]
        i = i + 1
    return False, H

def globalsearch(graph,v,k):
    components=nx.connected_components(nx.k_core(graph,k))
    for i in components:
        if v in i:
            return True,list(i)
    return False,"No community has been found with the given query node"

def cst(df,duration,v,k):
    graph=createGraphs(df[duration][:500])
    if k >upperBound(graph):
        return {
            "error": True,
            "message": "error, k is greater than the upperBound"
               }
    try:
        flag,C=candidateGeneration(graph,v,k)
    except KeyError:
        return {
            "error": True,
            "message": "No community has been found with the given query node"
        }
    if not flag:
        f,C=globalsearch(C,v,k)
    else:
        f=True
    if f:
        nodes, links = dftojson(df, duration, [C])
        return {
            'nodes': nodes,
            'links': links,
            'coms': [C],
            "size": 1,
            "method":'cs'
        }

    else:
        return {
            "error": True,
            "message": C
        }

def changed3(request):
    method=request.GET.get("method", None)
    k = int(request.GET.get("k", None))
    fn = request.GET.get("fn", None)
    date=request.GET.get("date",None)
    v=int(request.GET.get('v',None))
    print(date)
    df, i_ = return_df(fn)
    if method=='cs':
        data=cst(df,date,v,k)
    else:
        data=community_detection(df,method,date,k)
    return JsonResponse(data,encoder=NpEncoder)


def dftojson(old_df,duration,communities):
    df=old_df.rename(columns={'SRC': 'source', 'DST': 'target'},inplace=False)
    nodes1 = df[duration]['source'][:500].unique()
    nodes2 = df[duration]['target'][:500].unique()
    nodes = pd.DataFrame(np.unique(np.append(nodes1, nodes2)), columns=["id"])
    nodes['group'] = 0
    if communities is not None:
        for index, coms in enumerate(communities):
            for node in coms:
                inx = nodes[nodes['id'] == node].index[0]
                if nodes.iloc[inx]['group']==0:
                    nodes.iloc[inx]['group']=index+1
                else:
                    nodes.iloc[inx]['group']=-1
    part1 = nodes.to_json(orient='records')
    part2 = df[duration][:500].to_json(orient='records')
    return part1,part2



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



def degree_similarity(df,times,measure):
    print(measure)
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

def vertexcount(df,times):
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


def combine_similarity(df,times,m1,m2,r):
    if m1=='vc':
        s1=vertexcount(df,times)
        s2=degree_similarity(df,times,m2)
    elif m2=='vc':
        s2 = vertexcount(df,times)
        s1 = degree_similarity(df,times, m1)
    else:
        s1 = degree_similarity(df,times, m1)
        s2 = degree_similarity(df,times, m2)
    print(s1,s2)
    print(r,1-r)
    return np.multiply(s1,r)+np.multiply(s2,1-r)


def apollo(unixts,df,measure,m1=None,m2=None,r=None):
    times=return_time(df,unixts)
    if measure=="cs":
        heatmap=combine_similarity(df,times,m1,m2,r)
    elif measure=="vc":
        heatmap=vertexcount(df,times)
    else:
        heatmap = degree_similarity(df,times,measure)
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
    return heatmap,plot_div1,plot_div2

def change_apollo(request):
    sm=request.GET.get("sm",None)
    fn = request.GET.get("fn", None)
    fr=request.GET.get('from',None)
    to=request.GET.get('to',None)
    df,unixts=return_df(fn)
    if(sm=="cs"):
        m1=request.GET.get("m1", None)
        m2=request.GET.get("m2", None)
        r =float(request.GET.get('r'))
        heatmap,g1,g2=apollo(unixts[fr:to],df[fr:to],sm,m1,m2,r)
    else:
        heatmap,g1,g2=apollo(unixts[fr:to],df[fr:to],sm)
    data = {
        'heatmap': g1,
        'space':g2,
        'info':heatmap,
    }
    return JsonResponse(data,encoder=NpEncoder)

@csrf_exempt
def prediction(request):
    fn = request.POST.get("fn", None)
    fr=request.POST.get('from',None)
    to=request.POST.get('to',None)
    o=request.POST.get('o',None)
    p=float(request.POST.get('p',None))
    heatmap=json.loads(request.POST.get('info',None))
    print(heatmap)
    df,unixts=return_df(fn)
    months=return_time(df[fr:to],unixts[fr:to])
    g3=model(df[fr:to],p,months,heatmap,o)
    data={
        'prediction':g3
    }
    return JsonResponse(data, encoder=NpEncoder)


def createGraphs(df):
    g = nx.Graph()
    g.clear()
    for index, row in df.iterrows():
            g.add_edge(row['SRC'],row['DST'])
    g.remove_edges_from(g.selfloop_edges())
    graph = max(nx.connected_component_subgraphs(g), key=len)
    return graph

def louvain(df):
    g = createGraphs(df[:500])
    partition = community.best_partition(g)
    res = sorted(partition.items(), key=lambda d: d[1], reverse=True)
    listOfThings = []
    for key, items in groupby(res, lambda x: x[1]):
        listOfThings.append([thing[0] for thing in items])
    size=len(listOfThings)
    return listOfThings,size

def doc(request):
    return render(request, "pages/start.html")

def download(request,filename):
    file = open("static/files/"+filename, 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="'+filename+'"'
    return response

def zipdownload(request):
    file = open("static/others/files.zip", 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="Files.zip"'
    return response


def handler404(request,exception):
    return render(request, 'pages/404.html')


# def error(request):
#     return render(request, "pages/404.html")
def caloperator(df, operator):
    g = createGraphs(df)
    if operator == "d":
        output = nx.diameter(g)
    if operator == "s":
        output = nx.average_shortest_path_length(g)
    return output

def model(df,p,months,s,operator):
    print(p,s)
    length=len(months)
    ceilval=math.ceil(length*p)
    # if ceilval < 10:
    #     ceilval=10
    time_start = time.time()
    t={}
    print(ceilval)
    if length<=3:
        for inx in range(length):
            t[inx] = caloperator(df[months[inx]], operator)
    else:
        for i in range(ceilval):
            index=random.randint(0,length-1)
            if index not in t.keys():
                t[index]=caloperator(df[months[index]],operator)
                print(index, t[index])
        while len(t)<3:
            index=random.randint(0,length-1)
            if index not in t.keys():
                t[index]=caloperator(df[months[index]],operator)
        time_end = time.time()
        print('totally cost', time_end - time_start)
        for inx in range(length):
            counts=0
            sum_wi=0
            sum_value=0
            if inx not in t.keys():
                x=np.array(s)[inx]
                for i in np.argsort(x)[::-1]:
                    if i in t.keys():
                        counts=counts+1
                        sum_value=sum_value+x[i]*t[i]
                        sum_wi=x[i]+sum_wi
                    if counts >=3:
                        t[inx]=sum_value/sum_wi
                        break
    # final={k:t[k] for k in sorted(t.keys())}
    y_values=[t[k] for k in sorted(t.keys())]
    if operator=='d':
        text='Diameter'
    if operator=='s':
        text='Average Shortest Path'
    fig = go.Figure(
        data=[go.Scatter(x=months, y=y_values)],
        layout_title_text=text
    )
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    print(plot_div)
    return plot_div

def extra(request):
    global datasets
    context={
        'datasets':datasets
    }
    return render(request, "pages/form-uploads.html",context)

# def upload(request):
#     if request.method == 'POST' and request.FILES['myfile']:
#         myfile = request.FILES['myfile']
#         fs = FileSystemStorage()
#         filename = fs.save(myfile.name, myfile)
#         uploaded_file_url = fs.url(filename)
#         return render(request, 'core/simple_upload.html', {
#             'uploaded_file_url': uploaded_file_url
#         })
#     return render(request, 'core/simple_upload.html')

def upload_file(file):
    if not file:
        return False
    destination = open("static/datasets/"+file.name,'wb+')    # 打开特定的文件进行二进制的写操作
    for chunk in file.chunks():      # 分块写入文件
        destination.write(chunk)
    destination.close()
    return True


@csrf_exempt
def upload(request):
    global datasets
    id = int(datasets[-1][0]) + 1
    name=request.POST.get('dname')
    type=request.POST.get('dtype')
    nodes=int(request.POST.get('node'))
    tedges=int(request.POST.get('tedge'))
    sedges=int(request.POST.get('sedge'))
    tspan=request.POST.get('tspan')
    description=request.POST.get('des')
    file = request.FILES.get("file", None)  # 获取上传的文件，如果没有文件，则默认为None
    success=upload_file(file)
    if success:
        try:
            data=readotherfile(name)
            statistic=preprocess(data)
            df,unixts=return_odf(name)
            months=return_time(df,unixts)
            h,div1,div2=apollo(unixts, df, 'bc')
            mini=statistic[0]
            maxi=statistic[1]
            div3= generateImg(df,'m', name, mini[:4], maxi[:4])
            print(mini,maxi)
            result=community_detection(df,'l',mini[:4],3)
            # print(result['nodes'],,,result['method'])
            div4=caltotal(name,df,months)
            div5=model(df,0.2,months,h,'d')
            otherData.objects.create(
            Name = name,
            types= type,
            Nodes = nodes,
            Temporal_Edges = tedges,
            Static_Edges = sedges,
            TimeSpan = tspan,
            min_date = datetime.datetime.strptime(mini, "%Y-%m-%d %H:%M:%S"),
            max_date =datetime.datetime.strptime(maxi, "%Y-%m-%d %H:%M:%S"),
            max_degree = statistic[2],
            min_degree = statistic[3],
            average_degree = statistic[4],
            heatmap = div1,
            dataspace = div2,
            total_communities = div4,
            force_nodes = result['nodes'],
            force_links = result['links'],
            communities = result['coms'],
            size = result['size'],
            distribution = div3,
            prediction=div5,
            description=description,
            )
            datasets.append([id,name])
            print('success')
        except Exception as e:
             print(e)
             os.remove('static/datasets/'+name)
             print('error')
        context={
                 'id':id
             }
    else:
        return "Fail to upload the file"
    return JsonResponse(context,NpEncoder)

# def preprocess(data):
#     mini = data['UNIXTS'].min()
#     maxi = data['UNIXTS'].max()
#     mini_date = timestamp2datetime(mini)
#     maxi_date = timestamp2datetime(maxi)
#     values1 = data['SRC'].values.tolist()
#     values2 = data['DST'].values.tolist()
#     values = Counter(values1) + Counter(values2)
#     max_degree= values.most_common()[0][1]
#     min_degree= values.most_common()[-1][1]
#     avg_degree= round(len(data)*2/len(values))
#     # delta = calDelta(maxi_date, mini_date)
#     return [mini_date,maxi_date,max_degree,min_degree,avg_degree]
def caltotal(fn,df,months):
    pics=[]
    for month in months:
        fin,size=louvain(df[month])
        pics.append(size)
    fig = go.Figure(
        data=[go.Scatter(x=months, y=pics)],
        layout_title_text="" + fn[:-4]
    )
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    print(plot_div)
    return plot_div


def readotherfile(filename):
    if(filename[-3:]=='csv'):
        data=pd.read_csv("static/datasets/" +filename)
    else:
        data = pd.read_csv("static/datasets/" +filename,sep=" ")
    return data

def return_odf(filename):
    data=readotherfile(filename)
    unixts=createunixts(data)
    df = data[['SRC', 'DST']]
    df.index = unixts.index
    return df,unixts