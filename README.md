# Temporal Graph Management System

### 使用方法

首先下载项目文件,解压文件

这个项目用到Django, NetworkX, python-louvain, Plotly 等包，在运行程序之前，请先确认你已经安装好了所需的包

然后在有manage.py的目录下运行
```
python3 manage.py runserver
```
如果没有报错，则可成功运行。如果显示有包未安装，则先安装包再运行一次命令

### 数据库及相关内容

数据文件以文件的形式存储。该项目自带的文件，存放的位置为 static/files

要访问数据库，进入admin页面，如下：

```
http://127.0.0.1:8000/admin
```

username: super， password: 123456

之后即可浏览数据库存储的数据
