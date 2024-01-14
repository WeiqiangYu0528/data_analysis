# Temporal Graph Management System
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/WeiqiangYu0528/data_analysis/blob/master/README.md)
[![zh-cn](https://img.shields.io/badge/lang-zh--cn-blue.svg)](https://github.com/WeiqiangYu0528/data_analysis/blob/master/README.zh-cn.md)

如果您想阅读中文，请点击上方的“zh-cn”按钮。

### Manual
First, download the project files and extract them.

This project utilizes packages such as Django, NetworkX, python-louvain, Plotly, etc. Before running the program, please ensure that you have installed the required packages.

Then, navigate to the directory containing manage.py and run the following command:

```
python3 manage.py runserver
```

If no errors are reported, the program has been successfully executed. If any packages are not installed, install them first and then run the command again.

### Database and Related Content

Data files are stored in the form of files. The location for the project's files is static/files.

To access the database, go to the admin page as follows:

```
http://127.0.0.1:8000/admin
```
username: super， password: 123456

Afterwards, you can browse the data stored in the database.
