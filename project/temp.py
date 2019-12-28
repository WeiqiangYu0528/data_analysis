import plotly.express as px
import json
import pandas as pd
gapminder = px.data.gapminder()

my_data=pd.read_json("../a.json",encoding="utf-8", orient='records')
fig=px.scatter(my_data,y="amount", animation_frame="time",color="category",
           size="amount", size_max=55, range_y=[0,300000])
fig.show()
