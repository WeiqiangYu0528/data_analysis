import json
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

my_data=pd.read_json("../a.json",encoding="utf-8", orient='records')
fig = go.Figure(
    data=[go.Scatter(x=my_data.index, y=my_data['amount'])],
    layout_title_text="Total Community Number"
)
fig.show()
plot_div = plot(fig, output_type='div', include_plotlyjs=False)
print(plot_div)