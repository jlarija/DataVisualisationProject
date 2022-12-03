from dash import Dash, html, dcc, Input, Output
from utils import *
import json
import pickle
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px

# from dashboard import *

# DATA 
data_not_saved = True
if data_not_saved:
    df, variables_each_country = get_preprocessed_df()
    with open('df.pickle', 'wb') as file:
        pickle.dump([df, variables_each_country], file)

with open('df.pickle', 'rb') as dffile:
    df, variables_each_country = pickle.load(dffile)

months_list = get_list_months(df)
slider_months = [month[:3] + month[5:] for month in months_list]
months_df = get_month_df(df)

app = Dash(__name__)

df = get_month_df(df)  # Split months cause slider
df = df.groupby(['iso_code', 'month'], sort=False).mean().reset_index()
df = df[df['iso_code'].str.contains('OWID') == False]
print(df.columns)
app.layout = html.Div([html.H1('A look at the world:'),
                       dcc.Dropdown(df.columns, 'total_cases', id='chorplethdropdown'),
                       dcc.Graph(id='Choropleth Map'),
                       dcc.Slider(
                           0,
                           len(months_list) - 1,
                           marks={i: str(slider_months[i]) for i in range(len(slider_months))},
                           updatemode='mouseup',
                           value=0,
                           id='monthchoroplethmap'
                       )
                       ])


@app.callback(
    Output('Choropleth Map', 'figure'),
    Input('chorplethdropdown', 'value'),
    Input('monthchoroplethmap', 'value')  # gives a numerical value
)
def choropleth_map(choroplethdropdown, monthchoroplethmap):
    global df

    colorscale = ['#ffd7cd', '#e3ada0', '#c68475', '#a95c4c', '#893427', '#690000']

    current_month = months_list[monthchoroplethmap]
    my_df = df[df['month'] == current_month]
    min_color = np.max(my_df[str(choroplethdropdown)])
    max_color = np.min(my_df[str(choroplethdropdown)])

    fig = px.choropleth(my_df, locations='iso_code', color=str(choroplethdropdown),
                        color_continuous_scale=colorscale, hover_name="iso_code", height=600, width=1000,
                        range_color=(min_color, max_color))

    background_color = '#F5F2E8'

    fig.update_layout(font_family='Balto', font_color='#000000',
                      font_size=18,
                      geo=dict(
                          showframe=False,
                          showcoastlines=False,
                          countrycolor='#000000',
                          bgcolor=background_color,
                          lakecolor=background_color,
                          landcolor='rgba(51,17,0,0.2)',
                          subunitcolor='grey',

                      ))

    # Delete antartica
    fig.add_trace(go.Choropleth(locations=['ATA'],
                                z=[0],
                                colorscale=[[0, background_color], [1, background_color]],
                                marker_line_color=background_color,
                                showlegend=False,
                                showscale=False)
                  )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
