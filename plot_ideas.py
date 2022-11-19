import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
from sklearn import linear_model
import plotly.express as px
from datetime import datetime, timedelta
import math
from plotly.subplots import make_subplots
import plotly.io as pio
import random
import plotly.graph_objects as go
from utils import *

app = Dash(__name__)

df, variables_each_country = get_preprocessed_df()

variables_first_country = variables_each_country[df['location'][0]]
app.layout = html.Div([
    html.Div([
        html.Label("Country or continent"),
        dcc.RadioItems(['Country', 'Continent'], 'Country', id='country-continent-radio'),
        dcc.Dropdown([country for country in df['location'].unique()], df['location'][0], id='country-continent-choice'),

        html.Br(),
        html.Label("Variables to plot"),
        dcc.Dropdown(variables_first_country, variables_first_country[0], id='y-axis', multi=True)
    ]),

    dcc.Graph(id='variables-graph')
])


@app.callback(
    Output('country-continent-choice', 'options'),
    Output('country-continent-choice', 'value'),
    Input('country-continent-radio', 'value'))
def choose_country_or_continent(country_continent):
    if country_continent == 'Country':
        to_show = df['location'].unique()
    else:
        to_show = df['continent'].dropna().unique()
    return to_show, to_show[0]


@app.callback(
    Output('y-axis', 'options'),
    Output('y-axis', 'value'),
    Input('country-continent-radio', 'value'),
    Input('country-continent-choice', 'value'))
def set_data_pred_value(country_cont_radio, country_cont_choice):
    if country_cont_radio == 'Country':
        variables_to_show = variables_each_country[country_cont_choice]
    else:
        variables_to_show = df.columns.to_list()
    for col in columns_to_remove:
        if col in variables_to_show:
            variables_to_show.remove(col)
    return variables_to_show, [variables_to_show[0], variables_to_show[1]]


@app.callback(
    Output('variables-graph', 'figure'),
    Input('y-axis', 'value'),
    Input('country-continent-choice', 'value'),
    Input('country-continent-radio', 'value'))
def update_graph(variables_chosen, country_cont_choice, country_cont_radio):
    if country_cont_radio == 'Country':
        used_df = df[df['location'] == country_cont_choice]
    else:
        continent_df = df[df['continent'] == country_cont_choice]
        used_df = continent_df.groupby(['date'], as_index=False).sum()

    fig = go.Figure()

    dates = used_df['date']
    for i in range(len(variables_chosen)):
        if i == 0:
            fig.add_trace(go.Scatter(
                x=dates,
                y=used_df[variables_chosen[i]],
                name=variables_chosen[i],
            ))
        else:
            fig.add_trace(go.Scatter(
                x=dates,
                y=used_df[variables_chosen[i]],
                name=variables_chosen[i],
                yaxis="y" + str(i + 1)
            ))

    layout = {}
    color_hex = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
    layout['yaxis'] = {'tickfont': {'color': color_hex},
                       'title': {'font': {'color': color_hex}, 'text': variables_chosen[0]}}
    layout['xaxis'] = {'domain': [0.3, 0.9]}

    for i in range(len(variables_chosen)):
        if i == 0:
            continue
        else:
            color_hex = "#" + ''.join([random.choice('ABCDEF0123456789') for j in range(6)])
            pos = i * 0.3 / len(variables_chosen)
            layout['yaxis' + str(i + 1)] = {'anchor': 'free', 'position': pos, 'overlaying': 'y', 'side': 'left',
                                            'tickfont': {'color': color_hex},
                                            'title': {'font': {'color': color_hex}, 'text': variables_chosen[i]}}

    fig.update_layout(layout)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
