import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, dash_table
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

# Comment the next line and uncomment the 3 after to do your tests to avoid loading time
df, variables_each_country = get_preprocessed_df()
#url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
#df = pd.read_csv(url)
#variables_each_country = get_var_each_country()

temp_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

variables_first_country = variables_each_country[df['location'][0]]
app.layout = html.Div([
    html.H1(
        'Evolution of multiple variables in time',
        style={
            'textAlign': 'left',
            'color': 'black'
        }
    ),
    html.Div([
        html.Label("Country or continent"),
        dcc.RadioItems(['Country', 'Continent'], 'Country', id='country-continent-radio'),
        dcc.Dropdown([country for country in df['location'].unique()], df['location'][0],
                     id='country-continent-choice'),

        html.Br(),
        html.Label("Variables to plot"),
        dcc.Dropdown(variables_first_country, variables_first_country[0], id='y-axis', multi=True)
    ]),

    dcc.Graph(id='variables-graph'),

    html.Br(),
    html.H1(
        'Correlation of variables',
        style={
            'textAlign': 'left',
            'color': 'black'
        }
    ),
    html.Div([
        html.Label('Country'),
        dcc.Dropdown([country for country in df['location'].unique()], df['location'][0], id='country-choice')
    ], style={'width': '24%', 'float': 'left', 'display': 'inline-block'}),
    html.Div([
        html.Label('Variable'),
        dcc.Dropdown([var for var in variables_first_country], variables_first_country[0], id='var-choice'),
    ], style={'width': '24%', 'display': 'inline-block'}),
    html.Br(),
    html.Plaintext('Click on a coefficient to plot the corresponding variable'),
    html.Div([
        dash_table.DataTable(data=temp_df.to_dict('records'), columns=[{"name": i, "id": i, "selectable": True} for i in temp_df.columns],
                             id='corr-table')
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='corr-graph')
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})

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
def y_axis_based_on_location(country_cont_radio, country_cont_choice):
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
def update_graph_multi_var(variables_chosen, country_cont_choice, country_cont_radio):
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


@app.callback(
    Output('var-choice', 'options'),
    Output('var-choice', 'value'),
    Input('country-choice', 'value'))
def var_for_country(country_choice):
    variables_to_show = variables_each_country[country_choice]
    for col in columns_to_remove:
        if col in variables_to_show:
            variables_to_show.remove(col)
    for col in columns_fixed:
        if col in variables_to_show:
            variables_to_show.remove(col)
    return variables_to_show, variables_to_show[0]


@app.callback(
    Output('corr-table', 'data'),
    Output('corr-table', 'columns'),
    Input('country-choice', 'value'),
    Input('var-choice', 'value'))
def update_table_corr(country_choice, var_choice):
    all_features = variables_each_country[country_choice].copy()

    for column in columns_to_remove:
        if column in all_features:
            all_features.remove(column)

    for column in columns_fixed:
        if column in all_features:
            all_features.remove(column)

    country_df = df[df['location'] == country_choice][all_features]
    corr_mat = country_df.corr(method='pearson')

    pos_corr = corr_mat.sort_values(by=[var_choice], ascending=False, inplace=False)[var_choice]
    pos_corr = pos_corr[pos_corr < 0.9999]

    pos_variables_most_corr = []
    pos_corr_coeffs = []
    pos_all_vars = pos_corr.index
    for var in pos_all_vars:
        if 'hundred' not in var and 'smoothed' not in var and 'million' not in var and 'thousand' not in var:
            pos_variables_most_corr.append(var)
            pos_corr_coeffs.append(pos_corr[var])
            if len(pos_variables_most_corr) == 3:
                break

    neg_corr = corr_mat[corr_mat[var_choice] < 0.0]
    neg_corr = neg_corr.sort_values(by=[var_choice], ascending=True)[var_choice]

    neg_variables_most_corr = []
    neg_corr_coeffs = []
    neg_all_vars = neg_corr.index
    for var in neg_all_vars:
        neg_variables_most_corr.append(var)
        neg_corr_coeffs.append(neg_corr[var])
        if len(neg_variables_most_corr) == 3:
            break
    pos_variables_most_corr.extend(neg_variables_most_corr)
    pos_corr_coeffs.extend(neg_corr_coeffs)
    corr_dict = {'variables': pos_variables_most_corr,
                 'correlation coeff to ' + var_choice: pos_corr_coeffs}

    correlation_df = pd.DataFrame(corr_dict)
    correlation_df.set_index('variables')
    update_columns = [{"name": i, "id": i, "selectable": True} for i in correlation_df.columns]

    return correlation_df.to_dict('records'), update_columns


@app.callback(
    Output('corr-graph', 'figure'),
    Input('country-choice', 'value'),
    Input('var-choice', 'value'),
    Input('corr-table', 'active_cell'),
    Input('corr-table', 'data'))
def update_graphs(country_choice, var_choice, active_cell, data):
    if active_cell:
        cell_clicked = active_cell['row']
        var_clicked = data[cell_clicked]['variables']
    else:
        var_clicked = var_choice
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    all_dates = df[df['location'] == country_choice]['date']
    data_1 = df[df['location'] == country_choice][var_choice].to_list()
    fig.add_trace(
        go.Scatter(x=all_dates, y=data_1, name=var_choice),
        secondary_y=False,
    )
    data_2 = df[df['location'] == country_choice][var_clicked].to_list()
    fig.add_trace(
        go.Scatter(x=all_dates, y=data_2, name=var_clicked),
        secondary_y=True,
    )

    fig.update_xaxes(title_text='Dates')

    fig.update_yaxes(title_text=var_choice, secondary_y=False)
    fig.update_yaxes(title_text=var_clicked, secondary_y=True)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
