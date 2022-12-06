# Layout imports
from app import *
# Dash imports
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, dash_table
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
# Data Analysis happens in utils
from utils import *
# Functionalities
from sklearn import linear_model
from datetime import datetime, timedelta
import random
import warnings
import pickle
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": navig_bar,
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


def nav_bar():
    navbar = html.Div(
        [
            html.H2("Explore", className="display-4", style={'color': 'black'}),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink("COVID Data", href="/", active="exact"),
                    dbc.NavLink("Additional COVID trends", href="/page-1", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )
    return navbar


###############
## BEGINNING DATA ANALYSIS (NECESSARY FOR THE PLOTS)

with open('df_alio.pickle', 'rb') as dffile:
    df, variables_each_country = pickle.load(dffile)

img = Image.open('airplane-clipart-transparent-7.png')
fifa_df = get_fifa_data(df)


def fifa_plot(df):
    fig = px.scatter(df, x='total_cases_rank', y='fifa_rank',
                     hover_name='country_abrv',
                     hover_data=['fifa_rank', 'total_cases_rank']
                     )

    fig.update_traces(marker_color='#FFFFFF')

    min_dim = df[['fifa_rank', 'total_cases_rank']].max().idxmax()
    maxi = df[min_dim].max()
    for i, row in df.iterrows():
        country_iso = row['iso_2']
        fig.add_layout_image(
            dict(
                source=f"https://raw.githubusercontent.com/matahombres/CSS-Country-Flags-Rounded/master/flags/{country_iso}.png",
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=row["total_cases_rank"],
                y=row["fifa_rank"],
                sizex=np.sqrt(row["total_cases"] / df["total_cases"].max()) * maxi * 0.025 + maxi * 0.03,
                sizey=np.sqrt(row["total_cases"] / df["total_cases"].max()) * maxi * 0.025 + maxi * 0.03,
                sizing="contain",
                opacity=1,
                layer="above"
            )
        )

    fig.update_layout(
        height=600, width=1000, plot_bgcolor=background_color, paper_bgcolor=background_color,
        font=dict(size=font_size, color=legend_fonts))

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Fifa Rank</b>", showgrid=True,
                     griddash='dash', gridcolor='#727272', color=legend_fonts)
    fig.update_xaxes(title_text="<b>COVID Cases Rank</b>", showgrid=True,
                     griddash='dash', gridcolor='#727272', color=legend_fonts)

    return fig


def plane_data_plot(df):
    airtraffic = pd.read_csv('avia_tf_cm__custom_1858764_page_linear.csv')
    airtraffic = airtraffic[['unit', 'TIME_PERIOD', 'OBS_VALUE']]
    covid = df[df['location'] == 'Europe']
    covid = covid[['date', 'new_cases', 'new_deaths', 'total_vaccinations']]
    covid_monthly = get_month_df(covid)
    cc = covid_monthly.groupby(['month'], sort=False).mean().reset_index()
    cc = cc.truncate(after=23)
    airtraffic = airtraffic.truncate(after=23)
    airtraffic['new_cases'] = cc['new_cases']
    airtraffic['new_deaths'] = cc['new_deaths']
    airtraffic['total_vaccinations'] = cc['total_vaccinations']

    color_bins = '#006796'
    color_line1 = '#f75e91'
    color_line2 = '#ffa600'

    fig = go.Figure()

    fig.add_trace(go.Bar(x=airtraffic['TIME_PERIOD'], y=airtraffic['OBS_VALUE'], name='Passengers traveling by plane',
                         marker_color=color_bins))
    fig.add_trace(
        go.Scatter(x=airtraffic['TIME_PERIOD'], y=airtraffic['new_cases'], name='monthly new COVID cases', yaxis='y2',
                   marker_color=color_line1, opacity=0.8))
    fig.add_trace(
        go.Scatter(x=airtraffic['TIME_PERIOD'], y=airtraffic['new_deaths'], name='monthly new COVID deaths', yaxis='y3',
                   marker_color=color_line2, opacity=0.8))

    fig.update_layout(xaxis=dict(domain=[0.2, 0.9]), font_size=font_size,
                      yaxis=dict(title="Monthly Air Passengers", titlefont=dict(color=color_bins, ), tickfont=dict(
                          color=color_bins
                      )),
                      yaxis2=dict(
                          title="Monthly new COVID cases",
                          titlefont=dict(
                              color=color_line1
                          ),
                          tickfont=dict(
                              color=color_line1
                          ),
                          anchor="free",
                          overlaying="y",
                          side="right",
                          position=1
                      ),
                      yaxis3=dict(
                          title="Monthly new COVID deaths",
                          titlefont=dict(
                              color=color_line2
                          ),
                          tickfont=dict(
                              color=color_line2
                          ),
                          anchor="free",
                          overlaying="y",
                          side="right",
                          position=0.9
                      ))

    fig.update_layout(plot_bgcolor=background_color, paper_bgcolor=background_color, width=1000,
                      margin=dict(l=0, r=10, t=0, b=0),
                      legend=dict(
                          yanchor="top",
                          y=1.3,
                          xanchor="left",
                          x=0.5,
                          font_color=legend_fonts
                      )
                      )
    fig.update_xaxes(color=legend_fonts)
    # fig.add_layout_image(
    #     dict(
    #         source=img,
    #         xref="paper", yref="paper",
    #         x=0.5, y=0.5,
    #         sizex=0.45, sizey=0.45,
    #         xanchor="right", yanchor="bottom"
    #     )
    # )

    return fig


all_col = list(df.columns)
for col in columns_to_remove:
    all_col.remove(col)

original_df = df
constraint_added = []
none_all_col = columns_fixed.copy()
none_all_col.insert(0, 'None')

variables_first_country = variables_each_country[df['location'][0]]

months_list = get_list_months(df)
slider_months = [month[:3] + month[5:] for month in months_list]
months_df = get_month_df(df)

trust_df = pd.read_csv('share-who-trust-government.csv')
trust_df = trust_df.drop(['Code', 'Year'], axis=1)
trust_df.columns = ['location', 'trust_in_gov']
for countr in df['location'].unique():
    if countr not in list(trust_df['location']):
        trust_df.loc[len(trust_df)] = [countr, float("nan")]

col_fixed_new_df = columns_fixed.copy()
col_fixed_new_df.insert(0, 'trust_in_gov')

col_geomap = all_col.copy()
for column in col_geomap:
    col_geomap.remove(column)
filtering_dict = info_filtering(df)

layout1 = html.Div([

    dcc.Store(data=df.to_json(date_format='iso', orient='split'), id='df'),
    dcc.Store(data=months_df.to_json(date_format='iso', orient='split'), id='month-df'),

    dbc.Row(

        dbc.Col(
            html.Div(html.H1('COVID 19 Data Exploration', style={'color': 'white', 'font_size': '36px'})),
            width={"size": 6, "offset": 4},
        )
    ),

    dbc.Row(

        dbc.Col([
            html.Div([
                dcc.RadioItems(['Active', 'Reset'], 'Active', id='radio-filtering'),
                html.Div([
                    dcc.Dropdown(none_all_col, 'gdp_per_capita', id='variable-to-filter')
                ], style={'width': '35%', 'display': 'inline-block','color': 'black'}),
                html.Div([
                    dcc.Dropdown(['>', '>=', '=', '<', '<='], '>', id='sign-to-filter')
                ], style={'width': '10%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Input(id='num-to-filter', type='number', value=0),
                ], style={'width': '39%', 'display': 'inline-block'}),
                html.Div([html.Button(id='filtering-button', n_clicks=0, children='Filter')
                          ], style={'display': 'inline-block'}),
                html.Div(id='times-clicked')
            ], style={'width': '48%', 'display': 'inline-block'}),
             html.Div([
        dash_table.DataTable(id='filter-table',style_header={
                                               'backgroundColor': 'rgb(30, 30, 30)',
                                               'color': 'white'
                                           },
                                           style_data={
                                               'backgroundColor': 'rgb(50, 50, 50)',
                                               'color': 'white'
                                           }, )
    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        ], width={"offset": 4})
    ),

    html.Br(),
    html.Br(),

    dbc.Row(

        dbc.Col(html.H4('World Situation'), width={"offset": 5})

    ),

    dbc.Row(

        dbc.Col(
            html.Div([
                dcc.Dropdown(col_geomap, 'total_cases', id='chorplethdropdown', clearable=False)],
                style={'width': '26%'}),
            width={'offset': 5}

        )
    ),

    dbc.Row([

        dbc.Col([
            dcc.Graph(id='Choropleth Map'),
            dcc.Slider(0, len(months_list) - 1, marks={i: str(slider_months[i]) for i in range(len(slider_months))},
                       updatemode='mouseup', value=0, id='monthchoroplethmap')
        ])
    ]),

    html.Br(),
    html.Br(),

    dbc.Row(

        dbc.Col([
            html.H6('Evolution of multiple variables in time'),
            html.Div([
                html.Label('Country or continent'),
                dcc.Dropdown([country for country in df['location'].unique()], df['location'][0],
                             id='country-continent-choice')],
                style={"size": 3, 'width': '30%', 'display': 'inline-block', 'margin-top': '70px'}),
            html.Div([
                html.Label('Variables to show'),
                dcc.Dropdown(variables_first_country, variables_first_country[0], id='y-axis', multi=True)],
                     style={'width': '60%', 'display': 'inline-block', 'margin-top': '70px'}),
                    html.Br(),
                    dcc.Graph(id='variables-graph')
                ]),
            
            ),
    
    dbc.Row(

        dbc.Col([
            html.H6('Future predictions'),
            html.Label("Country or continent"),
            dcc.Dropdown([country for country in df['location'].unique()], df['location'][0],
                         id='country-predictions'),
            html.Label("Variable to predict"),
            dcc.Dropdown(variables_first_country, variables_first_country[0], id='var-to-pred'),
            html.Br(),
            dcc.Graph(id='predictions-graph')
                ])
            
            ),

    html.Br(),

    dbc.Row(

        dbc.Col(
            html.Div(
                html.H3('Exploring Correlations', style={'color': 'white', 'font_size': '30px', 'textAlign': 'center'}))
        )
    ),

    html.Br(),
    html.Br(),
    dbc.Row(
        dbc.Col([
            html.Label('Country or continent choice'),
            dcc.Dropdown([country for country in df['location'].unique()], df['location'][0],
                         id='country-choice', style={'width': '40%', 'float': 'left', 'display': 'inline-block'}),
            html.Br(),
            html.Div([dash_table.DataTable(id='corr-table-not-cumu',
                                           style_header={
                                               'backgroundColor': 'rgb(30, 30, 30)',
                                               'color': 'white'
                                           },
                                           style_data={
                                               'backgroundColor': 'rgb(50, 50, 50)',
                                               'color': 'white'
                                           }, )])

        ])
    ),
    html.Br(),

    dbc.Row(

        dbc.Col([
            html.H3('Correlations cumulative with fixed variables', style={'textAlign': 'left'}),
            html.Br(),
            html.Div([dash_table.DataTable(id='corr-table-cumu',
                                           style_header={
                                               'backgroundColor': 'rgb(30, 30, 30)',
                                               'color': 'white'
                                           },
                                           style_data={
                                               'backgroundColor': 'rgb(50, 50, 50)',
                                               'color': 'white'
                                           }, )])

        ])
    ),
    html.Br(),

    dbc.Row(

        dbc.Col([
            html.H4('Variable dependence on Citizens\'s thrust in the government (and additional fixed variables)',
                    style={'textAlign': 'center', 'color': 'white'}),
        ])
    ),

    html.Br(),
    html.Br(),

    dbc.Row([

        dbc.Col([
            html.Div([html.Label('Select x-axis variable'),
                      dcc.Dropdown(all_col, all_col[0], id='x-axis-dependence')
                      ], style={'width': '30%', 'float': 'left', 'display': 'inline-block'})
        ]),

        dbc.Col([
            html.Div([
                html.Label('y-axis'),
                dcc.Dropdown(all_col, all_col[1], id='y-axis-dependence'),
            ], style={'width': '30%', 'display': 'inline-block'})
        ]),

        dbc.Col([
            html.Div([
                html.Label('Size of the dots'),
                dcc.Dropdown(col_fixed_new_df, col_fixed_new_df[0], id='size-dot-dependence'),
            ], style={'width': '30%', 'display': 'inline-block'})
        ])

    ]),

    dbc.Row(

        dbc.Col([
            dcc.Graph(id='total-dependence-graph'),
            dcc.Slider(0, len(months_list) - 1, marks={i: str(slider_months[i]) for i in range(len(slider_months))},
                       updatemode='mouseup', value=10, id='month-slider-dependence')
        ])
    ),
])

layout2 = html.Div([

    dbc.Row(

        dbc.Col([
            html.Div([html.H1('Additional Unconventional Data to showcase COVID trends',
                              style={'textAlign': 'center', 'color': 'white'})]),

            html.Br(),

            html.Div([html.H3('Correlation is not causation: football rankings and the spread of COVID',
                              style={'textAlign': 'center', 'color': 'white'})]),

            html.Br(),

            html.Div([html.H6('Where COVID rank refers to the rank in the total cases for the chosen day. A higher COVID rank = more cases \
                            and similarly, a higher FIFA rank means a higher positon in the world footbals\'s association.',
                              style={'textAlign': 'center', 'color': 'white', 'font_size': '16px'})]),

        ])

    ),

    html.Br(),

    dbc.Row(
        dbc.Col([dcc.Graph(figure=fifa_plot(fifa_df))], width={'offset': 2})
    ),

    html.Br(),
    html.Br(),

    dbc.Row(

        dbc.Col([

            html.Div([html.H3('Fluctuations in freedom: travel data for the EU show \
                        the same trend as new COVID deaths and cases', style={'textAlign': 'center', 'color': 'white'})
                      ]),

            dcc.Graph(figure=plane_data_plot(df))
        ])
    )
])
