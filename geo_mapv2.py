from dash import Dash, html, dcc, Input, Output
from utils import *
import json
import pickle
import plotly.graph_objects as go
import pandas as pd
from plot_ideas_geo import *

# DATA 
data_not_saved = False
if data_not_saved:
    df, variables_each_country = get_preprocessed_df()  
    with open('df.pickle', 'wb') as file:
        pickle.dump([df,variables_each_country],file)

with open('df.pickle', 'rb') as dffile:
        df,variables_each_country = pickle.load(dffile)


# PLOTS BACK-END
def choropleth_map():
    df = get_month_df(df) # Split months cause slider
    df = df.groupby(['iso_code','month'], sort=False).mean().reset_index()

    # PLOTLY PART
    colorscale = ['#ffcdbf','#dea394','#bc7b6c','#9b5447','#792e25', '#570000']

    fig = go.Figure(data = go.Choropleth(
        locations = df['iso_code'],
    z = df['total_cases_per_million'], 
    text = df['iso_code'], 
    colorscale=colorscale,
    colorbar_title = 'Total Cases Per Million'
    ))
    background_color = '#F5F2E8'
    fig.update_layout(title_text='COVID',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            countrycolor='#000000',
            bgcolor= background_color,
            lakecolor= background_color, 
            landcolor='rgba(51,17,0,0.2)',
            subunitcolor='grey'
        ))

    # Delete antartica
    fig.add_trace(go.Choropleth(locations=['ATA'],
                z=[0],
                colorscale=[[0,background_color], [1, background_color]],
                marker_line_color=background_color,
                showlegend=False,
                showscale=False)
        )
    
    return fig


# if __name__ == '__main__':
#     app.run_server(debug = True)