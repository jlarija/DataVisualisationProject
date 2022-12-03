import pickle
from utils import *
import plotly.express as px
import pandas as pd
import pycountry
import numpy as np

with open('df.pickle', 'rb') as dffile:
        df,variables_each_country = pickle.load(dffile)

fifa_ranking = pd.read_csv('fifa_ranking-2022-10-06.csv')
fifa_ranking = fifa_ranking.drop(columns = ['total_points', 'previous_points', 'rank_change', 'confederation', 'country_full']).reset_index()

# get the 2020 ranking
fifa_ranking = fifa_ranking[fifa_ranking['rank_date'].str.contains('2020') == True]
fifa_ranking = fifa_ranking[fifa_ranking['rank_date'] == '2020-04-09']
fifa_ranking = fifa_ranking.sort_values(by='rank').reset_index()
fifa_ranking = fifa_ranking.drop(columns = ['level_0', 'index'])

# sort the original dataframe with covid data
df = df[df['iso_code'].str.contains('OWID')==False]
df = df[df['date'] == '2020-04-09']
df.rename(columns ={'iso_code':'country_abrv'}, inplace=True)
df = df[['country_abrv', 'total_cases']]

# match the dataframes
final_rank = df.merge(fifa_ranking, how='left', on='country_abrv')
final_rank = final_rank.dropna()
final_rank = final_rank.rename(columns = {'rank':'fifa_rank'})
# create a ranking of the covid cases
final_rank['total_cases_rank'] = final_rank['total_cases'].rank(method = 'max', ascending = False)

final_df = final_rank.sort_values(by='fifa_rank')
iso_2 = []

for element in final_df['country_abrv']:
    country_data = pycountry.countries.get(alpha_3 = str(element))
    iso_2.append(country_data.alpha_2)

alpha2 = pd.Series(iso_2)
# final_df['iso_2'] = alpha2
final_df = final_df.reset_index()
final_df['iso_2'] = alpha2
final_df = final_df.drop(columns = ['index'])


# PLOT

# Create figure with secondary y-axis
fig = px.scatter(
    final_df,
    x = 'total_cases_rank', 
    y = 'fifa_rank',
    hover_name = 'country_abrv',
    hover_data =['fifa_rank', 'total_cases_rank']
)

fig.update_traces(marker_color = '#000000')

min_dim = final_df[['fifa_rank', 'total_cases_rank']].max().idxmax()
maxi = final_df[min_dim].max()
for i, row in final_df.iterrows():
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
            sizey=np.sqrt(row["total_cases"] / df["total_cases"].max()) * maxi * 0.025+ maxi * 0.03,
            sizing="contain",
            opacity=0.95,
            layer="above"
        )
    ) 


fig.update_layout(
    title_text="COVID cases on 09-04-2020 vs Fifa World Ranking for the same date",
    height=600, width=1000, plot_bgcolor="#FFFFFF")


# Set y-axes titles
fig.update_yaxes(title_text="<b>Fifa Rank</b>", showgrid = True, 
griddash = 'dash', gridcolor = '#D4D4D4')
fig.update_xaxes(title_text="<b>COVID Cases Rank</b>", showgrid = True, 
griddash = 'dash', gridcolor = '#D4D4D4')

fig.show()

