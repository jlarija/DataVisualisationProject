from utils import *
import json
import pickle
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings("ignore")

with open('world_list.pickle', 'rb') as file1:
    world_names = pickle.load(file1)

df, variables_each_country = get_preprocessed_df()

def smooth_df(df):
    covid_df = df.loc[df['date'] == '2022-11-02', ('location','total_cases')]
    covid_df = covid_df.dropna().reset_index(drop=True)
    with open('covid_final.pickle', 'wb') as file:
        pickle.dump(covid_df,file)

    return covid_df

smoot = smooth_df(df)

def remove_countries_notinworldmap(df):

    lines_dropped = ['Tokelau', 'Sao Tome and Principe','Monaco','World', 'Macao', 'Pitcairn',\
         'Timor', 'Laos', 'Saint Pierre and Miquelon','Liechtenstein', 'Monaco',\
             'Northern Mariana Islands',  'Micronesia (country)', 'Cook Islands', \
            'Anguilla', 'Saint Kitts and Nevis', 'Kiribati', 'Marshall Islands', \
            'Cayman Islands', 'Turks and Caicos Islands', 'Maldives', 'Guam','United States Virgin Islands'\
            'Brunei', 'Seychelles', 'Palau', 'Eswatini', 'Guernsey', 'Wallis and Futuna',\
            'Tonga', 'Bermuda', 'Montserrat',  'Niue', 'Saint Vincent and the Grenadines',\
             'Aruba','Africa', 'Grenada','Jersey', 'Malta', 'Tuvalu', 'Nauru','Samoa', 'Bonaire Sint Eustatius and Saba', \
                'Lower middle income', 'International', 'Upper middle income', 'European union', 'High income' ,\
            'North america', 'Oceania','Gibraltar','South america' ,'England',  'Micronesia', 'Curacau', 'Sint Maarten', \
            'Scotland', 'Northern Ireland', 'Wales','British Virgin Islands', 'Brunei', 'Saint Helena', 'Low income', 'North America',\
        'Curacao', 'European Union', 'Asia', 'South America', 'Europe']

    for country in lines_dropped:
        df = df[df['location'].str.contains(country) == False]

    df = df.replace(to_replace=['Democratic Republic of Congo','Russia', 'Czechia', 'Vatican', 'North Korea', 'South Korea',"Cote d'Ivoire", 'Congo','Gambia','North Macedonia'],\
        value = ['Democratic Republic of the Congo', 'Russian Federation', 'Czech Republic', 'Vatican City', 'Dem. Rep. Korea', 'Republic of Korea',"Côte d'Ivoire", 'Republic of Congo', \
            'The Gambia', 'Macedonia'] )
            
    return df

covid = remove_countries_notinworldmap(smoot)

with open('world.geojson') as file:
     world_map = json.load(file)

covid_countries = covid['location'].unique()

diff_list2 = list(set(world_names) - set(covid_countries))

countries_to_delete = diff_list2

world_map2 = world_map

for i in range(len(world_map2['features'])):
        #print(i)
        current_country = world_map2['features'][i]['properties']['NAME_LONG'] 

        if current_country in countries_to_delete:
               world_map2['features'][i] = []


# now take a look at the resulting
world_map2['features'] = [lst for lst in world_map2['features'] if lst != []]

names = []
for i in range(len(world_map2['features'])):
        names.append(world_map2['features'][i]['properties']['NAME_LONG'])


app = Dash(__name__)



