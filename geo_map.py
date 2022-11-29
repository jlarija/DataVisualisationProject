'''
The final, standalone version of the geomap
'''
from dash import Dash, html, dcc, Input, Output
from utils import *
import json
import plotly.express as px
import pickle

with open('world_list.pickle', 'rb') as file1:
    world_names = pickle.load(file1)

def equalize_countries(df):

    '''This function makes the countries in the geoJSON and csv file the same. This means:
    
    1. Remove from the csv countries that are not in the JSON
    2. Remove from the JSON countries that are not in the CSV
    3. Rename the countries that are in both files, but the naming is slightly different (eg. Vatican -> Vatican City)
    
    Point 3 is the reason that the lines_dropped is in a list and not done iteratively; because otherwise
    if the naming was even slightly different the countries would be dropped automatically so I still 
    had to manually check (if you know a better way I am all up for it!)'''

    # First part: make the csv the same as the geojson
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

    # In the .CSV, rename those countries with a slightly different name than the geoJSON
    df = df.replace(to_replace=['Democratic Republic of Congo','Russia', 'Czechia', 'Vatican', 'North Korea', 'South Korea',"Cote d'Ivoire", 'Congo','Gambia','North Macedonia'],\
        value = ['Democratic Republic of the Congo', 'Russian Federation', 'Czech Republic', 'Vatican City', 'Dem. Rep. Korea', 'Republic of Korea',"Côte d'Ivoire", 'Republic of Congo', \
            'The Gambia', 'Macedonia'])

    # Now make the geoJSON the same as tehe csv
    with open('world.geojson', 'rb') as jsonfile:
        json_countries = json.load(jsonfile)

    df_countries = list(df['location'].unique())

    diff_list = list(set(world_names) - set(df_countries)) # here are the remaining countries to delete
    
    for i in range(len(json_countries['features'])):
        current_country = json_countries['features'][i]['properties']['NAME_LONG'] 

        if current_country in diff_list:
               json_countries['features'][i] = []

    json_countries['features'] = [lst for lst in json_countries['features'] if lst != []]


    # Some final checks to make sure the function did its job right
    names = []
    for i in range(len(json_countries['features'])):
        names.append(json_countries['features'][i]['properties']['NAME_LONG'])

    final_diff = list(set(names) - (set(df_countries)))

    return ('Here is the difference (should be empty)', final_diff), df, json_countries

# Load up variables from utils.py
df, variables_each_country = get_preprocessed_df()

# Now make the dataframe for plotting
check, covid_df, wrldmap = equalize_countries(df)
col_to_remove = ['iso_code', 'continent','tests_units'] # non numerical data, I think
covid_df = covid_df.drop(columns=col_to_remove)
covid_df = get_month_df(covid_df) # Split months cause slider
covid_df = covid_df.groupby(['month','location'], sort=False).mean().reset_index()

# Dash
app = Dash(__name__)

# I need to choose the port everytime for some reason, cause it keeps saying the previous one is still in use
def run_server(self,
               port=8050,
               debug=True,
               threaded=True,
               **flask_run_options):
    self.server.run(port=port, debug=debug, **flask_run_options)

app.layout = html.Div([
    html.H4('Geographical Analysis of Covid parameters'),
    dcc.Dropdown([column for column in covid_df.columns],
    covid_df['total_cases'], 
    id = 'variables-map'       
    ),
    dcc.Graph(id="map"),
])

@app.callback(
    Output("map", "figure"), 
    Input("variables-map", "value"))

def display_chloropleth(var):
    df = covid_df
    geojson = wrldmap
    figure = px.choropleth(df, geojson=geojson, color = var, locations ='location',\
     featureidkey='properties.NAME_LONG', animation_frame = 'month', height=600,width=1000,color_continuous_scale='Brwnyl')
    return figure

if __name__ == '__main__':
    app.run_server(debug=True, port = 8020)