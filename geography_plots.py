from utils import *
import json
import pickle

# with open('world.geojson') as file:
#     world_map = json.load(file)

# def world_names(world_df):
#     names = list()
#     for i in range(len(world_df['features'])):
#         names.append(world_df['features'][i]['properties']['NAME_LONG'])

#     with open('world_list.pickle', 'wb') as file1:
#         pickle.dump(names,file1)

#     return names

# world_names1 = world_names(world_map)

#df, variables_each_country = get_preprocessed_df()

def smooth_df(df):
    covid_df = df.loc[df['date'] == '2022-11-13', ('location','total_cases')]

    with open('covid_list.pickle', 'wb') as file:
        pickle.dump(covid_df,file)

    return covid_df

with open('world_list.pickle', 'rb') as file1:
    world_names = pickle.load(file1)

with open('covid_list.pickle', 'rb') as file:
    covid_df = pickle.load(file)

def remove_countries_notinworldmap(df):

    lines_dropped = ['Tokelau', 'Sao Tome and Principe','Monaco','World', 'Macau', 'Pitcairn',\
         'Timor', 'Laos', 'Saint Pierre and Miquelon','Liechtenstein', 'Monaco',\
             'Northern Mariana Islands',  'Micronesia (country)', 'Cook Islands', \
            'Anguilla', 'Saint Kitts and Nevis', 'Kiribati', 'Marshall Islands', \
            'Cayman Islands', 'Turks and Caicos Islands', 'Maldives', 'Guam','United States Virgin Islands'\
            'Brunei', 'Seychelles', 'Palau', 'Eswatini', 'Guernsey', 'Wallis and Futuna',\
            'Tonga', 'Bermuda', 'Montserrat',  'Niue', 'Saint Vincent and the Grenadines',\
             'Aruba', 'Grenada', 'Malta', 'Tuvalu', 'Nauru','Samoa', 'Bonaire Sint Eustatius and Saba', \
                'Lower middle income', 'International', 'Upper middle income', 'European union', 'High income' ,\
            'North america', 'Oceania','Gibraltar','South america' , 'Micronesia', 'Curacau', 'Sint Maarten']

    for country in lines_dropped:
        df.drop(df[df['location'] == str(country)].index, inplace =True)

    df['location'].replace({'Democratic Republic of Congo': 'Democratic Republic of the Congo','Russia': 'Russian Federation', 'Czechia' : 'Czech Republic', 'Vatican': 'Vatican City','North Korea': 'Dem. Rep. Korea', 'South Korea': 'Republic of Korea', 'Cote d’Ivoire':'Côte d’Ivoire','Congo': ' Republic of Congo', 'Gambia':'The Gambia', 'North Macedonia': 'Macedonia', \
        'England': 'United Kingdom', 'Scotland': 'United Kingdom', 'Wales': 'United Kingdom', 'Northern Ireland': 'United Kingdom', 'Wales': 'United Kingdom'})
    # UK is annoyingly split
    df.groupby(['location']).sum()
    return df['location'].unique()

final_df = remove_countries_notinworldmap(covid_df)
print(final_df)
# diff_list = list(set(covid_names) - set(world_names))
# print(diff_list)
# print()
# print()
# print(covid_names)
# print()
# print()
# print(world_names)
