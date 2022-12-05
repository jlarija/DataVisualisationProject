from dash import Dash, html, dcc, Input, Output, State, dash_table
from sklearn import linear_model
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import random
import plotly.graph_objects as go
from utils_2 import *
import warnings
import pickle
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

app = Dash(__name__)

# Comment the next line and uncomment the 3 after to do your tests to avoid loading time, but the nans might fail your
# tests
# df, variables_each_country = get_preprocessed_df()
# url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
# df = pd.read_csv(url)
# variables_each_country = get_var_each_country()

with open('df.pickle', 'rb') as dffile:
    df, variables_each_country = pickle.load(dffile)

temp_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

fifa_df = get_fifa_data(df)


def fifa_plot(df):
    fig = px.scatter(df, x='total_cases_rank', y='fifa_rank',
                     hover_name='country_abrv',
                     hover_data=['fifa_rank', 'total_cases_rank']
                     )

    fig.update_traces(marker_color='#000000')

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
                opacity=0.95,
                layer="above"
            )
        )

    fig.update_layout(
        title_text="COVID cases on 09-04-2020 vs Fifa World Ranking for the same date",
        height=600, width=1000, plot_bgcolor="#FFFFFF")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Fifa Rank</b>", showgrid=True,
                     griddash='dash', gridcolor='#D4D4D4')
    fig.update_xaxes(title_text="<b>COVID Cases Rank</b>", showgrid=True,
                     griddash='dash', gridcolor='#D4D4D4')

    return fig


all_col = list(df.columns)
for col in columns_to_remove:
    all_col.remove(col)
for col in columns_fixed:
    if col in all_col:
        all_col.remove(col)

original_df = df
constraint_added = []
# none_all_col = columns_fixed.copy()
# none_all_col.insert(0, 'None')

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

filtering_dict = info_filtering(df)

app.layout = html.Div([
    dcc.Store(data=df.to_json(date_format='iso', orient='split'), id='df'),
    dcc.Store(data=months_df.to_json(date_format='iso', orient='split'), id='month-df'),
    html.H1('COVID 19: The Data',
            style={
                'textAlign': 'center',
                'color': 'black',
                'font_size': '36px'
            }),
    html.H1(
        'Data filtering',
        style={
            'textAlign': 'left',
            'color': 'black'
        }
    ),
    html.Div([
        html.Label("Activate filtering"),
        dcc.RadioItems(['Active', 'Reset'], 'Active', id='radio-filtering'),
        html.Br(),
        html.Div([
            dcc.Dropdown(columns_fixed, columns_fixed[0], id='variable-to-filter')
        ], style={'width': '39%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(['>', '>=', '=', '<', '<='], '>', id='sign-to-filter')
        ], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            dcc.Input(id='num-to-filter', type='number', value=0),
        ], style={'width': '39%', 'display': 'inline-block'}),
        html.Br(),
        html.Button(id='filtering-button', n_clicks=0, children='Filter'),
        html.Div(id='times-clicked')
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dash_table.DataTable(id='filter-table')
    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    html.Br(),
    html.Div([html.H1('A look at the world:'),
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
              ]),
    html.Br(),
    html.H1(
        'Evolution of multiple variables in time',
        style={
            'textAlign': 'left',
            'color': 'black'
        }
    ),
    html.Div([
        html.Label("Country or continent"),
        dcc.Dropdown([country for country in df['location'].unique()], df['location'][0],
                     id='country-continent-choice'),

        html.Br(),
        html.Label("Variables to plot (max 5)"),
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
    html.H3(
        'Correlations over time',
        style={
            'textAlign': 'left',
            'color': 'blue'
        }
    ),
    html.Div([
        html.Div([
            html.Label('Country or continent'),
            dcc.Dropdown([country for country in df['location'].unique()], df['location'][0], id='country-choice')
        ], style={'width': '48%', 'float': 'left', 'display': 'inline-block'}),
    ]),
    html.Div([
        dash_table.DataTable(id='corr-table-not-cumu')
    ]),
    html.Br(),
    html.H3(
        'Correlations cumulative with fixed variables',
        style={
            'textAlign': 'left',
            'color': 'blue'
        }
    ),
    html.Div([
        dash_table.DataTable(id='corr-table-cumu')
    ]),

    html.Br(),
    html.H1(
        'Variables dependencies for all countries',
        style={
            'textAlign': 'left',
            'color': 'black'
        }
    ),
    html.Div([
        html.Div([
            html.Label('x-axis'),
            dcc.Dropdown(all_col, all_col[0], id='x-axis-dependence')
        ], style={'width': '30%', 'float': 'left', 'display': 'inline-block'}),
        html.Div([
            html.Label('y-axis'),
            dcc.Dropdown(all_col, all_col[1], id='y-axis-dependence'),
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Size of the dots'),
            dcc.Dropdown(col_fixed_new_df, col_fixed_new_df[0], id='size-dot-dependence'),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'margin-bottom': '0.5cm'}),
    dcc.Graph(id='total-dependence-graph'),
    dcc.Slider(
        0,
        len(months_list) - 1,
        marks={i: str(slider_months[i]) for i in range(len(slider_months))},
        updatemode='mouseup',
        value=10,
        id='month-slider-dependence'
    ),

    html.Br(),
    html.H1(
        'Predictions for the next 3 months',
        style={
            'textAlign': 'left',
            'color': 'black'
        }
    ),
    html.Div([
        html.Label("Country or continent"),
        dcc.Dropdown([country for country in df['location'].unique()], df['location'][0],
                     id='country-predictions'),

        html.Br(),
        html.Label("Variables to predict"),
        dcc.Dropdown(variables_first_country, variables_first_country[0], id='var-to-pred')
    ]),
    dcc.Graph(id='predictions-graph'),
    html.Br(),
    html.H1('A Story of COVID Through Unconventional Data',
            style={
                'textAlign': 'left',
                'color': 'black'
            }
            ),
    html.H3('The beginning of COVID: did football fans contribute to spreading COVID?'),
    dcc.Graph(figure=fifa_plot(fifa_df))
])


#####################
# Filtering
@app.callback(
    Output('filter-table', 'data'),
    Output('filter-table', 'columns'),
    Input('variable-to-filter', 'value')
)
def update_info_filtering(variable):
    info_used = filtering_dict[variable]
    dict_info = {'variables': list(info_used.keys()),
                 'value': list(info_used.values())}

    info_df = pd.DataFrame(dict_info)
    info_df.set_index('variables')
    update_columns = [{"name": i, "id": i, "selectable": False} for i in info_df.columns]
    return info_df.to_dict('records'), update_columns


@app.callback(
    Output('times-clicked', 'children'),
    Output('filtering-button', 'n_clicks'),
    Output('df', 'data'),
    Output('month-df', 'data'),
    Input('radio-filtering', 'value'),
    Input('filtering-button', 'n_clicks'),
    State('variable-to-filter', 'value'),
    State('sign-to-filter', 'value'),
    State('num-to-filter', 'value'),
    State('df', 'data'),
    State('month-df', 'data'),
)
def filtering(radio_activate, number_conditions_added, var_filter, sign_filter, num_filter, df_stored, month_df_stored):
    new_df = pd.read_json(df_stored, orient='split')
    new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')
    new_month_df = pd.read_json(month_df_stored, orient='split')
    new_month_df['date'] = new_month_df['date'].dt.strftime('%Y-%m-%d')
    if radio_activate == 'Reset':
        # constraint_added.clear()
        string = u'0 conditions added'
        times_clicked = 0
        new_df = original_df
        new_month_df = get_month_df(original_df)

    elif var_filter == 'None':
        string = u'{} conditions added'.format(max([0, number_conditions_added - 1]))
        times_clicked = max([0, number_conditions_added - 1])
    else:
        # constraint_added.append([var_filter, sign_filter, num_filter])
        new_df = apply_constraints(new_df, [var_filter, sign_filter, num_filter])
        new_month_df = get_month_df(new_df)
        string = u'{} conditions added'.format(number_conditions_added)
        times_clicked = number_conditions_added
    return string, times_clicked, new_df.to_json(date_format='iso', orient='split'), new_month_df.to_json(
        date_format='iso', orient='split')


#######################
# Multi variables
@app.callback(
    Output('country-continent-choice', 'options'),
    Output('country-continent-choice', 'value'),
    Input('df', 'data'))
def change_available_countries_mult(data):
    used_df = pd.read_json(data, orient='split')

    all_countries = used_df['location'].unique()
    return all_countries, all_countries[0]


@app.callback(
    Output('y-axis', 'options'),
    Input('country-continent-choice', 'value'))
def y_axis_based_on_location(country_cont_choice):
    variables_to_show = variables_each_country[country_cont_choice]
    for col in columns_to_remove:
        if col in variables_to_show:
            variables_to_show.remove(col)
    return variables_to_show


@app.callback(
    Output('y-axis', 'value'),
    Input('y-axis', 'options'),
    Input('y-axis', 'value'))
def limit_number_choice(options_available, values_chosen):
    for val in values_chosen:
        if val not in options_available:
            return [options_available[0], options_available[1]]
    if len(values_chosen) <= 5:
        return values_chosen
    else:
        return values_chosen[:5]


@app.callback(
    Output('variables-graph', 'figure'),
    Input('y-axis', 'value'),
    Input('country-continent-choice', 'value'),
    Input('df', 'data'))
def update_graph_multi_var(variables_chosen, country_cont_choice, data):
    stored_df = pd.read_json(data, orient='split')
    stored_df['date'] = stored_df['date'].dt.strftime('%Y-%m-%d')
    used_df = stored_df[stored_df['location'] == country_cont_choice]

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

    hex_colors_plotly = ['#636efa', '#ef553b', '#00cc96', '#ac65fa', '#ffa25b']
    layout = {}
    layout['yaxis'] = {'tickfont': {'color': hex_colors_plotly[0]},
                       'title': {'font': {'color': hex_colors_plotly[0]}, 'text': variables_chosen[0]}}
    layout['xaxis'] = {'domain': [0.3, 0.9]}

    for i in range(1, len(variables_chosen)):
        pos = ((i * 1.4) * 0.25 / len(variables_chosen)) - 0.05
        layout['yaxis' + str(i + 1)] = {'anchor': 'free', 'position': pos, 'overlaying': 'y', 'side': 'left',
                                        'tickfont': {'color': hex_colors_plotly[i]},
                                        'title': {'font': {'color': hex_colors_plotly[i]}, 'text': variables_chosen[i]}}

    fig.update_layout(layout)
    fig.update_layout(title='Evolution of the chosen variables over time')

    return fig


#######################
# Correlations
@app.callback(
    Output('country-choice', 'options'),
    Output('country-choice', 'value'),
    Input('df', 'data'))
def change_available_countries_corr(data):
    used_df = pd.read_json(data, orient='split')

    all_countries = used_df['location'].unique()
    return all_countries, all_countries[0]


@app.callback(
    Output('corr-table-not-cumu', 'data'),
    Output('corr-table-not-cumu', 'columns'),
    Input('country-choice', 'value'),
    Input('df', 'data'))
def update_not_cumu_corr(country_choice, data):
    stored_df = pd.read_json(data, orient='split')
    stored_df['date'] = stored_df['date'].dt.strftime('%Y-%m-%d')
    not_cumu_vars = ['new_cases_per_million', 'new_deaths_per_million', 'excess_mortality', 'icu_patients_per_million',
                     'hosp_patients_per_million', 'stringency_index', 'reproduction_rate', 'new_tests_per_thousand',
                     'positive_rate', 'new_vaccinations']
    country_vars = variables_each_country[country_choice]
    sorted_vars = []
    for var in not_cumu_vars:
        if var in country_vars:
            sorted_vars.append(var)
    not_cumu_vars = sorted_vars
    df_not_cumu = stored_df[stored_df['location'] == country_choice][not_cumu_vars]

    corr_mat_not_cumu = df_not_cumu.corr(method='pearson')

    corr_dict = {'variables': corr_mat_not_cumu.index}
    for col in corr_mat_not_cumu.columns:
        corr_dict[col] = list(corr_mat_not_cumu[col])

    correlation_df = pd.DataFrame(corr_dict)
    correlation_df = correlation_df.round(2)
    correlation_df.set_index('variables')

    update_columns = [{"name": i, "id": i, "selectable": False} for i in correlation_df.columns]

    return correlation_df.to_dict('records'), update_columns


@app.callback(
    Output('corr-table-cumu', 'data'),
    Output('corr-table-cumu', 'columns'),
    Input('df', 'data'))
def update_cumu_corr(data):
    stored_df = pd.read_json(data, orient='split')
    stored_df['date'] = stored_df['date'].dt.strftime('%Y-%m-%d')

    cumulative_vars = ['total_cases_per_million', 'total_deaths_per_million', 'excess_mortality_cumulative_per_million',
                       'total_tests_per_thousand', 'total_vaccinations_per_hundred']
    total_cumu = cumulative_vars.copy()
    for col in columns_fixed:
        total_cumu.append(col)
    final_df_dict = {i: [] for i in total_cumu}
    final_df = pd.DataFrame.from_dict(final_df_dict)
    total_cumu.append('iso_code')
    df_cumu = stored_df[total_cumu]
    prev_iso = df_cumu['iso_code'].iloc[0]

    for i in range(len(df_cumu)):
        curr_iso = df_cumu['iso_code'].iloc[i]
        if curr_iso != prev_iso:
            final_df.loc[len(final_df)] = df_cumu.iloc[i].drop('iso_code')
            prev_iso = curr_iso
    final_df.loc[len(final_df)] = df_cumu.iloc[len(df_cumu) - 1].drop('iso_code')
    corr_mat_cumu = final_df.corr(method='pearson')
    corr_mat_cumu = corr_mat_cumu.drop(cumulative_vars, axis=0)
    corr_mat_cumu = corr_mat_cumu.drop(columns_fixed, axis=1)

    corr_dict = {'variables': corr_mat_cumu.index}
    for col in corr_mat_cumu.columns:
        corr_dict[col] = list(corr_mat_cumu[col])

    correlation_df = pd.DataFrame(corr_dict)
    correlation_df = correlation_df.round(2)
    correlation_df.set_index('variables')

    update_columns = [{"name": i, "id": i, "selectable": False} for i in correlation_df.columns]

    return correlation_df.to_dict('records'), update_columns


#######################
# Dependencies
@app.callback(
    Output('total-dependence-graph', 'figure'),
    Input('x-axis-dependence', 'value'),
    Input('y-axis-dependence', 'value'),
    Input('month-slider-dependence', 'value'),
    Input('size-dot-dependence', 'value'),
    Input('month-df', 'data'))
def update_dependence_graphs(x_axis_var, y_axis_var, month_slider, size_dot, month_data):
    my_df = pd.read_json(month_data, orient='split')
    my_df['date'] = my_df['date'].dt.strftime('%Y-%m-%d')

    cont_df = my_df.copy()
    current_month = months_list[month_slider]
    my_df = my_df.groupby(['iso_code', 'month'], sort=False).mean().reset_index()
    my_df = my_df[my_df['iso_code'].str.contains('OWID') == False]
    my_df = my_df[my_df['month'] == current_month]

    cont_df = cont_df[cont_df['iso_code'].str.contains('OWID') == False]
    cont_df = cont_df[cont_df['month'] == current_month].drop_duplicates('iso_code')

    my_df['continent'] = list(cont_df['continent'])
    my_df['location'] = list(cont_df['location'])
    trust_df_indexed = trust_df.set_index('location')
    trusts = list(trust_df_indexed.loc[list(cont_df['location'])].fillna(1)['trust_in_gov'])
    my_df['trust_in_gov'] = trusts
    new_df = pd.DataFrame.from_dict(
        {'country': list(my_df['location']), 'continent': list(my_df['continent']), x_axis_var: list(my_df[x_axis_var]),
         y_axis_var: list(my_df[y_axis_var]), size_dot: list(my_df[size_dot])})
    fig = px.scatter(new_df, x=x_axis_var, y=y_axis_var,
                     size=size_dot, color="continent", hover_name="country",
                     size_max=18)
    return fig


########################
# Predictions
@app.callback(
    Output('country-predictions', 'options'),
    Output('country-predictions', 'value'),
    Input('df', 'data'))
def change_available_countries(data):
    used_df = pd.read_json(data, orient='split')

    all_countries = used_df['location'].unique()
    return all_countries, all_countries[0]


@app.callback(
    Output('var-to-pred', 'options'),
    Output('var-to-pred', 'value'),
    Input('country-predictions', 'value'))
def var_for_country_pred(country_choice):
    variables_to_show = variables_each_country[country_choice]
    for col in columns_to_remove:
        if col in variables_to_show:
            variables_to_show.remove(col)
    for col in columns_fixed:
        if col in variables_to_show:
            variables_to_show.remove(col)
    return variables_to_show, variables_to_show[0]


@app.callback(
    Output('predictions-graph', 'figure'),
    Input('country-predictions', 'value'),
    Input('var-to-pred', 'value'),
    Input('df', 'data'))
def update_graph7(country_predict, data_to_predict, data):
    data_used_for_prediction = ['total_cases', 'new_cases', 'reproduction_rate', 'stringency_index', 'new_tests',
                                'positive_rate']
    stored_df = pd.read_json(data, orient='split')
    stored_df['date'] = stored_df['date'].dt.strftime('%Y-%m-%d')
    ##################
    # data management
    ##################
    all_features = variables_each_country[country_predict]
    updated_data_used_for_pred = []
    for var in data_used_for_prediction:
        if var in all_features:
            updated_data_used_for_pred.append(var)
    data_used_for_prediction = updated_data_used_for_pred
    all_features_to_predict = data_used_for_prediction.copy()
    if data_to_predict not in all_features_to_predict:
        all_features_to_predict.append(data_to_predict)

    for column in columns_to_remove:
        if column in all_features:
            all_features.remove(column)
        if column in data_used_for_prediction:
            data_used_for_prediction.remove(column)
        if column in all_features_to_predict:
            all_features_to_predict.remove(column)

    updated_col_fixed = []
    for col in columns_fixed:
        if col in all_features_to_predict:
            all_features_to_predict.remove(col)
            updated_col_fixed.append(col)

    updated_data_used_for_pred = []
    for var in data_used_for_prediction:
        if var in all_features:
            updated_data_used_for_pred.append(var)

    data_used_for_prediction = updated_data_used_for_pred

    columns_fixed_ordered = []
    for col in all_features:
        if col in updated_col_fixed:
            columns_fixed_ordered.append(col)
    updated_col_fixed = columns_fixed_ordered

    new_data_used = data_used_for_prediction.copy()
    for col in data_used_for_prediction:
        new_data_used.append(str(col) + "_1")
        new_data_used.append(str(col) + "_2")
        new_data_used.append(str(col) + "_3")
        new_data_used.append(str(col) + "_4")
        new_data_used.append(str(col) + "_5")
        new_data_used.append(str(col) + "_6")
    data_used_for_prediction = new_data_used
    train_datas = stored_df[stored_df['location'] == country_predict][all_features_to_predict].reset_index(drop=True)
    idx_data_to_pred = all_features_to_predict.index(data_to_predict)

    new_model = linear_model.Lasso(alpha=2, normalize=True, max_iter=10000000)
    train_datas_7 = generate_data(training_data=train_datas)
    new_model.fit(train_datas_7[data_used_for_prediction].iloc[:-1], train_datas[all_features_to_predict].iloc[7:])
    weights = get_weights(data_used_for_prediction, new_model.coef_[:][idx_data_to_pred])
    non_zero_weights = []
    for key in weights.keys():
        if abs(weights[key]) > 0:
            non_zero_weights.append(key)

    all_predictions = []
    all_dates = []
    index_fixed = []
    for i in range(len(data_used_for_prediction)):
        if data_used_for_prediction[i] in updated_col_fixed:
            index_fixed.append(i)

    for i in range(90):  # 90 days for 3 months
        if i == 0:
            last_date = str(df['date'].iloc[-1])
        else:
            last_date = str(all_dates[-1])
        last_datetime = datetime.strptime(last_date, '%Y-%m-%d')
        new_datetime = last_datetime + timedelta(days=1)
        new_date = str(new_datetime)[:10]
        all_dates.append(new_date)

        predicted_data = new_model.predict(train_datas_7[data_used_for_prediction].iloc[-1].to_numpy().reshape(1, -1))
        all_predictions.append(predicted_data)
        if len(index_fixed) > 0:
            for idx in index_fixed:
                predicted_data[0].insert(idx, train_datas_7[data_used_for_prediction[i]].iloc[-1])

        new_row = []
        for feature in all_features_to_predict:
            feature_1 = str(feature) + "_1"
            new_row.append(train_datas_7[feature_1].iloc[-1])
        j = 0
        for feature in all_features_to_predict:
            feature_2 = str(feature) + "_2"
            new_row.append(train_datas_7[feature_2].iloc[-1])
            feature_3 = str(feature) + "_3"
            new_row.append(train_datas_7[feature_3].iloc[-1])
            feature_4 = str(feature) + "_4"
            new_row.append(train_datas_7[feature_4].iloc[-1])
            feature_5 = str(feature) + "_5"
            new_row.append(train_datas_7[feature_5].iloc[-1])
            feature_6 = str(feature) + "_6"
            new_row.append(train_datas_7[feature_6].iloc[-1])
            new_row.append(predicted_data[0][j])
            j = j + 1

        train_datas_7.loc[len(train_datas_7)] = new_row

    x = df[df['location'] == country_predict]['date'].tolist()
    y = train_datas[data_to_predict].tolist()
    for i in range(len(all_dates)):
        x.append(all_dates[i])
        y.append(all_predictions[i][0][idx_data_to_pred])

    ##################
    # plot the predictions
    ##################
    prediction_df = pd.DataFrame({"date": x, "value": y})
    fig = px.line(prediction_df, x="date", y="value")

    fig.update_yaxes(title=str(data_to_predict + " predicted for next 3 months"))
    fig.add_vline(x=x[-91], line_width=1, line_color="red")

    return fig


@app.callback(
    Output('Choropleth Map', 'figure'),
    Input('chorplethdropdown', 'value'),
    Input('monthchoroplethmap', 'value'),  # gives a numerical value
    Input('month-df', 'data'))
def choropleth_map(choroplethdropdown, monthchoroplethmap, month_df_loaded):
    my_df = pd.read_json(month_df_loaded, orient='split')
    my_df['date'] = my_df['date'].dt.strftime('%Y-%m-%d')

    my_df = my_df.groupby(['iso_code', 'month'], sort=False).mean().reset_index()
    my_df = my_df[my_df['iso_code'].str.contains('OWID') == False]

    colorscale = ['#ffd7cd', '#e3ada0', '#c68475', '#a95c4c', '#893427', '#690000']

    current_month = months_list[monthchoroplethmap]
    my_df = my_df[my_df['month'] == current_month]
    min_color = np.max(my_df[str(choroplethdropdown)])
    max_color = np.min(my_df[str(choroplethdropdown)])

    fig = px.choropleth(my_df, locations='iso_code', color=str(choroplethdropdown),
                        color_continuous_scale=colorscale, hover_name="iso_code", range_color=(min_color, max_color))

    background_color = '#F5F2E8'

    fig.update_layout(font_family='Balto', font_color='#000000',
                      font_size=18, plot_bgcolor=background_color,
                      geo=dict(
                          showframe=False,
                          showcoastlines=False,
                          countrycolor='#000000',
                          bgcolor=background_color,
                          lakecolor=background_color,
                          landcolor='rgba(51,17,0,0.2)',
                          subunitcolor='grey'

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


if __name__ == "__main__":
    app.run_server(debug=True)
