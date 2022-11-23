from dash import Dash, html, dcc, Input, Output, State, dash_table
from sklearn import linear_model
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import random
import plotly.graph_objects as go
from utils import *
import warnings

warnings.filterwarnings('ignore')

app = Dash(__name__)

# Comment the next line and uncomment the 3 after to do your tests to avoid loading time, but the nans might fail your
# tests
df, variables_each_country = get_preprocessed_df()
# url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
# df = pd.read_csv(url)
# variables_each_country = get_var_each_country()

temp_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

all_col = list(df.columns)
for col in columns_to_remove:
    all_col.remove(col)

original_df = df
constraint_added = []
none_all_col = columns_fixed
none_all_col.insert(0, 'None')

variables_first_country = variables_each_country[df['location'][0]]

months_list = get_list_months(df)
months_df = get_month_df(df)

trust_df = pd.read_csv('share-who-trust-government.csv')
trust_df = trust_df.drop(['Code', 'Year'], axis=1)
trust_df.columns = ['location', 'trust_in_gov']
for countr in df['location'].unique():
    if countr not in trust_df['location']:
        trust_df.loc[len(trust_df)] = [countr, float("nan")]

col_fixed_new_df = columns_fixed.copy()
col_fixed_new_df.insert(0, 'trust_in_gov')

app.layout = html.Div([
    dcc.Store(data=df.to_json(date_format='iso', orient='split'), id='df'),
    dcc.Store(data=months_df.to_json(date_format='iso', orient='split'), id='month-df'),
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
            dcc.Dropdown(none_all_col, none_all_col[0], id='variable-to-filter')
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
        html.Div([
            html.Label('Country or continent'),
            dcc.Dropdown([country for country in df['location'].unique()], df['location'][0], id='country-choice')
        ], style={'width': '48%', 'float': 'left', 'display': 'inline-block'}),
        html.Div([
            html.Label('Variable'),
            dcc.Dropdown([var for var in variables_first_country], variables_first_country[0], id='var-choice'),
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Plaintext('Click on a coefficient to plot the corresponding variable')
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dash_table.DataTable(data=temp_df.to_dict('records'),
                             columns=[{"name": i, "id": i, "selectable": True} for i in temp_df.columns],
                             id='corr-table')
    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    html.Br(),
    html.Div([
        dcc.Graph(id='corr-time-graph')
    ], style={'width': '48%', 'float': 'left', 'display': 'inline-block', 'padding': '0 20', 'margin-bottom': '5cm'}),
    html.Div([
        dcc.Graph(id='corr-graph')
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'margin-bottom': '5cm'}),

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
            dcc.Dropdown(columns_fixed, columns_fixed[0], id='size-dot-dependence'),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'margin-bottom': '0.5cm'}),
    dcc.Graph(id='total-dependence-graph'),
    dcc.Slider(
        0,
        len(months_list) - 1,
        marks={i: str(months_list[i]) for i in range(len(months_list))},
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
])


#####################
# Filtering
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
        constraint_added.clear()
        string = u'0 conditions added'
        times_clicked = 0
        new_df = original_df
        new_month_df = get_month_df(original_df)

    elif var_filter == 'None':
        string = u'{} conditions added'.format(max([0, number_conditions_added - 1]))
        times_clicked = max([0, number_conditions_added - 1])
    else:
        constraint_added.append([var_filter, sign_filter, num_filter])
        new_df = apply_constraints(new_df, constraint_added)
        new_month_df = get_month_df(new_df)
        string = u'{} conditions added'.format(number_conditions_added)
        times_clicked = number_conditions_added
    return string, times_clicked, new_df.to_json(date_format='iso', orient='split'), new_month_df.to_json(
        date_format='iso', orient='split')


#######################
# Multi variables
@app.callback(
    Output('y-axis', 'options'),
    Output('y-axis', 'value'),
    Input('country-continent-choice', 'value'),
    Input('df', 'data'))
def y_axis_based_on_location(country_cont_choice, data):
    used_df = pd.read_json(data, orient='split')
    used_df['date'] = used_df['date'].dt.strftime('%Y-%m-%d')
    variables_to_show = variables_each_country[country_cont_choice]
    for col in columns_to_remove:
        if col in variables_to_show:
            variables_to_show.remove(col)
    return variables_to_show, [variables_to_show[0], variables_to_show[1]]


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
    fig.update_layout(title='Evolution of the chosen variables over time')

    return fig


#######################
# Correlations
@app.callback(
    Output('country-choice', 'options'),
    Output('country-choice', 'value'),
    Input('df', 'data'))
def change_available_countries(data):
    used_df = pd.read_json(data, orient='split')
    used_df['date'] = used_df['date'].dt.strftime('%Y-%m-%d')

    all_countries = used_df['location']
    return all_countries, all_countries[0]


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
    Input('var-choice', 'value'),
    Input('df', 'data'))
def update_table_corr(country_choice, var_choice, data):
    stored_df = pd.read_json(data, orient='split')
    stored_df['date'] = stored_df['date'].dt.strftime('%Y-%m-%d')
    all_features = variables_each_country[country_choice].copy()

    for column in columns_to_remove:
        if column in all_features:
            all_features.remove(column)

    for column in columns_fixed:
        if column in all_features:
            all_features.remove(column)

    country_df = stored_df[stored_df['location'] == country_choice][all_features]
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
            if len(pos_variables_most_corr) == 5:
                break

    neg_corr = corr_mat[corr_mat[var_choice] < 0.0]
    neg_corr = neg_corr.sort_values(by=[var_choice], ascending=True)[var_choice]

    neg_variables_most_corr = []
    neg_corr_coeffs = []
    neg_all_vars = neg_corr.index
    for var in neg_all_vars:
        neg_variables_most_corr.append(var)
        neg_corr_coeffs.append(neg_corr[var])
        if len(neg_variables_most_corr) == 5:
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
    Output('corr-time-graph', 'figure'),
    Output('corr-graph', 'figure'),
    Input('country-choice', 'value'),
    Input('var-choice', 'value'),
    Input('corr-table', 'active_cell'),
    Input('corr-table', 'data'),
    Input('df', 'data'))
def update_graphs(country_choice, var_choice, active_cell, data, data_stored):
    stored_df = pd.read_json(data_stored, orient='split')
    stored_df['date'] = stored_df['date'].dt.strftime('%Y-%m-%d')
    if active_cell:
        cell_clicked = active_cell['row']
        var_clicked = data[cell_clicked]['variables']
    else:
        var_clicked = var_choice
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    all_dates = stored_df[stored_df['location'] == country_choice]['date']
    data_1 = stored_df[stored_df['location'] == country_choice][var_choice].to_list()
    fig.add_trace(
        go.Scatter(x=all_dates, y=data_1, name=var_choice),
        secondary_y=False,
    )
    data_2 = stored_df[stored_df['location'] == country_choice][var_clicked].to_list()
    fig.add_trace(
        go.Scatter(x=all_dates, y=data_2, name=var_clicked),
        secondary_y=True,
    )
    fig.update_layout(title=str('Evolution of ' + var_choice + ' and ' + var_clicked + ' over time'))
    fig.update_xaxes(title_text='Dates')

    fig.update_yaxes(title_text=var_choice, secondary_y=False)
    fig.update_yaxes(title_text=var_clicked, secondary_y=True)

    fig2 = px.scatter(x=data_1, y=data_2)
    fig2.update_xaxes(title_text=var_choice)

    fig2.update_yaxes(title_text=var_clicked)
    fig2.update_layout(title=str(var_clicked + ' depending on ' + var_choice),
                       xaxis={'autorange': False, 'range': [min(data_1), max(data_1)]},
                       yaxis={'autorange': False, 'range': [min(data_2), max(data_2)]})
    return fig, fig2


#######################
# Dependencies
@app.callback(
    Output('total-dependence-graph', 'figure'),
    Input('x-axis-dependence', 'value'),
    Input('y-axis-dependence', 'value'),
    Input('month-slider-dependence', 'value'),
    Input('size-dot-dependence', 'value'),
    Input('month-df', 'data'))
def update_dependence_graphs(x_axis_var, y_axis_var, month, size_dot, month_data):
    stored_df = pd.read_json(month_data, orient='split')
    stored_df['date'] = stored_df['date'].dt.strftime('%Y-%m-%d')
    month = months_list[month]
    all_countries = stored_df['location'].unique()
    x_values = []
    y_values = []
    all_continents = []
    size_dot_values = []
    for country in all_countries:
        country_df = stored_df[stored_df['location'] == country]
        if x_axis_var in variables_each_country[country]:
            country_df_x = country_df[['month', x_axis_var]]
            if month in country_df_x['month'].unique():
                x_values.append(country_df_x[country_df_x['month'] == month][x_axis_var].mean())
            else:
                x_values.append(0)
        else:
            x_values.append(0)

        if y_axis_var in variables_each_country[country]:
            country_df_y = country_df[['month', y_axis_var]]
            if month in country_df_y['month'].unique():
                y_values.append(country_df_y[country_df_y['month'] == month][y_axis_var].mean())
            else:
                y_values.append(0)
        else:
            y_values.append(0)

        all_continents.append(country_df['continent'].iloc[0])
        if size_dot == 'trust_in_gov':
            val = trust_df[trust_df['location'] == country]['trust_in_gov'].item()
        else:
            val = country_df[size_dot].iloc[0]
        if not math.isnan(val) and val != 0:
            size_dot_values.append(val)
        else:
            size_dot_values.append(1)
    new_df = pd.DataFrame({'country': all_countries, 'continent': all_continents, x_axis_var: x_values,
                           y_axis_var: y_values, size_dot: size_dot_values})

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
    used_df['date'] = used_df['date'].dt.strftime('%Y-%m-%d')

    all_countries = used_df['location']
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

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
