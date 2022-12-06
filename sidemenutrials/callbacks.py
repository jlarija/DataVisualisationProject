from layouts import *

'''Here are all the callbacks'''

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
# Chorpleth Map
@app.callback(
    Output('Choropleth Map', 'figure'),
    Input('chorplethdropdown', 'value'),
    Input('monthchoroplethmap', 'value') # gives a numerical value
)
def choropleth_map(choroplethdropdown, monthchoroplethmap):
    global df

    my_df = get_month_df(df) # Split months cause slider
    my_df = my_df.groupby(['iso_code','month'], sort=False).mean().reset_index()
    my_df = my_df[my_df['iso_code'].str.contains('OWID')==False]
    
    colorscale = ['#FFFDE7','#FFF59D','#FFEE58', '#FDD835','#F9A825', '#F57F17']

    current_month = months_list[monthchoroplethmap]
    my_df = my_df[my_df['month'] == current_month]
    min_color = np.max(my_df[str(choroplethdropdown)])
    max_color = np.min(my_df[str(choroplethdropdown)])

    fig = px.choropleth(my_df, locations = 'iso_code', color = str(choroplethdropdown),
    color_continuous_scale = colorscale,hover_name="iso_code",range_color = (min_color,max_color))

    background_color = '#282a36'

    fig.update_layout(font_family = 'Balto',font_color = '#FFFFFF',
    font_size = 18, plot_bgcolor=background_color,paper_bgcolor = background_color,
    coloraxis_colorbar_x=-0.15, coloraxis_colorbar_y=0.5,
    margin=dict(l=0, r=20, b=0, t=0,autoexpand=True),
        geo=dict(
            showframe=False,
            showcoastlines=False,
            countrycolor='#FFFFFF',
            bgcolor=background_color,
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
    color_options = ['#ffb14e','#fa8775','#ea5f94','#cd34b5','#9d02d7','#5854e2']
    dates = used_df['date']
    for i in range(len(variables_chosen)):
        if i == 0:
            fig.add_trace(go.Scatter(
                x=dates,
                y=used_df[variables_chosen[i]],
                name=variables_chosen[i],
                line=dict(color='#ffd700')
            ))
        else:
            fig.add_trace(go.Scatter(
                x=dates,
                y=used_df[variables_chosen[i]],
                name=variables_chosen[i],
                yaxis="y" + str(i + 1),
                line=dict(color=color_options[i])
            ))

    layout = {}
    layout['yaxis'] = {'tickfont': {'color': '#ffd700'},
                       'title': {'font': {'color':'#ffd700' }, 'text': variables_chosen[0]}}
    layout['xaxis'] = {'domain': [0.3, 0.9]}

    for i in range(len(variables_chosen)):
        if i == 0:
            continue
        else:
            color_hex = color_options[i]
            pos = i * 0.3 / len(variables_chosen)
            layout['yaxis' + str(i + 1)] = {'anchor': 'free', 'position': pos, 'overlaying': 'y', 'side': 'left',
                                            'tickfont': {'color': color_hex},
                                            'title': {'font': {'color': color_hex}, 'text': variables_chosen[i]}}
    layout['plot_bgcolor'] = '#282a36'
    layout['paper_bgcolor'] = '#282a36'

    fig.update_layout(layout,legend_font_color='#BF360C',
    margin=dict(l=0, r=5, b=0, t=0,autoexpand=True))
    # fig.update_layout(title='Evolution of the chosen variables over time')

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
    fig = px.line(prediction_df, x="date", y="value",)
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.95, y0=0,
        x1=1, y1=1,
        line=dict(
            color="#B2DFDB",
            width=3,
        ),
        layer="below",
        fillcolor="#B2DFDB",
        )
    
    fig.update_yaxes(title=str(data_to_predict + " predicted for next 3 months"), color='#03DAC6')
    fig.update_xaxes(color='#03DAC6')
    fig.update_layout(plot_bgcolor = '#282a36',paper_bgcolor = '#282a36')
    fig.update_traces(line_color='#03DAC6', line_width=4)
    
    
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