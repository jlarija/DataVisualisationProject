import dash
import dash_bootstrap_components as dbc

#Instantiates the Dash app and identify the server
app = dash.Dash(__name__)
server = app.server

#### COLORS
navig_bar = '#BBBBBB'
legend_fonts = '#FFFFFF'
background_color = '#121212'
font_size = 15