from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import callbacks
from app import *
# import your navigation, styles and layouts from layouts.py here
from layouts import nav_bar, layout1, layout2, CONTENT_STYLE

# Define basic structure of app:
# A horizontal navigation bar on the left side with page content on the right.
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # this locates this structure to the url
    nav_bar(),
    html.Div(id='page-content', style=CONTENT_STYLE)  # we'll use a callback to change the layout of this section
])


# This callback changes the layout of the page based on the URL
# For each layout read the current URL page "http://127.0.0.1:5000/pagename" and return the layout
@app.callback(Output('page-content', 'children'),  # this changes the content
              [Input('url', 'pathname')])  # this listens for the url in use
def display_page(pathname):
    if pathname == '/':
        return layout1
    elif pathname == '/page-1':
        return layout2
    else:
        return '404'  # If page not found return 404


if __name__ == '__main__':
    app.run_server(port=9877)
