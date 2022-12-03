import dash
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
# import dash_daq as daq
import plotly.express as px
import pandas as pd
from jupyter_dash import JupyterDash

app = Dash(__name__)

light_theme = {
    "main-background": "#ffe7a6",
    "header-text": "#376e00",
    "sub-text": "#0c5703",
}

dark_theme = {
    "main-background": "#000000",
    "header-text": "#ff7575",
    "sub-text": "#ffd175",
}

app.layout = html.Div(
    id="parent_div",
    children=[
        daq.BooleanSwitch(on=False, id="bool-switch-input"),
        html.Div(id="bool-switch-output"),
        html.H1(
            children="This is a heading text",
            id="head-txt",
            style={"color": light_theme["header-text"]},
        ),
        html.H2(
            children="This is a subtext",
            id="sub-txt",
            style={"color": light_theme["sub-text"]},
        ),
    ],
    style={"backgroundColor": light_theme["main-background"]},
)


@app.callback(
    [
        Output("parent_div", "style"),
        Output("head-txt", "style"),
        Output("sub-txt", "style"),
    ],
    Input("bool-switch-input", "on"),
)
def update_output(on):
    theme = dark_theme if on else light_theme

    return (
        {"backgroundColor": theme["main-background"]},
        {"color": theme["header-text"]},
        {"color": theme["sub-text"]},
    )


if __name__ == "__main__":
    app.run_server(mode="inline")