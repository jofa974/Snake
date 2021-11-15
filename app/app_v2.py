#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
import pygame
from dash.dependencies import Input, Output, State

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {"background": "#111111", "text": "#7FD BFF"}

nb_ind = 100
nb_gen = 150
all_fitness = np.random.random((nb_ind, nb_gen))


@app.callback(
    Output("fitness-figure", "figure"),
    [Input("generation", "value"), Input("individual", "value")],
)
def build_fitness_figure(generation, individual):
    if generation == "":
        generation = 0
    if individual == "":
        individual = 0
    fig = go.Figure(
        data=[go.Heatmap(z=all_fitness.T)],
        layout={
            "title": {"text": "All fitness", "x": 0.5},
            "plot_bgcolor": colors["background"],
            "paper_bgcolor": colors["background"],
            "font": {"color": colors["text"]},
        },
    )
    fig.add_trace(
        go.Scatter(
            x=[generation],
            y=[individual],
            marker=dict(color="crimson", size=12),
            mode="markers",
            name="Current",
        )
    )
    return fig


app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Snake AI", style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="An AI agent learning to play Snake.",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Hr(),
        html.Div(
            [html.Button("Learn", id="learn-button"), html.Div(id="learn-status"),],
        ),
        html.Hr(),
        html.Div(
            [html.Button("Load data", id="load-button"), html.Div(id="load-status"),],
        ),
        html.Hr(),
        html.Div([html.Button("Play", id="play-button"), html.Div(id="play-status"),],),
        html.Hr(),
        html.Div(
            [
                html.Label("Generation"),
                dcc.Slider(
                    id="generation",
                    min=1,
                    max=nb_gen,
                    value=1,
                    marks={str(i): str(i) for i in range(0, nb_gen + 1, 5)},
                    step=1,
                ),
            ],
            style={"width": "49%", "display": "inline-block",},
        ),
        html.Div(
            [
                html.Label("Individual"),
                dcc.Slider(
                    id="individual",
                    min=1,
                    max=nb_ind,
                    value=1,
                    marks={str(i): str(i) for i in range(0, nb_ind + 1, 200)},
                    step=1,
                ),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [dcc.Graph(id="nn-figure"),],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [dcc.Graph(id="fitness-figure"),],
            style={"width": "49%", "display": "inline-block"},
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
