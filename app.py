#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from neural_net.neural_network import NeuralNetwork
from stats.stats import read_fitness

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {"background": "#111111", "text": "#7FD BFF"}


def build_fitness_figure():
    all_fitness = read_fitness(11, 500)
    fig = go.Figure(
        data=[go.Heatmap(z=all_fitness.T)],
        layout={
            "title": {"text": "All fitness", "x": 0.5},
            "plot_bgcolor": colors["background"],
            "paper_bgcolor": colors["background"],
            "font": {"color": colors["text"]},
        },
    )
    return fig


def build_nn_figure():
    nn = NeuralNetwork(gen_id=(19, 0), dna=None, hidden_nb=[4])
    nn.act[5] = 1
    left, right, bottom, top = (
        0.1,
        0.9,
        0.1,
        0.9,
    )
    layer_sizes = list(itertools.chain([nn.input_nb], nn.hidden_nb[:], [nn.output_nb]))
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    x = []
    y = []
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size):
            x.append(n * h_spacing + left)
            y.append(layer_top - m * v_spacing)

    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=20, color=nn.act, colorscale="Viridis"),
        ),
        layout={"showlegend": False,},
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
        html.Div(
            [dcc.Graph(id="", figure=build_fitness_figure())],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [dcc.Graph(id="", figure=build_nn_figure())],
            style={"width": "49%", "display": "inline-block"},
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
