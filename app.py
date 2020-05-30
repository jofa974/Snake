#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

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


@app.callback(
    Output("nn-figure", "figure"),
    [Input("generation", "value"), Input("individual", "value")],
)
def build_nn_figure(generation, individual):
    if generation == "":
        generation = 0
    if individual == "":
        individual = 0
    neural_network = NeuralNetwork(
        gen_id=(int(generation), int(individual)), dna=None, hidden_nb=[4]
    )
    left, right, bottom, top = (
        0.1,
        0.9,
        0.1,
        0.9,
    )
    layer_sizes = list(
        itertools.chain(
            [neural_network.input_nb],
            neural_network.hidden_nb[:],
            [neural_network.output_nb],
        )
    )
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
            marker=dict(size=20, color=neural_network.act, colorscale="Viridis"),
        ),
        layout={
            "title": {"text": "Neural Network", "x": 0.5},
            "plot_bgcolor": colors["background"],
            "paper_bgcolor": colors["background"],
            "font": {"color": colors["text"]},
            "margin": {"r": 0, "l": 0, "b": 0,},
            "showlegend": False,
            "yaxis": {
                "range": [0, 1],
                "showgrid": False,
                "zeroline": False,
                "visible": False,
            },
            "xaxis": {
                "range": [0, 1],
                "showgrid": False,
                "zeroline": False,
                "visible": False,
            },
        },
    )
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
        i = 0
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                if neural_network.weights[i] < 0:
                    color = "red"
                else:
                    color = "blue"

                fig.add_shape(
                    type="line",
                    x0=n * h_spacing + left,
                    x1=(n + 1) * h_spacing + left,
                    y0=layer_top_a - m * v_spacing,
                    y1=layer_top_b - o * v_spacing,
                    line=dict(color=color, width=abs(neural_network.weights[i]) * 1,),
                )
                i += 1

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
            [
                html.Label("Generation"),
                dcc.Input(id="generation", value="0", type="text"),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Individual"),
                dcc.Input(id="individual", value="0", type="text"),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [dcc.Graph(id="nn-figure"),],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [dcc.Graph(id="fitness-figure", figure=build_fitness_figure()),],
            style={"width": "49%", "display": "inline-block"},
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
