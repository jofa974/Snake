import itertools
import pickle
from pathlib import Path

import matplotlib
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pygame
from scipy.special import softmax


def forward_layer(weights, bias, act, func):
    reshaped_weights = np.reshape(weights, [len(bias), len(act)])
    return func(np.dot(reshaped_weights, act) + bias)


# TODO put this in a better place
def create_surf_from_figure_on_canvas(fig):
    matplotlib.use("Agg")
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    return surf


class ANN:
    def __init__(self, gen_id=(-1, -1), dna=None, hidden_nb=[5, 4]):
        self.input_nb = 6
        self.output_nb = 3
        self.hidden_nb = hidden_nb

        self.nb_neurons = self.input_nb + sum(self.hidden_nb) + self.output_nb
        self.load_data(gen_id, dna)
        self.act = np.zeros(self.nb_neurons)

    @staticmethod
    def sigmoid(s):
        # activation function
        return 1 / (1 + np.exp(-s))

    @staticmethod
    def relu(s):
        # activation function
        return np.where(s < 0, 0, s)
        # return max(0, s)

    @staticmethod
    def softmax(s):
        return softmax(s)

    def decide_direction(self):
        decisions = ["forward", "left", "right"]
        idx_max = np.argmax(self.get_layer_data(len(self.hidden_nb), self.act))
        return decisions[idx_max]

    def forward(self, input_data):
        self.set_layer_data(-1, self.act, input_data)
        for n_layer in range(len(self.hidden_nb) + 1):
            ww = self.weights_layer_idx(n_layer)
            aa = self.get_layer_data(n_layer - 1, self.act)
            bb = self.get_layer_data(n_layer, self.bias)
            self.set_layer_data(
                n_layer, self.act, forward_layer(ww, bb, aa, self.relu)
            )

    @property
    def nb_weights(self):
        list_nb = list(
            itertools.chain(
                [self.input_nb], self.hidden_nb[:], [self.output_nb]
            )
        )
        res = sum([i * j for i, j in zip(list_nb, list_nb[1:])])
        return res

    def count_weights_before_layer(self, layer_idx):
        if layer_idx == 0:
            return 0
        elif layer_idx < sum(self.hidden_nb) + 2:
            list_nb = list(
                itertools.chain([self.input_nb], self.hidden_nb[:layer_idx])
            )
            weights_nb = sum([i * j for i, j in zip(list_nb, list_nb[1:])])
            return weights_nb
        else:
            raise ValueError("Invalid layer index")

    def get_layer_data(self, layer_idx, data):
        if layer_idx == -1:
            return data[0 : self.input_nb]
        elif layer_idx == 0:
            return data[
                self.input_nb : self.input_nb + self.hidden_nb[layer_idx]
            ]
        elif layer_idx == len(self.hidden_nb):
            return data[self.input_nb + sum(self.hidden_nb) :]
        else:
            return data[
                self.input_nb
                + sum(self.hidden_nb[:layer_idx]) : self.input_nb
                + sum(self.hidden_nb[: layer_idx + 1])
            ]

    def set_layer_data(self, layer_idx, data, values):
        if layer_idx == -1:
            data[0 : self.input_nb] = values
        elif layer_idx == 0:
            data[
                self.input_nb : self.input_nb + self.hidden_nb[layer_idx]
            ] = values
        elif layer_idx == len(self.hidden_nb):
            data[self.input_nb + sum(self.hidden_nb) :] = values
        else:
            data[
                self.input_nb
                + sum(self.hidden_nb[:layer_idx]) : self.input_nb
                + sum(self.hidden_nb[: layer_idx + 1])
            ] = values

    def weights_layer_idx(self, layer_idx):
        if layer_idx == 0:
            return self.weights[: self.input_nb * self.hidden_nb[0]]
        elif layer_idx == len(self.hidden_nb):
            inf = self.count_weights_before_layer(layer_idx)
            return self.weights[inf:]
        else:
            inf = self.count_weights_before_layer(layer_idx)
            sup = (
                inf + self.hidden_nb[layer_idx - 1] * self.hidden_nb[layer_idx]
            )
            return self.weights[inf:sup]

    def plot(self, fig):
        left, right, bottom, top = (
            0.1,
            0.9,
            0.1,
            0.9,
        )
        layer_sizes = list(
            itertools.chain(
                [self.input_nb], self.hidden_nb[:], [self.output_nb]
            )
        )
        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / float(len(layer_sizes) - 1)

        plt.cla()
        plt.clf()
        ax = plt.gca()
        ax.axis("off")

        x = []
        y = []
        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = (
                v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
            )
            for m in range(layer_size):
                x.append(n * h_spacing + left)
                y.append(layer_top - m * v_spacing)

        ax.scatter(x, y, s=100, c=self.act)

        weights = self.weights
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layer_top_a = (
                v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
            )
            layer_top_b = (
                v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
            )
            i = 0
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    if weights[i] < 0:
                        color = "r"
                    else:
                        color = "b"
                    line = plt.Line2D(
                        [n * h_spacing + left, (n + 1) * h_spacing + left],
                        [
                            layer_top_a - m * v_spacing,
                            layer_top_b - o * v_spacing,
                        ],
                        c=color,
                        lw=abs(weights[i]) * 1,
                    )
                    ax.add_artist(line)
                    i += 1
        ax.set_aspect("equal", adjustable="box")

    def dump_data(self, gen_id, fitness):
        file_path = Path(
            "genetic_data/data_{}_{}.pickle".format(gen_id[0], gen_id[1])
        )
        with open(file_path, "wb") as f:
            pickle.dump([fitness, self.weights, self.bias], f)

    def load_data(self, gen_id, dna=None):
        if dna is not None:
            self.weights = dna[0]
            self.bias = dna[1]
        else:
            file_path = Path(
                "genetic_data/data_{}_{}.pickle".format(gen_id[0], gen_id[1])
            )
            try:
                f = open(file_path, "rb")
                print(
                    "Loading generation {} id {}".format(gen_id[0], gen_id[1])
                )
                fitness, self.weights, self.bias = pickle.load(f)
            except IOError:
                # print("Initialising random NN")
                self.weights = np.random.normal(size=self.nb_weights)
                self.bias = np.random.normal(size=self.nb_neurons)


if __name__ == "__main__":
    nn = ANN(hidden_nb=[6, 5, 4])

    nn.act[6:12] = np.ones(6)
    nn.act[12:17] = np.ones(5) * 2
    nn.act[17:21] = np.ones(4) * 3

    fig = plt.figure()
    nn.plot(fig)
    plt.show()
