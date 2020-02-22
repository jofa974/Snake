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


class NeuralNetwork:
    def __init__(self, gen_id=(-1, -1), dna=None, hidden_nb=[4, 5]):
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
        idx_max = np.argmax(self.act_output)
        return decisions[idx_max]

    # def forward(self, input_data):
    #     self.act_input = input_data
    #     for n_layer in len(self.hidden_nb):
    #         w = self.weights(n_layer)
    #         a = self.act_hidden(n_layer-1)
    #         b = self.bias_hidden(n_layer-1)
    #         self.act_hidden(n_layer) = forward_layer(w, b, a, func)

    def plot(self):
        fig = plt.figure(figsize=[5, 5], dpi=100)
        left, right, bottom, top = (
            0.1,
            0.9,
            0.1,
            0.9,
        )
        layer_sizes = list(
            itertools.chain([self.input_nb], self.hidden_nb[:], [self.output_nb])
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
            layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
            for m in range(layer_size):
                x.append(n * h_spacing + left)
                y.append(layer_top - m * v_spacing)

        ax.scatter(x, y, s=100, c=self.act)

        weights = self.weights
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
            layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
            i = 0
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    if weights[i] < 0:
                        color = "r"
                    else:
                        color = "b"
                    line = plt.Line2D(
                        [n * h_spacing + left, (n + 1) * h_spacing + left],
                        [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                        c=color,
                    )
                    ax.add_artist(line)
                    i += 1
        ax.set_aspect("equal", adjustable="box")
        return fig

    def dump_data(self, gen_id, fitness):
        file_path = Path("genetic_data/data_{}_{}.pickle".format(gen_id[0], gen_id[1]))
        with open(file_path, "wb") as f:
            pickle.dump([fitness, self.weights, self.bias], f)

    def load_data(self, gen_id, dna=None):
        if dna is not None:
            self.weights = dna[0]
            self.bias = dna[2]
        else:
            file_path = Path(
                "genetic_data/data_{}_{}.pickle".format(gen_id[0], gen_id[1])
            )
            try:
                f = open(file_path, "rb")
                print("Loading generation {} id {}".format(gen_id[0], gen_id[1]))
                fitness, self.weights, self.bias = pickle.load(f)
            except IOError:
                print("Initialising random NN")
                self.weights = np.random.normal(size=self.nb_weights)
                self.bias = np.random.normal(size=self.nb_neurons)

    @property
    def nb_weights(self):
        list_nb = list(
            itertools.chain([self.input_nb], self.hidden_nb[:], [self.output_nb])
        )
        res = sum([i * j for i, j in zip(list_nb, list_nb[1:])])
        return res

    @property
    def act_input(self):
        return self.act[: self.input_nb]

    @act_input.setter
    def act_input(self, value):
        self.act[: self.input_nb] = value

    def act_hidden(self, idx):
        if idx == -1:
            return self.act[0 : self.input_nb]
        elif idx == 0:
            return self.act[self.input_nb : self.input_nb + self.hidden_nb[idx]]
        else:
            return self.act[
                self.input_nb
                + sum(self.hidden_nb[:idx]) : self.input_nb
                + sum(self.hidden_nb[: idx + 1])
            ]

    # @act_hidden.setter
    # def act_hidden(self, idx, value):
    #     self.act[self.input_nb : self.input_nb + sum(self.hidden_nb)] = value

    @property
    def act_output(self):
        return self.act[self.input_nb + sum(self.hidden_nb) :]

    @act_output.setter
    def act_output(self, value):
        self.act[self.input_nb + sum(self.hidden_nb) :] = value

    @property
    def bias_input(self):
        return self.bias[: self.input_nb]

    @bias_input.setter
    def bias_input(self, value):
        self.bias[: self.input_nb] = value

    @property
    def bias_hidden(self):
        return self.bias[self.input_nb : self.input_nb + sum(self.hidden_nb)]

    @bias_hidden.setter
    def bias_hidden(self, value):
        self.bias[self.input_nb : self.input_nb + sum(self.hidden_nb)] = value

    @property
    def bias_output(self):
        return self.bias[self.input_nb + sum(self.hidden_nb) :]

    @bias_output.setter
    def bias_output(self, value):
        self.bias[self.input_nb + sum(self.hidden_nb) :] = value

    @property
    def x_input(self):
        return self.x[: self.input_nb]

    @x_input.setter
    def x_input(self, value):
        self.x[: self.input_nb] = value

    @property
    def x_hidden(self):
        return self.x[self.input_nb : self.input_nb + sum(self.hidden_nb)]

    @x_hidden.setter
    def x_hidden(self, value):
        self.x[self.input_nb : self.input_nb + sum(self.hidden_nb)] = value

    @property
    def x_output(self):
        return self.x[self.input_nb + sum(self.hidden_nb) :]

    @x_output.setter
    def x_output(self, value):
        self.x[self.input_nb + sum(self.hidden_nb) :] = value

    @property
    def y_input(self):
        return self.y[: self.input_nb]

    @y_input.setter
    def y_input(self, value):
        self.y[: self.input_nb] = value

    @property
    def y_hidden(self):
        return self.y[self.input_nb : self.input_nb + sum(self.hidden_nb)]

    @y_hidden.setter
    def y_hidden(self, value):
        self.y[self.input_nb : self.input_nb + sum(self.hidden_nb)] = value

    @property
    def y_output(self):
        return self.y[self.input_nb + sum(self.hidden_nb) :]

    @y_output.setter
    def y_output(self, value):
        self.y[self.input_nb + sum(self.hidden_nb) :] = value


if __name__ == "__main__":
    nn = NeuralNetwork()
    input_data = np.random.randn(6)
    nn.forward(input_data)
    nn.plot()
