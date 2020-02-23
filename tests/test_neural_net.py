import itertools

import matplotlib.pyplot as plt
import numpy as np

from neural_net.neural_network import NeuralNetwork, forward_layer


def test_nb_weights():
    nn = NeuralNetwork(hidden_nb=[4, 5])

    expected = 6 * 4 + 4 * 5 + 5 * 3

    result = nn.nb_weights

    assert result == expected


def test_get_layer_data():
    nn = NeuralNetwork(hidden_nb=[4, 5, 6])

    nn.act[6:10] = np.ones(4)
    nn.act[10:15] = np.ones(5) * 2
    nn.act[15:21] = np.ones(6) * 3

    assert np.all(nn.get_layer_data(-1, nn.act) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.all(nn.get_layer_data(0, nn.act) == [1.0, 1.0, 1.0, 1.0])
    assert np.all(nn.get_layer_data(1, nn.act) == [2.0, 2.0, 2.0, 2.0, 2.0])
    assert np.all(nn.get_layer_data(2, nn.act) == [3.0, 3.0, 3.0, 3.0, 3.0, 3.0])


def test_count_weights_before_layer():
    nn = NeuralNetwork(hidden_nb=[4, 5])

    assert nn.count_weights_before_layer(0) == 0
    assert nn.count_weights_before_layer(1) == 24
    assert nn.count_weights_before_layer(2) == 44


def test_weights_layer_idx():
    nn = NeuralNetwork(hidden_nb=[4])

    w1 = [[i] * 4 for i in range(6)]
    w1 = [item for sublist in w1 for item in sublist]
    w2 = [[i] * 3 for i in range(10, 14)]
    w2 = [item for sublist in w2 for item in sublist]

    nn.weights = list(itertools.chain(w1, w2))

    expected_layer_0 = [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
    ]
    expected_layer_1 = [10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13]

    assert np.all(expected_layer_0 == nn.weights_layer_idx(0))
    assert np.all(expected_layer_1 == nn.weights_layer_idx(1))


def test_plot():
    nn = NeuralNetwork(hidden_nb=[6, 5, 4])

    nn.act[6:12] = np.ones(6)
    nn.act[12:17] = np.ones(5) * 2
    nn.act[17:21] = np.ones(4) * 3

    fig = nn.plot()
    plt.show()


def test_forward_layer():
    input_nb = 3
    output_nb = 2

    act = np.ones(input_nb)

    weights = np.ones(output_nb * input_nb)
    biases = np.ones(output_nb)

    def func(x):
        return x

    expected = np.array([4.0, 4.0])

    result = forward_layer(weights, biases, act, func)
    assert np.all(np.equal(result, expected))
