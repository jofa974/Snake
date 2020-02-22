import matplotlib.pyplot as plt
import numpy as np

from neural_net.neural_network import NeuralNetwork, forward_layer


def test_nb_weights():
    nn = NeuralNetwork(hidden_nb=[4, 5])

    expected = 6 * 4 + 4 * 5 + 5 * 3

    result = nn.nb_weights

    assert result == expected


def test_activation_layer():
    nn = NeuralNetwork(hidden_nb=[4, 5, 6])

    nn.act[6:10] = np.ones(4)
    nn.act[10:15] = np.ones(5) * 2
    nn.act[15:21] = np.ones(6) * 3

    assert np.all(nn.activation_layer(-1) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.all(nn.activation_layer(0) == [1.0, 1.0, 1.0, 1.0])
    assert np.all(nn.activation_layer(1) == [2.0, 2.0, 2.0, 2.0, 2.0])
    assert np.all(nn.activation_layer(2) == [3.0, 3.0, 3.0, 3.0, 3.0, 3.0])


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
