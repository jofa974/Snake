from neural_net.neural_network import NeuralNetwork

nn = NeuralNetwork()


def test_nb_weights():
    expected = 6 * 4 + 4 * 5 + 5 * 3

    result = nn.nb_weights

    assert result == expected
