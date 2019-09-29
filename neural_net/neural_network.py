import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self):
        self.input_nb = 6
        self.output_nb = 3
        self.hidden_nb = 5

        self.nb_neurons = self.input_nb + self.hidden_nb + self.output_nb
        self.weights_1 = np.random.randn(self.input_nb, self.hidden_nb)
        self.weights_2 = np.random.randn(self.hidden_nb, self.output_nb)
        self.act = np.zeros(self.nb_neurons)
        self.x = np.zeros(self.nb_neurons)
        self.y = np.zeros(self.nb_neurons)

    # def draw(self):
    #     surface = pygame.display.set_mode((2 * ui.WIDTH, ui.HEIGHT))
    #     pygame.draw.circle(surface, color, center, radius)

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def activate(self, input_data):
        self.act_input = input_data
        self.act_hidden = self.sigmoid(
            np.dot(self.weights_1, self.act_input) + self.bias_hidden)
        self.act_output = self.sigmoid(
            np.dot(self.weights_2, self.act_hidden) + self.bias_output)

    def plot(self):
        max_r = max(self.input_nb, self.output_nb, self.hidden_nb)

        self.x_input = np.zeros(self.input_nb)
        self.x_hidden = np.ones(self.hidden_nb)
        self.x_output = np.ones(self.output_nb) * 2

        self.y_input = np.linspace(1, max_r, self.input_nb)
        self.y_hidden = np.linspace(1, max_r, self.hidden_nb)
        self.y_output = np.linspace(1, max_r, self.output_nb)

        plt.figure()
        ms = 60
        col = 'k'
        plt.scatter(self.x, self.y, s=ms)

        for i in range(self.input_nb):
            for j in range(self.hidden_nb):
                if self.weights_1[i, j] > 0:
                    plt.plot((self.x[i], self.x[self.input_nb + j]),
                             (self.y[i], self.y[self.input_nb + j]), 'r')
                else:
                    plt.plot((self.x[i], self.x[self.input_nb + j]),
                             (self.y[i], self.y[self.input_nb + j]), 'b')

        for i in range(self.hidden_nb):
            for j in range(self.output_nb):
                if self.weights_2[i, j] > 0:
                    plt.plot((self.x[self.input_nb + i],
                              self.x[self.input_nb + self.hidden_nb + j]),
                             (self.y[self.input_nb + i],
                              self.y[self.input_nb + self.hidden_nb + j]), 'r')
                else:
                    plt.plot((self.x[self.input_nb + i],
                              self.x[self.input_nb + self.hidden_nb + j]),
                             (self.y[self.input_nb + i],
                              self.y[self.input_nb + self.hidden_nb + j]), 'b')

        plt.show()

    @property
    def act_input(self):
        return self.act[:self.input_nb]

    @act_input.setter
    def act_input(self, value):
        self.act[:self.input_nb] = value

    @property
    def act_hidden(self):
        return self.act[self.input_nb:self.input_nb + self.hidden_nb]

    @act_hidden.setter
    def act_hidden(self, value):
        self.act[self.input_nb:self.input_nb + self.hidden_nb] = value

    @property
    def act_output(self):
        return self.act[self.input_nb + self.hidden_nb:]

    @act_output.setter
    def act_output(self, value):
        self.act[self.input_nb + self.hidden_nb:] = value

    @property
    def bias_input(self):
        return self.bias[:self.input_nb]

    @bias_input.setter
    def bias_input(self, value):
        self.bias[:self.input_nb] = value

    @property
    def bias_hidden(self):
        return self.bias[self.input_nb:self.input_nb + self.hidden_nb]

    @bias_hidden.setter
    def bias_hidden(self, value):
        self.bias[self.input_nb:self.input_nb + self.hidden_nb] = value

    @property
    def bias_output(self):
        return self.bias[self.input_nb + self.hidden_nb:]

    @bias_output.setter
    def bias_output(self, value):
        self.bias[self.input_nb + self.hidden_nb:] = value

    @property
    def x_input(self):
        return self.x[:self.input_nb]

    @x_input.setter
    def x_input(self, value):
        self.x[:self.input_nb] = value

    @property
    def x_hidden(self):
        return self.x[self.input_nb:self.input_nb + self.hidden_nb]

    @x_hidden.setter
    def x_hidden(self, value):
        self.x[self.input_nb:self.input_nb + self.hidden_nb] = value

    @property
    def x_output(self):
        return self.x[self.input_nb + self.hidden_nb:]

    @x_output.setter
    def x_output(self, value):
        self.x[self.input_nb + self.hidden_nb:] = value

    @property
    def y_input(self):
        return self.y[:self.input_nb]

    @y_input.setter
    def y_input(self, value):
        self.y[:self.input_nb] = value

    @property
    def y_hidden(self):
        return self.y[self.input_nb:self.input_nb + self.hidden_nb]

    @y_hidden.setter
    def y_hidden(self, value):
        self.y[self.input_nb:self.input_nb + self.hidden_nb] = value

    @property
    def y_output(self):
        return self.y[self.input_nb + self.hidden_nb:]

    @y_output.setter
    def y_output(self, value):
        self.y[self.input_nb + self.hidden_nb:] = value


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.plot()
