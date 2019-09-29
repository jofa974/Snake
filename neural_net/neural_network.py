import numpy as np
import matplotlib.pyplot as plt

# class Layer():

#     def __init__(


class NeuralNetwork():
    def __init__(self):
        self.input_nb = 6
        self.output_nb = 3
        self.hidden_nb = 5

        self.nb_neurons = self.input_nb + self.hidden_nb + self.output_nb
        self.weights_1 = np.random.randn(self.input_nb, self.hidden_nb)
        self.weights_2 = np.random.randn(self.hidden_nb, self.output_nb)

    # def draw(self):
    #     surface = pygame.display.set_mode((2 * ui.WIDTH, ui.HEIGHT))
    #     pygame.draw.circle(surface, color, center, radius)

    def plot(self):
        max_r = max(self.input_nb, self.output_nb, self.hidden_nb)
        x = np.zeros(self.nb_neurons)
        y = np.zeros(self.nb_neurons)

        x[:self.input_nb] = np.zeros(self.input_nb)
        x[self.input_nb:self.hidden_nb + self.input_nb] = np.ones(
            self.hidden_nb)
        x[self.hidden_nb +
          self.input_nb:self.nb_neurons] = np.ones(self.output_nb) * 2

        y[:self.input_nb] = np.linspace(1, max_r, self.input_nb)
        y[self.input_nb:self.hidden_nb + self.input_nb] = np.linspace(
            1, max_r, self.hidden_nb)
        y[self.hidden_nb + self.input_nb:self.nb_neurons] = np.linspace(
            1, max_r, self.output_nb)

        plt.figure()
        ms = 60
        col = 'k'
        plt.scatter(x, y, s=ms, facecolor=col)

        for i in range(self.input_nb):
            for j in range(self.hidden_nb):
                if self.weights_1[i, j] > 0:
                    plt.plot((x[i], x[self.input_nb+j]),
                             (y[i], y[self.input_nb+j]), 'r')
                else:
                    plt.plot((x[i], x[self.input_nb+j]),
                             (y[i], y[self.input_nb+j]), 'b')

        for i in range(self.hidden_nb):
            for j in range(self.output_nb):
                if self.weights_2[i, j] > 0:
                    plt.plot((x[self.input_nb+i], x[self.input_nb+self.hidden_nb+j]),
                             (y[self.input_nb+i], y[self.input_nb+self.hidden_nb+j]), 'r')
                else:
                    plt.plot((x[self.input_nb+i], x[self.input_nb+self.hidden_nb+j]),
                             (y[self.input_nb+i], y[self.input_nb+self.hidden_nb+j]), 'b')

        plt.show()


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.plot()
