import numpy as np
import torch.optim as optim
import ui
from neural_net.pytorch_cnn import ConvolutionalNeuralNetwork, ReplayMemory
from torch import nn

from .dqn import DQN


class DQN_CNN(DQN):
    def __init__(
        self,
        batch_size=128,
        gamma=0.9,
        memory_size=200,
        learning=True,
    ):
        super().__init__(
            batch_size=batch_size,
            gamma=gamma,
            memory_size=memory_size,
            learning=learning,
        )
        self.input_size = (1, ui.X_GRID, ui.Y_GRID)
        self.caption("Snake: Pytorch Convolutional Neural Network")

        nb_actions = 3
        self.model = ConvolutionalNeuralNetwork(self.input_size, nb_actions)
        self.model.to(self.device)
        self.memory = ReplayMemory(500)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.batch_size = 20
        self.brain_file = "output/dqn_cnn/last_brain.pth"

    def get_input_data(self):
        arr = np.zeros(self.input_size)
        # Walls
        arr[0, 0, :] = 0.1
        arr[0, -1, :] = 0.1
        arr[0, :, 0] = 0.1
        arr[0, :, -1] = 0.1
        # Apple
        apple_pos = self.apple.get_position()
        arr[0, apple_pos[0], apple_pos[1]] = 1.0
        # Snake
        for idx in range(len(self.snake.body_list)):
            position = self.snake.get_position(idx)
            arr[0, position[0], position[1]] = 0.5
        return arr
