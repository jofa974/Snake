import torch
import torch.nn.functional as F
import torch.optim as optim

import ui
from components.apple import Apple
from components.snake import Snake
from neural_net.pytorch_cnn import ConvolutionalNeuralNetwork, ReplayMemory


class DQN_CNN(DQN):
    def __init__(self, nb_actions=3, gamma=0.9):
        input_size = (3, ui.X_GRID, ui.Y_GRID)
        super().__init__(
            input_size=input_size, nb_actions=nb_actions, gamma=gamma
        )
        self.env.set_caption("Snake: Pytorch Convolutional Neural Network")

        self.model = ConvolutionalNeuralNetwork(input_size, nb_actions)
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(100)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.brain_file = "last_brain.pth"
