import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
import torch.optim as optim
from PIL import Image
from torch import nn

import ui
from neural_net.pytorch_cnn import ConvolutionalNeuralNetwork, ReplayMemory

from .dqn import DQN


class DQN_CNN(DQN):
    def __init__(self, nb_actions=3, gamma=0.98, do_display=False, learning=True):
        self.input_size = (1, ui.X_GRID, ui.Y_GRID)
        super().__init__(
            input_size=self.input_size,
            nb_actions=nb_actions,
            gamma=gamma,
            do_display=do_display,
            learning=learning,
        )
        self.env.set_caption("Snake: Pytorch Convolutional Neural Network")

        self.model = ConvolutionalNeuralNetwork(self.input_size, nb_actions)
        self.memory = ReplayMemory(1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.batch_size = 32

    def get_input_data(self):
        self.env.take_screenshot()
        with Image.open("screenshot.png") as img:
            img_conv = img.convert("L")
            # img_conv = np.transpose(img_conv, (2, 0, 1))
            arr = np.asarray(img_conv)
            arr = np.array([arr[:: ui.BASE_SIZE, :: ui.BASE_SIZE] / 255.0])
        return arr

