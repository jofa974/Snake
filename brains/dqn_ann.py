import itertools
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import pygame
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import ui
from components.apple import Apple
from components.snake import Snake
from neural_net.pytorch_ann import NeuralNetwork, ReplayMemory

from .dqn import DQN


class DQN_ANN(DQN):
    def __init__(self, input_size=10, nb_actions=3, gamma=0.98):
        super().__init__(input_size=input_size, nb_actions=nb_actions, gamma=gamma)
        self.env.set_caption("Snake: Pytorch Artificial Neural Network")

        self.model = NeuralNetwork(self.input_size, nb_actions)
        self.memory = ReplayMemory(1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = F.smooth_l1_loss
        self.batch_size = 64

    def get_input_data(self):
        apple_pos = self.apple.get_position()
        input_data = [
            # self.snake.get_distance_to_north_wall(norm=2),
            # self.snake.get_distance_to_south_wall(norm=2),
            # self.snake.get_distance_to_east_wall(norm=2),
            # self.snake.get_distance_to_west_wall(norm=2),
            # self.snake.get_distance_to_target(
            #     self.snake.get_position(0), self.apple.get_position(), norm=2
            # ),
            # self.snake.get_distance_to_target(
            #     self.snake.get_position(0), self.snake.get_position(-1), norm=2
            # ),
            int(self.snake.is_clear_ahead()),
            int(self.snake.is_clear_left()),
            int(self.snake.is_clear_right()),
            int(self.snake.is_food_ahead(apple_pos)),
            int(self.snake.is_food_left(apple_pos)),
            int(self.snake.is_food_right(apple_pos)),
            int(self.snake.is_going_up()),
            int(self.snake.is_going_down()),
            int(self.snake.is_going_right()),
            int(self.snake.is_going_left()),
        ]
        return input_data

