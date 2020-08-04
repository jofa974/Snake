import itertools
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import pygame
import torch
import torch.nn.functional as F
import torch.optim as optim

import game
import neural_net.pytorch_cnn as cnn
import ui
from components.apple import Apple
from components.snake import Snake
from neural_net.pytorch_ann import NeuralNetwork, ReplayMemory


class DQN_conv(DQN):
    def __init__(self, gamma=0.9):
        input_size = ui.X_GRID * ui.Y_GRID
        super().__init__(input_size=input_size, nb_actions=3, gamma=gamma)
