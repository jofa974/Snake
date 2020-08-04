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
import ui
from components.apple import Apple
from components.snake import Snake
from neural_net.pytorch_ann import NeuralNetwork, ReplayMemory

from . import brain


class DQN(brain):
    def __init__(self, input_size=10, nb_actions=3, gamma=0.9):
        super().__init__(do_display=True)
        self.model = None
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(100)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.brain_file = "last_brain.pth"

    def play(self, max_move, training_data=None):
        pass

    def get_input_data(self):
        pass

    def update(self, reward, new_signal):
        pass

    def select_action(self, state):
        temperature = 50
        probs = F.softmax(self.model(state) * temperature)
        # action = probs.multinomial(num_samples=1)
        directions = ["forward", "left", "right"]
        idx_max = torch.argmax(probs)
        if directions[idx_max] == "forward":
            action = pygame.K_SPACE
        if directions[idx_max] == "left":
            if self.snake.speed[0] > 0:
                action = pygame.K_UP
            if self.snake.speed[0] < 0:
                action = pygame.K_DOWN
            if self.snake.speed[1] > 0:
                action = pygame.K_RIGHT
            if self.snake.speed[1] < 0:
                action = pygame.K_LEFT
        if directions[idx_max] == "right":
            if self.snake.speed[0] > 0:
                action = pygame.K_DOWN
            if self.snake.speed[0] < 0:
                action = pygame.K_UP
            if self.snake.speed[1] > 0:
                action = pygame.K_LEFT
            if self.snake.speed[1] < 0:
                action = pygame.K_RIGHT
        return action

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).max(1)[0]
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        targets = batch_reward + self.gamma * next_outputs
        td_loss = F.smooth_l1_loss(outputs, targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)

    def save(self):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.brain_file,
        )

    def load(self):
        if os.path.isfile(self.brain_file):
            print("=> loading checkpoint ...")
            checkpoint = torch.load(self.brain_file)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("done !")
        else:
            print("no checkpoint found ...")
