import os
import random
from abc import abstractmethod

import pygame
import torch
import torch.nn.functional as F

from neural_net.pytorch_ann import ReplayMemory

from . import Brain


class DQN(Brain):
    def __init__(self, input_size, nb_actions=-1, gamma=-1.0, do_display=True):
        super().__init__(do_display=True)
        self.model = None
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(-1)
        self.optimizer = None
        self.last_state = None
        self.last_action = 0
        self.last_reward = 0
        self.brain_file = "last_brain.pth"

    @abstractmethod
    def get_input_data(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, reward, new_signal):
        raise NotImplementedError

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(3))
        else:
            probs = F.softmax(self.model(state), dim=1)
            action = probs.multinomial(num_samples=1)[0][0]
            return action.item()

    def action2direction_key(self, action):
        directions = ["forward", "left", "right"]
        if directions[action] == "forward":
            return pygame.K_SPACE
        elif directions[action] == "left":
            if self.snake.speed[0] > 0:
                return pygame.K_UP
            if self.snake.speed[0] < 0:
                return pygame.K_DOWN
            if self.snake.speed[1] > 0:
                return pygame.K_RIGHT
            if self.snake.speed[1] < 0:
                return pygame.K_LEFT
        elif directions[action] == "right":
            if self.snake.speed[0] > 0:
                return pygame.K_DOWN
            if self.snake.speed[0] < 0:
                return pygame.K_UP
            if self.snake.speed[1] > 0:
                return pygame.K_LEFT
            if self.snake.speed[1] < 0:
                return pygame.K_RIGHT

    @abstractmethod
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        return NotImplementedError

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
            # print("=> loading checkpoint ...")
            checkpoint = torch.load(self.brain_file)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # print("done !")
        else:
            print("no checkpoint found ...")
