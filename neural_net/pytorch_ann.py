import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, nb_actions):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, nb_actions),
        )

    def forward(self, state):
        q_values = self.model(state)
        return q_values


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory[1:], batch_size))
        return map(lambda x: Variable(torch.cat(x, dim=0)), samples)
