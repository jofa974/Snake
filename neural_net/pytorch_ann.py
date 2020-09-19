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
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, nb_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
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
