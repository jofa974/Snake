import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=6, nb_actions=3):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, nb_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = F.relu(self.fc3(x))
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
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, dim=0)), samples)
