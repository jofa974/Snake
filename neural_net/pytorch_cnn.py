import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, image_dim=(1, 80, 80), nb_actions):
        super(NeuralNetwork, self).__init__()
        self.convolution1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5
        )
        self.convolution2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3
        )
        self.convolution3 = nn.Conv2d(
            in_channels=32, out_channels=, kernel_size=2
        )
        self.fc1 = nn.Linear(self.count_neurons(image_dim), 40)
        self.fc2 = nn.Linear(40, nb_actions)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, state):
        x = F.relu(F.max_pool2d(self.convolution1(state), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # Flattening
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
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
