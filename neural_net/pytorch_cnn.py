import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, image_dim, nb_actions=3):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.convolution1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=2, stride=2
        )
        # self.convolution2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
        self.fc1 = nn.Linear(self.count_neurons(image_dim), nb_actions)
        # self.fc2 = nn.Linear(32, nb_actions)
        # self.fc3 = nn.Linear(32, nb_actions)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.avg_pool2d(self.convolution1(x), 2, 2))
        # x = F.relu(F.max_pool2d(self.convolution2(x), 2, 2))
        # x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, state):
        x = F.relu(F.avg_pool2d(self.convolution1(state), 2, 2))
        # x = F.relu(F.max_pool2d(self.convolution2(x), 2, 2))
        # x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # Flattening
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_values = x
        # q_values = self.fc2(x)
        # q_values = self.fc3(x)
        return q_values


# class ReplayMemory:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []

#     def push(self, event):
#         self.memory.append(event)
#         if len(self.memory) > self.capacity:
#             del self.memory[0]

#     def sample(self, batch_size):
#         # Exclude first one because initial state is not relevant
#         idxs = random.choices(range(1, len(self.memory)), k=batch_size)
#         batch_state = []
#         batch_next_state = []
#         batch_action = []
#         batch_reward = []
#         for idx in idxs:
#             batch_state.append(self.memory[idx][0])
#             batch_next_state.append(self.memory[idx][1])
#             batch_action.append(self.memory[idx][2])
#             batch_reward.append(self.memory[idx][3])
#         return (
#             torch.cat(batch_state),
#             torch.cat(batch_next_state),
#             torch.cat(batch_action),
#             torch.cat(batch_reward),
#         )
