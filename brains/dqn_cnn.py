import itertools
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
from components.apple import Apple
from components.snake import Snake
from neural_net.pytorch_cnn import ConvolutionalNeuralNetwork, ReplayMemory

from . import Reward
from .dqn import DQN


class DQN_CNN(DQN):
    def __init__(self, nb_actions=-1, gamma=-1.0):
        self.input_size = (1, ui.X_GRID, ui.Y_GRID)
        super().__init__(
            input_size=self.input_size, nb_actions=nb_actions, gamma=gamma
        )
        self.env.set_caption("Snake: Pytorch Convolutional Neural Network")

        self.model = ConvolutionalNeuralNetwork(self.input_size, nb_actions)
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()
        self.last_state = torch.Tensor(self.input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.brain_file = "last_brain.pth"
        self.loss_history = []
        # self.fig = plt.figure(figsize=[3, 3], dpi=100)
        matplotlib.use("Agg")

    def play(self, max_move=-1, training_data=None, epsilon=0):
        if training_data:
            training_data = itertools.cycle(training_data)
            self.apple = Apple(xy=next(training_data))
        else:
            self.apple = Apple()

        self.snake = Snake()
        nb_moves = 0
        nb_apples = 0

        fig = plt.figure(figsize=[3, 3], dpi=100)

        while not self.snake.dead:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True

            nb_moves += 1

            score_text = "Score: {}".format(nb_apples)
            self.env.draw_everything(
                score_text, [self.snake, self.apple], flip=False
            )
            self.plot_progress()
            self.env.make_surf_from_figure_on_canvas(fig)

            last_signal = self.get_input_data()
            next_move = self.update(
                self.last_reward,
                last_signal,
                batch_size=100,
                nb_steps=nb_moves,
                epsilon=epsilon,
            )

            if next_move in ui.CONTROLS:
                self.snake.change_direction(next_move)

            prev_dist = self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            )
            self.snake.move()
            new_dist = self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            )

            if new_dist < prev_dist:
                self.last_reward = Reward.CLOSER
            else:
                self.last_reward = Reward.FURTHER

            self.snake.detect_collisions()
            if self.snake.dead:
                self.last_reward = Reward.DEAD

            if self.snake.eat(self.apple):
                nb_moves = 0
                nb_apples += 1
                self.snake.grow()
                self.snake.update()
                if training_data:
                    x, y = next(training_data)
                    self.apple.new(x, y)
                else:
                    self.apple.new_random()
                self.last_reward = Reward.EAT
            # else:
            #     self.last_reward = nb_moves_wo_apple

            if nb_moves == max_move:
                break

            # time.sleep(0.01)

        print("Final score: {}".format(nb_apples))
        plt.close(fig)
        return nb_apples

    def plot_progress(self):
        plt.cla()
        plt.clf()
        ax = plt.gca()
        ax.scatter(
            np.arange(len(self.loss_history)),
            np.array(self.loss_history),
            s=20,
            c="r",
        )
        # ax.set_aspect("equal", adjustable="box")
        # self.make_surf_from_figure_on_canvas(fig)

    def get_input_data(self):
        self.env.take_screenshot()
        with Image.open("screenshot.png") as img:
            img_conv = img.convert("L")
            # img_conv = np.transpose(img_conv, (2, 0, 1))
            arr = np.asarray(img_conv)
            arr = np.array([arr[:: ui.BASE_SIZE, :: ui.BASE_SIZE] / 255.0])
        return arr

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).max(1)[0]
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        targets = batch_reward + self.gamma * next_outputs
        loss = self.loss(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update(
        self, reward, new_signal, batch_size=-1, nb_steps=-1, epsilon=-1.0
    ):
        new_state = torch.Tensor(new_signal).unsqueeze(0)
        self.memory.push(
            (
                self.last_state,
                new_state,
                torch.LongTensor([int(self.last_action)]),
                torch.Tensor([self.last_reward]),
            )
        )
        action = self.select_action(new_state, epsilon)

        if nb_steps % 1 == 0:
            if (len(self.memory.memory)) > batch_size:
                (
                    batch_state,
                    batch_next_state,
                    batch_action,
                    batch_reward,
                ) = self.memory.sample(batch_size)
                last_error = self.learn(
                    batch_state, batch_next_state, batch_reward, batch_action
                )
                self.loss_history.append(last_error)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
