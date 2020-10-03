import itertools
import os
import random
import sys
from abc import abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
import torch.nn.functional as F
import ui
from components.apple import Apple
from components.snake import Snake
from neural_net.pytorch_ann import ReplayMemory

from . import Brain


class DQN(Brain):
    def __init__(
        self, batch_size, gamma, memory_size, do_display=False, learning=True,
    ):
        super().__init__(do_display=do_display)
        self.model = None
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.optimizer = None
        self.steps = 0
        self.last_state = None
        self.last_action = 0
        self.last_reward = 0
        self.brain_file = "last_brain.pth"
        self.loss_history = []
        self.mean_reward_history = []
        self.last_state = None
        self.last_action = 0
        self.last_reward = 0
        self.brain_file = "last_brain.pth"
        self.learning = learning

        self.fig = plt.figure(figsize=[4, 3], dpi=100)

        if self.do_display:
            matplotlib.use("Agg")

    @abstractmethod
    def get_input_data(self):
        raise NotImplementedError

    def select_action(self, state, epsilon):
        probs = F.softmax(self.model(state), dim=1)
        if self.learning and random.random() < epsilon:
            action = probs.multinomial(num_samples=1)[0][0]
        else:
            action = probs.argmax()
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

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = (
            self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        targets = batch_reward + self.gamma * next_outputs
        loss = self.loss(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # if self.steps % 1 == 0:
        #     self.loss_history.append(loss.item())
        #     self.mean_reward_history.append(self.mean_reward())
        if loss.item() > 1.0e7:
            print("outputs {}".format(outputs))
            print("targets {}".format(targets))
        return loss.item()

    def mean_reward(self):
        return sum(self.reward_window)

    def save(self, filename=None):
        if filename is None:
            filename = self.brain_file
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )

    def save_best(self):
        self.save(filename="best_brain.pth")

    def load(self, filename=None):
        if filename is None:
            filename = self.brain_file
        print("Loading brain stored in {}".format(filename))
        if os.path.isfile(filename):
            # print("=> loading checkpoint ...")
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # print("done !")
        else:
            print("no checkpoint found ...")

    def load_best(self):
        self.load(filename="best_brain.pth")

    def plot_progress(self):
        plt.cla()
        plt.clf()
        ax1 = plt.subplot(1, 2, 1)
        ax1.scatter(
            np.arange(len(self.loss_history)), np.array(self.loss_history), s=20, c="r",
        )
        ax2 = plt.subplot(1, 2, 2)
        ax2.scatter(
            np.arange(len(self.mean_reward_history)),
            np.array(self.mean_reward_history),
            s=20,
            c="b",
        )

    def play(self, max_move=-1, init_training_data=None, epsilon=0):
        self.snake = Snake()

        forbidden_positions = self.snake.get_body_position_list()
        if init_training_data:
            training_data = itertools.cycle(init_training_data)
            self.apple = Apple(forbidden=forbidden_positions, xy=next(training_data))
        else:
            self.apple = Apple(forbidden=forbidden_positions)

        nb_moves = 0
        nb_apples = 0

        while (not self.snake.dead) and (nb_moves < max_move):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            nb_moves += 1
            self.steps += 1

            score_text = "Score: {}".format(nb_apples)
            if self.do_display:
                self.env.draw_everything(
                    score_text, [self.snake, self.apple], flip=False
                )
                self.plot_progress()
                self.env.make_surf_from_figure_on_canvas(self.fig)

            last_signal = self.get_input_data()

            next_action = self.update(
                self.last_reward, last_signal, nb_steps=nb_moves, epsilon=epsilon,
            )

            next_move = self.action2direction_key(next_action)
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
                self.last_reward = (prev_dist - new_dist) / (
                    np.sqrt(ui.X_GRID ** 2 + ui.Y_GRID ** 2)
                )
            else:
                self.last_reward = -0.7

            self.snake.detect_collisions()
            if self.snake.dead:
                self.last_reward = -1

            if self.snake.eat(self.apple):
                nb_apples += 1
                self.snake.grow()
                self.snake.update()
                forbidden_positions = self.snake.get_body_position_list()
                if init_training_data:
                    x, y = next(training_data)
                    self.apple.new(x, y, forbidden=forbidden_positions)
                else:
                    self.apple.new_random(forbidden=forbidden_positions)
                self.last_reward = 1
            # else:
            #     self.last_reward = nb_moves_wo_apple

        if self.learn and nb_moves < max_move:
            # Restart game
            self.play(
                max_move=max_move - nb_moves,
                init_training_data=init_training_data,
                epsilon=epsilon,
            )

        plt.close(self.fig)
        return nb_apples

    def update(self, reward, new_signal, nb_steps=-1, epsilon=-1.0):
        new_state = torch.Tensor(new_signal).unsqueeze(0)

        if self.learning:
            self.memory.push(
                (
                    self.last_state,
                    new_state,
                    torch.LongTensor([int(self.last_action)]),
                    torch.Tensor([self.last_reward]),
                )
            )

        action = self.select_action(new_state, epsilon)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > self.batch_size:
            del self.reward_window[0]
        return action
