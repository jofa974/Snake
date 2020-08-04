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

from .dqn import DQN


class DQN_ANN(DQN):
    def __init__(self, input_size=10, nb_actions=3, gamma=0.9):
        super().__init__(do_display=True)
        self.env.set_caption("Snake: Pytorch Artificial Neural Network")

        self.model = NeuralNetwork(input_size, nb_actions)
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(100)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.brain_file = "last_brain.pth"

    def play(self, max_move, training_data=None):
        if training_data:
            training_data = itertools.cycle(training_data)
            self.apple = Apple(xy=next(training_data))
        else:
            self.apple = Apple()

        self.snake = Snake()
        nb_moves_wo_apple = 0
        nb_apples = 0
        scores = []

        while not self.snake.dead:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True

            last_signal = self.get_input_data()
            next_move = self.update(self.last_reward, last_signal)
            scores.append(self.score())

            if next_move in ui.CONTROLS:
                self.snake.change_direction(next_move)

            prev_dist = self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            )
            self.snake.move()
            new_dist = self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            )

            nb_moves_wo_apple += 1

            if new_dist < prev_dist:
                self.last_reward = 8
            else:
                self.last_reward = -10

            self.snake.detect_collisions()
            if self.snake.dead:
                self.last_reward = -50

            if self.snake.eat(self.apple):
                nb_moves_wo_apple = 0
                nb_apples += 1
                self.snake.grow()
                self.snake.update()
                if training_data:
                    x, y = next(training_data)
                    self.apple.new(x, y)
                else:
                    self.apple.new_random()
                self.last_reward = 1000
            else:
                self.last_reward -= nb_moves_wo_apple
            score_text = "Score: {}".format(nb_apples)
            self.env.draw_everything(score_text, [self.snake, self.apple])
            time.sleep(0.01)

        print("Final score: {}".format(nb_apples))

    def get_input_data(self):
        apple_pos = self.apple.get_position()
        input_data = [
            # self.snake.get_distance_to_north_wall(norm=2),
            # self.snake.get_distance_to_south_wall(norm=2),
            # self.snake.get_distance_to_east_wall(norm=2),
            # self.snake.get_distance_to_west_wall(norm=2),
            # self.snake.get_distance_to_target(
            #     self.snake.get_position(0), self.apple.get_position(), norm=2
            # ),
            # self.snake.get_distance_to_target(
            #     self.snake.get_position(0), self.snake.get_position(-1), norm=2
            # ),
            self.snake.is_clear_ahead(),
            self.snake.is_clear_left(),
            self.snake.is_clear_right(),
            self.snake.is_food_ahead(apple_pos),
            self.snake.is_food_left(apple_pos),
            self.snake.is_food_right(apple_pos),
            self.snake.is_going_up(),
            self.snake.is_going_down(),
            self.snake.is_going_right(),
            self.snake.is_going_left(),
        ]
        return input_data

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).max(1)[0]
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        targets = batch_reward + self.gamma * next_outputs
        td_loss = F.smooth_l1_loss(outputs, targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (
                self.last_state,
                new_state,
                torch.LongTensor([int(self.last_action)]),
                torch.Tensor([self.last_reward]),
            )
        )
        action = self.select_action(new_state)

        if (len(self.memory.memory)) > 50:
            (
                batch_state,
                batch_next_state,
                batch_action,
                batch_reward,
            ) = self.memory.sample(50)
            self.learn(
                batch_state, batch_next_state, batch_reward, batch_action
            )

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
