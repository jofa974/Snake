import itertools
import os
import random
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from components.apple import Apple
from components.snake import Snake
from neural_net.pytorch_ann import ReplayMemory


class DQN:
    def __init__(
        self,
        batch_size,
        gamma,
        memory_size,
        learning=True,
    ):
        self.model = torch.nn.Module()
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.optimizer = None
        self.steps = 0
        self.last_state = None
        self.last_action = 0
        self.last_reward = 0
        self.output_path = Path("output")
        self.brain_file = ""
        self.loss_history = []
        self.mean_reward_history = []
        self.list_of_rewards = []
        self.learning = learning
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        if directions[action] == "left":
            if self.snake.speed[0] > 0:
                return "up"
            if self.snake.speed[0] < 0:
                return "down"
            if self.snake.speed[1] > 0:
                return "right"
            if self.snake.speed[1] < 0:
                return "left"
        if directions[action] == "right":
            if self.snake.speed[0] > 0:
                return "down"
            if self.snake.speed[0] < 0:
                return "up"
            if self.snake.speed[1] > 0:
                return "left"
            if self.snake.speed[1] < 0:
                return "right"
        else:
            return "forward"

    def mean_reward(self):
        return np.mean(self.list_of_rewards)

    def cumulative_reward(self):
        return np.sum(self.list_of_rewards)

    def save(self):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            str(self.brain_file),
        )

    def load(self):
        print("Loading brain stored in {}".format(self.brain_file))
        if os.path.isfile(self.brain_file):
            # print("=> loading checkpoint ...")
            checkpoint = torch.load(self.brain_file)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # print("done !")
        else:
            print("no checkpoint found ...")

    def load_best(self):
        self.load(filename="best_brain.pth")

    def play(self, max_moves=-1, init_training_data=None, epsilon=0, env=None):
        self.snake = Snake()

        forbidden_positions = self.snake.get_body_position_list()
        if init_training_data:
            training_data = itertools.cycle(init_training_data)
            self.apple = Apple(forbidden=forbidden_positions, xy=next(training_data))
        else:
            self.apple = Apple(forbidden=forbidden_positions)

        nb_moves = 0
        nb_apples = 0

        if env:
            env.set_caption(self.caption)

        while (not self.snake.dead) and (nb_moves < max_moves):

            nb_moves += 1
            self.steps += 1

            if env:
                score_text = "Score: {}".format(nb_apples)
                env.draw_everything(self.snake, self.apple, score_text, flip=True)
                time.sleep(0.1)

            last_signal = self.get_input_data()

            next_action = self.update(
                self.last_reward,
                last_signal,
                epsilon=epsilon,
            )

            next_move = self.action2direction_key(next_action)
            self.snake.change_direction(next_move)

            prev_dist = self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            )
            self.snake.move()
            new_dist = self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            )

            # if new_dist < prev_dist:
            #     # self.last_reward = (prev_dist - new_dist) / (
            #     #     np.sqrt(ui.X_GRID ** 2 + ui.Y_GRID ** 2)
            #     # )
            #     self.last_reward = 0.5
            # else:
            #     self.last_reward = -0.7

            self.last_reward = 0

            self.snake.detect_collisions()
            if self.snake.dead:
                self.last_reward = -1

            if self.snake.eat(self.apple.get_position()):
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

            self.list_of_rewards.append(self.last_reward)

        if self.learning and nb_moves < max_moves:
            # Restart game and try to finish episode
            self.play(
                max_moves=max_moves - nb_moves,
                init_training_data=init_training_data,
                epsilon=epsilon,
            )

        return nb_apples

    def update(self, reward, new_signal, epsilon=-1.0):
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

        action = self.select_action(new_state.to(self.device), epsilon)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > self.batch_size:
            del self.reward_window[0]
        return action

    def learn(self, epochs):
        loss = 0.0
        for epoch in range(epochs):

            (
                batch_state,
                batch_next_state,
                batch_action,
                batch_reward,
            ) = self.memory.sample(self.batch_size)

            batch_state = batch_state.to(self.device)
            batch_next_state = batch_next_state.to(self.device)
            batch_action = batch_action.to(self.device)
            batch_reward = batch_reward.to(self.device)

            outputs = (
                self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            )
            next_outputs = self.model(batch_next_state).detach().max(1)[0]
            targets = batch_reward + self.gamma * next_outputs
            loss += self.loss(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if loss.item() > 1.0e7:
            print("outputs {}".format(outputs))
            print("targets {}".format(targets))
        return loss.item() / epochs
