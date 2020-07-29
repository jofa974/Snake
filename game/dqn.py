import itertools
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


class DQN(game.Game):
    def __init__(self, input_size=6, nb_actions=3, gamma=0.9):
        super().__init__(do_display=True)
        self.model = NeuralNetwork(input_size, nb_actions)
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
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
        nb_moves = 0
        scores = []

        matplotlib.use("Agg")
        pygame.display.set_caption("Snake: Neural Network mode")
        myfont = pygame.font.SysFont("Comic Sans MS", 30)
        fig = plt.figure(figsize=[5, 5], dpi=100)

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

            if new_dist < prev_dist:
                self.last_reward = 2
            else:
                self.last_reward = -3

            self.snake.detect_collisions()
            if self.snake.dead:
                self.last_reward = -10

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.snake.update()
                if training_data:
                    x, y = next(training_data)
                    self.apple.new(x, y)
                else:
                    self.apple.new_random()
                self.last_reward = 20

            # Draw Everything
            self.screen.fill(ui.BLACK)
            self.walls.draw(self.screen)
            self.snake.draw(self.screen)
            self.apple.draw(self.screen)
            pygame.display.flip()
            time.sleep(0.01 / 1000.0)

    def get_input_data(self):
        apple_pos = self.apple.get_position()
        input_data = [
            self.snake.is_clear_ahead(),
            self.snake.is_clear_left(),
            self.snake.is_clear_right(),
            self.snake.is_food_ahead(apple_pos),
            self.snake.is_food_left(apple_pos),
            self.snake.is_food_right(apple_pos),
        ]
        return input_data

    def select_action(self, state):
        temperature = 75
        probs = F.softmax(self.model(state) * temperature)
        # action = probs.multinomial(num_samples=1)
        directions = ["forward", "left", "right"]
        idx_max = torch.argmax(probs)
        if directions[idx_max] == "forward":
            action = pygame.K_SPACE
        if directions[idx_max] == "left":
            if self.snake.speed[0] > 0:
                action = pygame.K_UP
            if self.snake.speed[0] < 0:
                action = pygame.K_DOWN
            if self.snake.speed[1] > 0:
                action = pygame.K_RIGHT
            if self.snake.speed[1] < 0:
                action = pygame.K_LEFT
        if directions[idx_max] == "right":
            if self.snake.speed[0] > 0:
                action = pygame.K_DOWN
            if self.snake.speed[0] < 0:
                action = pygame.K_UP
            if self.snake.speed[1] > 0:
                action = pygame.K_LEFT
            if self.snake.speed[1] < 0:
                action = pygame.K_RIGHT
        return action

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = (
            self.model(batch_state)
            .gather(1, batch_action.unsqueeze(1))
            .squeeze(1)
        )
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

        if (len(self.memory.memory)) > 100:
            (
                batch_state,
                batch_next_state,
                batch_action,
                batch_reward,
            ) = self.memory.sample(100)
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
            print("=> loading checkpoint ...")
            checkpoint = torch.load(self.brain_file)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("done !")
        else:
            print("no checkpoint found ...")
