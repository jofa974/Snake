from pathlib import Path

import torch
import torch.optim as optim
import ui
from neural_net.pytorch_ann import NeuralNetwork, ReplayMemory
from torch import nn

from .dqn import DQN


class DQN_ANN(DQN):
    def __init__(
        self,
        batch_size=128,
        gamma=0.9,
        memory_size=200,
        learning=True,
    ):
        super().__init__(
            batch_size=batch_size,
            gamma=gamma,
            memory_size=memory_size,
            learning=learning,
        )
        self.caption = "Snake: Pytorch Artificial Neural Network"

        input_size = 18
        nb_actions = 3
        self.model = NeuralNetwork(input_size, nb_actions)
        self.model.to(self.device)
        self.memory = ReplayMemory(memory_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.batch_size = batch_size
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.brain_file = self.output_path / "dqn_ann/last_brain.pth"

        Path.mkdir(self.brain_file.parent, exist_ok=True)

    def get_input_data(self):
        apple_pos = self.apple.get_position()
        input_data = [
            # Normalized Distances to walls
            self.snake.get_distance_to_north_wall(norm=2) / (ui.X_GRID * ui.Y_GRID),
            self.snake.get_distance_to_south_wall(norm=2) / (ui.X_GRID * ui.Y_GRID),
            self.snake.get_distance_to_east_wall(norm=2) / (ui.X_GRID * ui.Y_GRID),
            self.snake.get_distance_to_west_wall(norm=2) / (ui.X_GRID * ui.Y_GRID),
            # Normalized Distance to apple
            self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            )
            / (ui.X_GRID * ui.Y_GRID),
            # Indicate collision on next move
            int(self.snake.is_clear_ahead()),
            int(self.snake.is_clear_left()),
            int(self.snake.is_clear_right()),
            # Hint at apple's direction
            int(self.snake.is_food_ahead(apple_pos)),
            int(self.snake.is_food_left(apple_pos)),
            int(self.snake.is_food_right(apple_pos)),
            # Snake direction
            int(self.snake.is_going_up()),
            int(self.snake.is_going_down()),
            int(self.snake.is_going_right()),
            int(self.snake.is_going_left()),
            # Snake normalized length
            len(self.snake.body_list) / (ui.X_GRID * ui.Y_GRID),
            # Apple normalized x coordinate
            apple_pos[0] / ui.X_GRID,
            # Apple normalized x coordinate
            apple_pos[1] / ui.Y_GRID,
        ]
        return input_data
