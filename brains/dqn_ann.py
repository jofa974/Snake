import torch.optim as optim
from torch import nn

from neural_net.pytorch_ann import NeuralNetwork, ReplayMemory

from .dqn import DQN


class DQN_ANN(DQN):
    def __init__(
        self, input_size=16, nb_actions=3, gamma=0.9, do_display=False, learning=True
    ):
        super().__init__(
            input_size=input_size,
            nb_actions=nb_actions,
            gamma=gamma,
            do_display=do_display,
            learning=learning,
        )
        self.env.set_caption("Snake: Pytorch Artificial Neural Network")

        self.model = NeuralNetwork(self.input_size, nb_actions)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()
        self.batch_size = 32

    def get_input_data(self):
        apple_pos = self.apple.get_position()
        input_data = [
            self.snake.get_distance_to_north_wall(norm=2),
            self.snake.get_distance_to_south_wall(norm=2),
            self.snake.get_distance_to_east_wall(norm=2),
            self.snake.get_distance_to_west_wall(norm=2),
            self.snake.get_distance_to_target(
                self.snake.get_position(0), self.apple.get_position(), norm=2
            ),
            self.snake.get_distance_to_target(
                self.snake.get_position(0), self.snake.get_position(-1), norm=2
            ),
            int(self.snake.is_clear_ahead()),
            int(self.snake.is_clear_left()),
            int(self.snake.is_clear_right()),
            int(self.snake.is_food_ahead(apple_pos)),
            int(self.snake.is_food_left(apple_pos)),
            int(self.snake.is_food_right(apple_pos)),
            int(self.snake.is_going_up()),
            int(self.snake.is_going_down()),
            int(self.snake.is_going_right()),
            int(self.snake.is_going_left()),
        ]
        return input_data
