import itertools
import time

import matplotlib
import matplotlib.pyplot as plt
import pygame

import ui
from components.apple import Apple
from components.snake import Snake
from neural_net.artificial_neural_network import ANN

from . import Brain


def play_individual(individual, gen_nb, game_id, training_data):
    game = NN_GA(do_display=False, gen_id=(gen_nb, game_id), dna=individual)
    score, fitness = game.play(
        max_move=1000, dump=True, training_data=training_data
    )
    return fitness


class NN_GA(Brain):
    """
    Class that will play the game with a neural network optimized
    using a genetic algorithm.
    """

    def __init__(self, do_display, gen_id=(-1, -1), dna=None):
        super().__init__(do_display=do_display)
        self.nn = ANN(gen_id, dna, hidden_nb=[4])
        self.gen_id = gen_id

    def play(self, max_move, dump=False, training_data=None):
        if training_data:
            training_data = itertools.cycle(training_data)
            self.apple = Apple(xy=next(training_data))
        else:
            self.apple = Apple()

        self.snake = Snake()
        score = 0
        fitness = 0
        nb_moves = 0

        if self.do_display:
            matplotlib.use("Agg")
            self.env.set_caption(
                "Snake: Custom Neural Network optimized with a Genetic Algorithm"
            )
            fig = plt.figure(figsize=[3, 3], dpi=100)

        while not self.snake.dead and nb_moves < max_move:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True

            # Feed the NN with input data
            input_data = self.get_input_data()
            self.nn.forward(input_data)

            # Take decision
            next_direction = self.nn.decide_direction()

            # Deduce which key to press based on next direction
            next_move = self.get_move_from_direction(next_direction)
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
                fitness += 2
            else:
                fitness -= 3

            self.snake.detect_collisions()
            if self.snake.dead:
                fitness -= 10

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.snake.update()
                if training_data:
                    x, y = next(training_data)
                    self.apple.new(x, y)
                else:
                    self.apple.new_random()
                score += 1
                fitness += 20

            if self.do_display:
                score_text = "Score: {}".format(score)
                self.env.draw_everything(
                    score_text, [self.snake, self.apple], flip=False
                )
                self.nn.plot(fig)
                self.env.make_surf_from_figure_on_canvas(fig)
                time.sleep(0.01 / 1000.0)

            nb_moves += 1
        if dump:
            self.nn.dump_data(self.gen_id, fitness)

        return score, fitness

    def get_move_from_direction(self, direction):
        if direction == "forward":
            return "forward"
        if direction == "left":
            if self.snake.speed[0] > 0:
                return pygame.K_UP
            if self.snake.speed[0] < 0:
                return pygame.K_DOWN
            if self.snake.speed[1] > 0:
                return pygame.K_RIGHT
            if self.snake.speed[1] < 0:
                return pygame.K_LEFT
        if direction == "right":
            if self.snake.speed[0] > 0:
                return pygame.K_DOWN
            if self.snake.speed[0] < 0:
                return pygame.K_UP
            if self.snake.speed[1] > 0:
                return pygame.K_LEFT
            if self.snake.speed[1] < 0:
                return pygame.K_RIGHT

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
