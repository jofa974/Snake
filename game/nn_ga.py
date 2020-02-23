import itertools
import time

import matplotlib
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pygame

import game
import ui
from components.apple import Apple
from components.snake import Snake
from neural_net.neural_network import (NeuralNetwork,
                                       create_surf_from_figure_on_canvas)


class NN_GA(game.Game):
    def __init__(self, display, gen_id=(-1, -1), dna=None):
        super().__init__()
        self.display = display
        self.nn = NeuralNetwork(gen_id, dna, hidden_nb=[4])
        self.gen_id = gen_id

    def play(self, max_move, dump=False, learn=True):
        if learn:
            apple_x = []
            apple_y = []
            with open("apple_position.in", "r") as f:
                lines = f.readlines()
                for l in lines:
                    apple_x.append(int(l.split()[0]))
                    apple_y.append(int(l.split()[1]))
            apple_pos = itertools.cycle([(x, y) for x, y in zip(apple_x, apple_y)])
            self.apple = Apple(xy=next(apple_pos))
        else:
            self.apple = Apple()

        self.snake = Snake()
        score = 0
        fitness = 0
        nb_moves = 0

        if self.display:
            matplotlib.use("Agg")
            pygame.display.set_caption("Snake: Neural Network mode")
            myfont = pygame.font.SysFont("Comic Sans MS", 30)

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

            prev_dist = self.snake.get_distance_to_apple(
                self.snake.get_position(0), self.apple, norm=2
            )
            self.snake.move()
            new_dist = self.snake.get_distance_to_apple(
                self.snake.get_position(0), self.apple, norm=2
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
                if learn:
                    x, y = next(apple_pos)
                    self.apple.new(x, y)
                else:
                    self.apple.new_random()
                score += 1
                fitness += 20

            if self.display:
                score_text = myfont.render("Score: {}".format(score), False, ui.WHITE)
                fitness_text = myfont.render(
                    "Fitness: {}".format(fitness), False, ui.WHITE
                )
                moves_text = myfont.render(
                    "Moves: {}".format(nb_moves), False, ui.WHITE
                )
                # Draw Everything
                self.screen.fill(ui.BLACK)
                self.screen.blit(score_text, (ui.WIDTH + 50, 50))
                self.screen.blit(fitness_text, (ui.WIDTH + 150, 50))
                self.screen.blit(moves_text, (ui.WIDTH + 350, 50))
                self.walls.draw(self.screen)
                self.snake.draw(self.screen)
                self.apple.draw(self.screen)
                fig = self.nn.plot()
                surf = create_surf_from_figure_on_canvas(fig)
                self.screen.blit(surf, (6 * ui.WIDTH / 5, ui.HEIGHT / 5))
                pygame.display.flip()
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
        input_data = [
            self.snake.is_clear_ahead(),
            self.snake.is_clear_left(),
            self.snake.is_clear_right(),
            self.snake.is_food_ahead(self.apple),
            self.snake.is_food_left(self.apple),
            self.snake.is_food_right(self.apple),
        ]
        return input_data
