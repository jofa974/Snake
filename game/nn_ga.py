import time
import pygame
import ui
import game
import numpy as np
from components.apple import Apple
from components.snake import Snake
from neural_net.neural_network import NeuralNetwork
import matplotlib.pyplot as plt


class NN_GA(game.Game):
    def __init__(self, display, gen_id=(-1, -1)):
        super().__init__()
        self.display = display
        self.nn = NeuralNetwork(gen_id)
        self.gen_id = gen_id

    def play(self, max_move, dump=False):
        self.apple = Apple()
        self.snake = Snake()
        score = 0
        fitness = 0
        nb_moves = 0

        if self.display:
            pygame.display.set_caption('Snake: Neural Network mode')
            myfont = pygame.font.SysFont('Comic Sans MS', 30)
            fig = plt.figure(figsize=[5, 5], dpi=100)

        while not self.snake.dead and nb_moves < max_move:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True

            # Feed the NN with input data
            # input_data = np.random.randn(6)
            input_data = self.get_input_data()
            self.nn.forward(input_data)

            # Take decision
            next_direction = self.nn.decide_direction()

            # Deduce which key to press based on next direction
            next_move = self.get_move_from_direction(next_direction)

            if next_move in ui.CONTROLS:
                self.snake.change_direction(next_move)

            prev_dist = self.snake.get_distance_to_apple(self.snake.get_position(0), self.apple, norm=1)
            self.snake.move()
            new_dist = self.snake.get_distance_to_apple(self.snake.get_position(0), self.apple, norm=1)

            if new_dist < prev_dist:
                fitness += 2
            else:
                fitness -= 3

            self.snake.detect_collisions()

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.snake.update()
                self.apple.new_random()
                score += 1
                fitness += 10

            if self.display:
                score_text = myfont.render('Score: {}'.format(score), False,
                                           ui.WHITE)
                fitness_text = myfont.render('Fitness: {}'.format(fitness),
                                             False, ui.WHITE)
                moves_text = myfont.render('Moves: {}'.format(nb_moves), False,
                                           ui.WHITE)
                # Draw Everything
                self.screen.fill(ui.BLACK)
                self.screen.blit(score_text, (ui.WIDTH + 50, 50))
                self.screen.blit(fitness_text, (ui.WIDTH + 150, 50))
                self.screen.blit(moves_text, (ui.WIDTH + 350, 50))
                self.walls.draw(self.screen)
                self.snake.draw(self.screen)
                self.apple.draw(self.screen)
                surf = self.nn.plot(fig)
                self.screen.blit(surf, (6 * ui.WIDTH / 5, ui.HEIGHT / 5))
                pygame.display.flip()
                time.sleep(1.0 / 1000.0)

            nb_moves += 1

        if dump:
            self.nn.dump_data(self.gen_id, fitness)

        return score

    def get_move_from_direction(self, direction):
        if direction == 'forward':
            return 'forward'
        if direction == 'left':
            if self.snake.speed[0] > 0:
                return pygame.K_UP
            if self.snake.speed[0] < 0:
                return pygame.K_DOWN
            if self.snake.speed[1] > 0:
                return pygame.K_RIGHT
            if self.snake.speed[1] < 0:
                return pygame.K_LEFT
        if direction == 'right':
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
            self.snake.is_food_right(self.apple)
        ]
        return input_data
