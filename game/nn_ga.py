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
    def __init__(self, display, gen=0, nb=0):
        super().__init__()
        self.display = display
        self.nn = NeuralNetwork()
        self.gen = gen
        self.nb = nb

    def play(self, max_move):
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
            input_data = np.random.randn(6)
            self.nn.forward(input_data)

            # Take decision
            next_direction = self.nn.decide_direction()

            # Deduce which key to press based on next direction
            next_move = self.get_move_from_direction(next_direction)

            if next_move in ui.CONTROLS:
                self.snake.change_direction(next_move)

            prev_dist = self.get_distance_to_apple(norm=1)
            self.snake.move()
            new_dist = self.get_distance_to_apple(norm=1)

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
                moves_text = myfont.render('Moves: {}'.format(nb_moves),
                                           False, ui.WHITE)
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

        self.nn.dump_data(self.gen, self.nb, fitness)
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

    def get_distance_to_apple(self, norm=1):
        apple_pos = self.apple.get_position()
        snake_pos = self.snake.get_position(0)
        dist = sum([pow(abs(snake_pos[i]-apple_pos[i]), norm) for i in range(2)])
        dist = pow(dist, 1./norm)
        return dist
