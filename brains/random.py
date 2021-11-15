import random
import time

import pygame
from components.apple import Apple
from components.snake import Snake


class Random:
    def __init__(self):
        self.score = 0

    def play(self, env):
        self.apple = Apple()
        self.snake = Snake()

        env.set_caption("Snake: Random mode")

        while not self.snake.dead:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True

            if random.randint(0, 1):
                if self.snake.speed[0] == 0:
                    key = random.choice([pygame.K_LEFT, pygame.K_RIGHT])
                else:
                    key = random.choice([pygame.K_UP, pygame.K_DOWN])
                self.snake.change_direction(key)

            self.snake.move()

            self.snake.detect_collisions()

            if self.snake.eat(self.apple.get_position()):
                self.snake.grow()
                self.snake.update()
                self.apple.new_random()
                self.score += 1

            score_text = "Score: {}".format(self.score)
            env.draw_everything(self.snake, self.apple, score_text)

            time.sleep(100.0 / 1000.0)

        final_text = "GAME OVER! The random score is {}".format(self.score)
        env.draw_everything(self.snake, self.apple, final_text)
        time.sleep(2)
