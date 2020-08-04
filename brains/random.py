import random
import time

import pygame

import game
import ui
from components.apple import Apple
from components.snake import Snake

from . import brain


class Random(brain):
    def __init__(self):
        super().__init__()
        pygame.display.set_caption("Snake: Random mode")

    def play(self):
        self.apple = Apple()
        self.snake = Snake()
        myfont = pygame.font.SysFont("Comic Sans MS", 30)
        score = 0

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

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.apple.new_random()
                score += 1

            self.snake.update()
            textsurface = myfont.render(
                "Score: {}".format(score), False, ui.WHITE
            )

            # Draw Everything
            self.screen.fill(ui.BLACK)
            self.screen.blit(textsurface, (ui.WIDTH + 50, 50))
            self.snake.draw(self.screen)
            self.apple.draw(self.screen)
            self.walls.draw(self.screen)
            pygame.display.flip()

            time.sleep(100.0 / 1000.0)
