import time

import pygame

import ui
from components.apple import Apple
from components.snake import Snake

from . import brain


class Human(brain):
    def __init__(self):
        super().__init__(do_display=True)
        self.env.set_caption("Snake: Human mode")

    def play(self):
        self.snake = Snake()
        self.apple = Apple()
        score = 0

        while not self.snake.dead:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True
                if event.type == pygame.KEYDOWN and event.key in ui.CONTROLS:
                    self.snake.change_direction(event.key)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.env.take_screenshot()

            self.snake.move()

            self.snake.detect_collisions()

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.snake.update()
                self.apple.new_random()
                score += 1

            score_text = "Score: {}".format(score)
            self.env.draw_everything(score_text, [self.snake, self.apple])

            time.sleep(150.0 / 1000.0)

        final_text = "GAME OVER! Your score is {}".format(score)

        self.env.draw_everything(final_text, [self.snake, self.apple])

        time.sleep(2)
