import time

import pygame
import ui
from components.apple import Apple
from components.snake import Snake
from game.environment import Environment
from graphics.sprite import BasicSprite
from ui.controls import CONTROLS


class Human:
    def __init__(self):
        self.env = Environment()
        self.env.set_caption("Snake: Human mode")

    def play(self):
        self.snake = Snake()
        self.apple = Apple()
        score = 0

        while not self.snake.dead:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True
                if event.type == pygame.KEYDOWN and event.key in CONTROLS:
                    self.snake.change_direction(CONTROLS[event.key])
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.env.take_screenshot()

            self.snake.move()

            self.snake.detect_collisions()

            if self.snake.eat(self.apple.get_position()):
                self.snake.grow()
                self.snake.update()
                self.apple.new_random()
                score += 1

            score_text = "Score: {}".format(score)
            self.env.draw_everything(self.snake, self.apple, score_text)

            time.sleep(150.0 / 1000.0)

        final_text = "GAME OVER! Your score is {}".format(score)

        self.env.draw_everything(self.snake, self.apple, final_text)
        time.sleep(2)
