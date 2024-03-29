import time

import pygame
from components.apple import Apple
from components.snake import Snake
from ui.controls import CONTROLS


class Human:
    def __init__(self):
        self.snake = Snake()
        self.apple = Apple()
        self.score = 0

    def play(self, env):
        env.set_caption("Snake: Human mode")
        while not self.snake.dead:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True
                if event.type == pygame.KEYDOWN and event.key in CONTROLS:
                    self.snake.change_direction(CONTROLS[event.key])

            self.snake.move()

            self.snake.detect_collisions()

            if self.snake.eat(self.apple.get_position()):
                self.snake.grow()
                self.snake.update()
                self.apple.new_random()
                self.score += 1

            score_text = "Score: {}".format(self.score)
            env.draw_everything(self.snake, self.apple, score_text)

            time.sleep(150.0 / 1000.0)

        final_text = "GAME OVER! Your score is {}".format(self.score)
        env.draw_everything(self.snake, self.apple, final_text)
        time.sleep(2)
