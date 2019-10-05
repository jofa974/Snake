import pygame
import time
import ui
from game import Game
from components.apple import Apple
from components.snake import Snake


class Human(Game):
    def __init__(self):
        super().__init__()
        pygame.display.set_caption('Snake: Human mode')

    def play(self):
        self.apple = Apple()
        self.snake = Snake()
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        score = 0

        while not self.snake.dead:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True
                if event.type == pygame.KEYDOWN and event.key in ui.CONTROLS:
                    self.snake.change_direction(event.key)

            self.snake.move()

            self.snake.detect_collisions()

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.snake.update()
                self.apple.new_random()
                score += 1

            textsurface = myfont.render('Score: {}'.format(score), False,
                                        ui.WHITE)

            # Draw Everything
            self.screen.fill(ui.BLACK)
            self.screen.blit(textsurface, (ui.WIDTH + 50, 50))
            self.snake.draw(self.screen)
            self.apple.draw(self.screen)
            self.walls.draw(self.screen)
            pygame.display.flip()

            time.sleep(150.0 / 1000.0)

        self.screen.fill(ui.BLACK)
        textsurface = myfont.render(
            'GAME OVER! Your score is {}'.format(score), False, ui.WHITE)
        self.screen.blit(textsurface, (ui.WIDTH + 50, 50))
        self.snake.draw(self.screen)
        self.apple.draw(self.screen)
        self.walls.draw(self.screen)
        pygame.display.flip()

        time.sleep(2)
        pygame.quit()
