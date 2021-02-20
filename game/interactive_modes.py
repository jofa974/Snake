import brains
import pygame


def human():
    game = brains.human.Human()
    game.play()
    pygame.quit()


def random():
    for _ in range(3):
        game = brains.random.Random()
        game.play()
    pygame.quit()
