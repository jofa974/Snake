import pygame
import os

data_dir = os.path.join('/home/jonathan/Projects/PyGames/Snake/graphics')

BASE_SIZE = 40

WIDTH = BASE_SIZE * 20
HEIGHT = BASE_SIZE * 20

BLACK = (0, 0, 0)
BROWN = (210, 105, 30)
WHITE = (255, 255, 255)

CONTROLS = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]

BASE_SPEED = BASE_SIZE
