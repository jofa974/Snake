import pygame
import random
from ui.load_image import load_image
from ui import WIDTH, HEIGHT
from .walls import WALL_WIDTH


class Apple(pygame.sprite.Sprite):
    """Apple to feed the snake"""

    def __init__(self):
        super().__init__()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("apple_alpha.png", -1)

    def move(self, position):
        self.rect.midtop = position

    def new_random(self):
        self.rect.midtop = (random.randint(WALL_WIDTH, WIDTH-WALL_WIDTH),
                            random.randint(WALL_WIDTH, HEIGHT-WALL_WIDTH))
