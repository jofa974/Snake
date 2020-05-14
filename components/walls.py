import pygame
from ui import WHITE, BASE_SIZE


class Wall(pygame.sprite.Sprite):
    WALL_WIDTH = BASE_SIZE
    """ Wall the player can (but should not) run into. """

    def __init__(self, x, y, width, height):
        """ Constructor for the wall that the player can run into. """
        # Call the parent's constructor
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(WHITE)

        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
