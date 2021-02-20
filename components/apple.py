import random

import pygame
from ui import BASE_SIZE, X_GRID, Y_GRID


class Apple(pygame.sprite.Sprite):
    """Apple to feed the snake"""

    def __init__(self, forbidden=[], xy=None):
        super().__init__()
        size = (BASE_SIZE, BASE_SIZE)
        colour = 155, 0, 0
        self.image = pygame.Surface(size)
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        if xy is not None:
            self.new(xy[0], xy[1], forbidden)
        else:
            self.new_random()

    def move(self, position):
        self.rect.center = position

    def new(self, x, y, forbidden=[]):
        if (x, y) in forbidden:
            self.new_random(forbidden)
        else:
            self.rect.center = (x * BASE_SIZE, y * BASE_SIZE)

    def new_random(self, forbidden=[]):
        x = random.randint(3, X_GRID - 2) * BASE_SIZE
        y = random.randint(3, X_GRID - 2) * BASE_SIZE
        while (x, y) in forbidden:
            x = random.randint(3, X_GRID - 2) * BASE_SIZE
            y = random.randint(3, X_GRID - 2) * BASE_SIZE
        self.rect.center = (x, y)

    def draw(self, surface):
        surface.blit(self.image, self.rect.center)

    def get_position(self):
        return self.rect.centerx, self.rect.centery
