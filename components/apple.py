import pygame
import random
from ui import BASE_SIZE


class Apple(pygame.sprite.Sprite):
    """Apple to feed the snake"""

    def __init__(self):
        super().__init__()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        size = (BASE_SIZE, BASE_SIZE)
        colour = 155, 0, 0
        self.image = pygame.Surface(size)
        self.image.fill(colour)
        self.rect = self.image.get_rect()

        self.new_random()

    def move(self, position):
        self.rect.center = position

    def new_random(self):
        self.rect.center = (random.randint(3, 18) * BASE_SIZE,
                            random.randint(3, 18) * BASE_SIZE)

    def draw(self, surface):
        surface.blit(self.image, self.rect.center)
