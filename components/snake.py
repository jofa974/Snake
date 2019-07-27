import pygame
from ui.load_image import load_image

BLUE = (0, 0, 255)


class Snake(pygame.sprite.Sprite):
    """Snake player"""

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("snake_alpha.png", -1)
        self.speed = (1, 0)

    def update(self):
        self.rect.midtop = tuple(
            sum(x) for x in zip(self.rect.midtop, self.speed))
