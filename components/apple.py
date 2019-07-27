import sys
import pygame
sys.path.append("../ui")
from ui.load_image import load_image


class Apple(pygame.sprite.Sprite):
    """Apple to feed the snake"""
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("apple_alpha.png", -1)

    def move(self, position):
        self.rect.midtop = position
