import pygame
import random
from ui.load_image import load_image
from ui import WIDTH, HEIGHT, BASE_SIZE
from .walls import Wall


class Apple(pygame.sprite.Sprite):
    """Apple to feed the snake"""

    def __init__(self):
        super().__init__()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("apple_alpha.png", -1)
        self.new_random()

    def move(self, position):
        self.rect.center = position

    def new_random(self):
        # self.rect.center = (random.randint(2 * Wall.WALL_WIDTH,
        #                                    WIDTH - 2 * Wall.WALL_WIDTH),
        #                     random.randint(2 * Wall.WALL_WIDTH,
        #                                    HEIGHT - 2 * Wall.WALL_WIDTH))
        self.rect.center = (random.randint(3, 18) * BASE_SIZE,
                            random.randint(3, 18) * BASE_SIZE)

    def draw(self, surface):
        surface.blit(self.image, self.rect.center)
