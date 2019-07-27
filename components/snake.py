import pygame
from ui.load_image import load_image
from .walls import Wall


class Snake(pygame.sprite.Sprite):
    """Snake player"""

    controls = [pygame.K_UP, pygame.K_DOWN,
                pygame.K_LEFT, pygame.K_RIGHT]

    base_speed = 5

    def __init__(self):
        super().__init__()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("snake_alpha.png", -1)
        self.speed = (Snake.base_speed, 0)
        self.wall = None

    def update(self):
        self.rect.midtop = tuple(
            sum(x) for x in zip(self.rect.midtop, self.speed))

    def eat(self, target):
        """returns true if the snake collides with the target"""
        hitbox = self.rect.inflate(-5, -5)
        return hitbox.colliderect(target.rect)

    def change_direction(self, key):
        if key == pygame.K_UP and self.speed[1] == 0:
            self.speed = (0, -Snake.base_speed)
        elif key == pygame.K_DOWN and self.speed[1] == 0:
            self.speed = (0, Snake.base_speed)
        elif key == pygame.K_LEFT and self.speed[0] == 0:
            self.speed = (-Snake.base_speed, 0)
        elif key == pygame.K_RIGHT and self.speed[0] == 0:
            self.speed = (Snake.base_speed, 0)
