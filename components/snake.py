import pygame
import random
import copy
import math
from collections import deque
from ui.load_image import load_image
from ui import WIDTH, HEIGHT, BASE_SPEED
from .walls import Wall


def norm_speed(speed):
    return math.floor(speed / BASE_SPEED)


def speed_sign(speed):
    return int(math.copysign(1, norm_speed(speed)))


class SnakePart(pygame.sprite.Sprite):
    def __init__(self):
        self.image, self.rect = load_image("snake_alpha.png", -1)

    def draw(self, surface):
        surface.blit(self.image, self.rect.center)


class Snake(pygame.sprite.Sprite):
    """Snake player"""

    def __init__(self):
        super().__init__()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("snake_alpha.png", -1)
        self.rect.center = (int(WIDTH / 2), int(HEIGHT / 2))
        self.speed = (BASE_SPEED, 0)
        self.walls = None
        self.dead = False
        self.body_list = []  # list of sprites

    def init_walls(self, wall_list):
        self.walls = wall_list

    def update(self):
        for i in range(len(self.body_list) - 1, 0, -1):
            self.body_list[i].rect = self.body_list[i - 1].rect.copy()

        if self.body_list:
            self.body_list[0].rect = self.rect.copy()

        for i in range(len(self.body_list) - 1, 0, -1):
            print(i, self.body_list[i].rect.center)
        self.rect.move_ip(self.speed)

        # Did this update cause us to hit a wall?
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        if block_hit_list:
            self.dead = True

        # Did this update cause us to hit a body part (other than the closest to head)?
        body_hit_list = pygame.sprite.spritecollide(self, self.body_list,
                                                    False)
        if body_hit_list:
            print("I ATE MYSELF !")
            self.dead = True

    def eat(self, target):
        """returns true if the snake collides with the target"""
        hitbox = self.rect.inflate(-5, -5)
        return hitbox.colliderect(target.rect)

    def change_direction(self, key):
        if key == pygame.K_UP and self.speed[1] == 0:
            self.speed = (0, -BASE_SPEED)
        elif key == pygame.K_DOWN and self.speed[1] == 0:
            self.speed = (0, BASE_SPEED)
        elif key == pygame.K_LEFT and self.speed[0] == 0:
            self.speed = (-BASE_SPEED, 0)
        elif key == pygame.K_RIGHT and self.speed[0] == 0:
            self.speed = (BASE_SPEED, 0)

    def grow(self):
        self.body_list.append(SnakePart())

    def draw(self, surface):
        surface.blit(self.image, self.rect.center)
        for part in self.body_list:
            part.draw(surface)
