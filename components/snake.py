import pygame
import random
import copy
import math
from collections import deque
from ui.load_image import load_image
from ui import WIDTH, HEIGHT, BASE_SPEED
from .walls import Wall


def norm_speed(speed):
    return math.floor(speed/BASE_SPEED)


def speed_sign(speed):
    return int(math.copysign(1, norm_speed(speed)))


class SnakePart(pygame.sprite.Sprite):
    def __init__(self, previous):
        super().__init__()
        self.image, self.rect = load_image("snake_alpha.png", -1)
        self.speed = previous.speed
        self.rect = previous.rect.copy()
        self.rect.center = (
            self.rect.centerx - speed_sign(self.speed[0]) * self.rect.w,
            self.rect.centery - speed_sign(self.speed[1]) * self.rect.h)

        self.moves = previous.moves_list_for_next
        self.moves_list_for_next = deque(maxlen=BASE_SPEED)

    def update(self):
        self.moves_list_for_next.append(copy.deepcopy(self.rect.center))
        new_position = self.moves.popleft()
        self.rect.center = new_position


class Snake(pygame.sprite.Sprite):
    """Snake player"""

    def __init__(self):
        super().__init__()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("snake_alpha.png", -1)
        self.rect.center = (random.randint(Wall.WALL_WIDTH,
                                           WIDTH - Wall.WALL_WIDTH),
                            random.randint(Wall.WALL_WIDTH,
                                           HEIGHT - Wall.WALL_WIDTH))
        self.speed = (BASE_SPEED, 0)
        self.walls = None
        self.dead = False
        self.body_list = []
        self.moves_list_for_next = deque(maxlen=BASE_SPEED)

    def init_walls(self, wall_list):
        self.walls = wall_list

    def update(self):
        # self.previous_position = copy.deepcopy(self.rect.center)
        self.moves_list_for_next.append(self.rect.center)
        self.rect.move_ip(self.speed)

        # Did this update cause us to hit a wall?
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        if block_hit_list:
            self.dead = True
        # Did this update cause us to hit a body part?
        body_hit_list = pygame.sprite.spritecollide(self, self.body_list[1:],
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
        if self.body_list:
            new_part = SnakePart(previous=self.body_list[-1])
        else:
            new_part = SnakePart(previous=self)
        self.body_list.append(new_part)
