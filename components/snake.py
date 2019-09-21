import pygame
import math
from ui import WIDTH, HEIGHT, BASE_SPEED
from ui import BASE_SIZE


def norm_speed(speed):
    return math.floor(speed / BASE_SPEED)


def speed_sign(speed):
    return int(math.copysign(1, norm_speed(speed)))


class SnakePart(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        size = (BASE_SIZE, BASE_SIZE)
        colour = 0, 155, 0
        self.image = pygame.Surface(size)
        self.image.fill(colour)
        self.rect = self.image.get_rect()

    def draw(self, surface):
        surface.blit(self.image, self.rect.center)


class Snake():
    """Snake player"""

    def __init__(self):
        self.speed = (BASE_SPEED, 0)
        self.dead = False
        head = SnakePart()
        head.rect.center = (int(WIDTH / 2), int(HEIGHT / 2))
        self.body_list = [head, SnakePart(), SnakePart()]  # list of sprites

    def update(self, walls):
        for i in range(len(self.body_list) - 1, 0, -1):
            self.body_list[i].rect = self.body_list[i - 1].rect.copy()

        self.body_list[0].rect.move_ip(self.speed)

        # Did this update cause us to hit a wall?
        block_hit_list = pygame.sprite.spritecollide(self.body_list[0], walls,
                                                     False)
        if block_hit_list:
            self.dead = True

        # Did this update cause us to hit a body part
        body_hit_list = pygame.sprite.spritecollide(self.body_list[0],
                                                    self.body_list[1:], False)
        if body_hit_list:
            print("I ATE MYSELF !")
            self.dead = True

    def eat(self, target):
        """returns true if the snake collides with the target"""
        hitbox = self.body_list[0].rect
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
        for part in self.body_list:
            part.draw(surface)
