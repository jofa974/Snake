import pygame
import math
from ui import WIDTH, HEIGHT, BASE_SPEED
from ui import BASE_SIZE, X_GRID, Y_GRID


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
        self.body_list = [head]
        for i in range(1, 3):
            body = SnakePart()
            body.rect.center = (int(WIDTH / 2) - i * BASE_SPEED,
                                int(HEIGHT / 2))
            self.body_list.append(body)

    def update(self):
        for i in range(len(self.body_list) - 1, 0, -1):
            self.body_list[i].rect = self.body_list[i - 1].rect.copy()

        self.body_list[0].rect.move_ip(self.speed)

    def detect_collisions(self):
        # Did this update cause us to hit a wall?
        if self.is_collision_wall():
            self.dead = True

        # Did this update cause us to hit a body part
        if self.is_collision_body():
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

    def get_position(self, idx):
        return int(self.body_list[idx].rect.centerx / BASE_SIZE), int(
            self.body_list[idx].rect.centery / BASE_SIZE)

    def get_neighbors(self, position, direction):
        """Get the coordinates of the nearest neighbors: in front, to the left and the right"""
        neighbors = {}
        if self.speed == (BASE_SPEED, 0):
            neighbors['front'] = (position[0]+1, position[1])
            neighbors['left'] = (position[0], position[1]-1)
            neighbors['right'] = (position[0], position[1]+1)
        if self.speed == (-BASE_SPEED, 0):
            neighbors['front'] = (position[0]-1, position[1])
            neighbors['left'] = (position[0], position[1]+1)
            neighbors['right'] = (position[0], position[1]-1)
        if self.speed == (0, BASE_SPEED):
            neighbors['front'] = (position[0], position[1]+1)
            neighbors['left'] = (position[0]+1, position[1])
            neighbors['right'] = (position[0]-1, position[1])
        if self.speed == (0, -BASE_SPEED):
            neighbors['front'] = (position[0], position[1]-1)
            neighbors['left'] = (position[0]-1, position[1])
            neighbors['right'] = (position[0]+1, position[1])
        return neighbors

    def get_next_key(self, way):
        if way == 'front':
            return 'front', self.speed
        if way == 'left':
            if self.speed == (BASE_SPEED, 0):
                return pygame.K_UP, (0, -BASE_SPEED)
            if self.speed == (-BASE_SPEED, 0):
                return pygame.K_DOWN, (0, BASE_SPEED)
            if self.speed == (0, BASE_SPEED):
                return pygame.K_LEFT, (-BASE_SPEED, 0)
            if self.speed == (0, -BASE_SPEED):
                return pygame.K_RIGHT, (BASE_SPEED, 0)
        if way == 'right':
            if self.speed == (BASE_SPEED, 0):
                return pygame.K_DOWN, (0, BASE_SPEED)
            if self.speed == (-BASE_SPEED, 0):
                return pygame.K_UP, (0, -BASE_SPEED)
            if self.speed == (0, BASE_SPEED):
                return pygame.K_RIGHT, (BASE_SPEED, 0)
            if self.speed == (0, -BASE_SPEED):
                return pygame.K_LEFT, (-BASE_SPEED, 0)

    def is_collision_wall(self):
        x_head, y_head = self.get_position(0)
        return x_head == 0 or x_head == X_GRID-1 \
            or y_head == 0 or y_head == Y_GRID-1

    def is_collision_body(self):
        x_head, y_head = self.get_position(0)
        for idx in range(1, len(self.body_list)):
            x_b, y_b = self.get_position(idx)
            if x_b == x_head and y_b == y_head:
                return True
        return False
