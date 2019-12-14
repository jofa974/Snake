import pygame

import ui
from components.walls import Wall


class Game:
    def __init__(self):
        pygame.init()
        w = max(1600, 2 * ui.WIDTH)
        h = max(800, ui.HEIGHT)
        self.screen = pygame.display.set_mode((w, h))
        self.walls = pygame.sprite.Group()
        wall_left = Wall(0, 0, Wall.WALL_WIDTH, ui.HEIGHT)
        wall_right = Wall(ui.WIDTH - Wall.WALL_WIDTH, 0, Wall.WALL_WIDTH, ui.HEIGHT)
        wall_top = Wall(0, 0, ui.WIDTH, Wall.WALL_WIDTH)
        wall_bottom = Wall(0, ui.HEIGHT - Wall.WALL_WIDTH, ui.WIDTH, Wall.WALL_WIDTH)
        for wall in [wall_left, wall_right, wall_top, wall_bottom]:
            self.walls.add(wall)

    def play(self):
        pass
