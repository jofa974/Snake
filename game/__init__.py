import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pygame
from PIL import Image

from components.walls import Wall
from ui import BLACK, HEIGHT, WHITE, WIDTH


class Environment:
    def __init__(self, do_display):
        pygame.init()
        w = 3 * WIDTH
        h = HEIGHT

        self.font = pygame.font.SysFont("Comic Sans MS", 30)

        self.walls = pygame.sprite.Group()
        wall_left = Wall(0, 0, Wall.WALL_WIDTH, HEIGHT)
        wall_right = Wall(WIDTH - Wall.WALL_WIDTH, 0, Wall.WALL_WIDTH, HEIGHT)
        wall_top = Wall(0, 0, WIDTH, Wall.WALL_WIDTH)
        wall_bottom = Wall(0, HEIGHT - Wall.WALL_WIDTH, WIDTH, Wall.WALL_WIDTH)
        for wall in [wall_left, wall_right, wall_top, wall_bottom]:
            self.walls.add(wall)

        # self.do_display = do_display
        self.screen = None
        if do_display:
            self.screen = pygame.display.set_mode((w, h))

    def draw_everything(self, text=None, sprites=None, flip=True):
        if text:
            textsurface = self.font.render(text, False, WHITE)
        self.screen.fill(BLACK)
        self.screen.blit(textsurface, (WIDTH + 50, 50))
        for s in sprites:
            s.draw(self.screen)
        self.walls.draw(self.screen)
        if flip:
            pygame.display.flip()

    def set_caption(self, caption="Snake"):
        pygame.display.set_caption(caption)

    # TODO: refactor this
    def make_surf_from_figure_on_canvas(self, fig):
        import matplotlib.backends.backend_agg as agg

        matplotlib.use("Agg")
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.screen.blit(surf, (WIDTH + 75, 75))
        pygame.display.flip()

    def take_screenshot(self):
        rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
        sub = self.screen.subsurface(rect)
        pygame.image.save(sub, "screenshot.png")

    def load_screenshot(self):
        with Image.open("screenshot.png") as img:
            return img


def read_training_data():
    training_data = []
    with open("apple_position.in", "r") as f:
        for line in f.readlines():
            training_data.append((int(line.split()[0]), int(line.split()[1])))
    return training_data
