import pygame

from components.walls import Wall
from ui import BLACK, HEIGHT, WHITE, WIDTH


class Environment:
    def __init__(self, do_display=False):
        pygame.init()
        w = max(1600, 2 * WIDTH)
        h = max(800, HEIGHT)

        self.font = pygame.font.SysFont("Comic Sans MS", 30)

        self.walls = pygame.sprite.Group()
        wall_left = Wall(0, 0, Wall.WALL_WIDTH, HEIGHT)
        wall_right = Wall(WIDTH - Wall.WALL_WIDTH, 0, Wall.WALL_WIDTH, HEIGHT)
        wall_top = Wall(0, 0, WIDTH, Wall.WALL_WIDTH)
        wall_bottom = Wall(0, HEIGHT - Wall.WALL_WIDTH, WIDTH, Wall.WALL_WIDTH)
        for wall in [wall_left, wall_right, wall_top, wall_bottom]:
            self.walls.add(wall)

        self.do_display = do_display
        self.screen = None
        if do_display:
            self.screen = pygame.display.set_mode((w, h))

    def draw_everything(self, text=None, sprites=None):
        if text:
            textsurface = self.font.render(text, False, WHITE)
        self.screen.fill(BLACK)
        self.screen.blit(textsurface, (WIDTH + 50, 50))
        for s in sprites:
            s.draw(self.screen)
        self.walls.draw(self.screen)
        pygame.display.flip()

    def set_caption(self, caption="Snake"):
        pygame.display.set_caption(caption)


def read_training_data():
    training_data = []
    with open("apple_position.in", "r") as f:
        for line in f.readlines():
            training_data.append((int(line.split()[0]), int(line.split()[1])))
    return training_data
