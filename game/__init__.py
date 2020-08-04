import matplotlib
import matplotlib.backends.backend_agg as agg
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

    def make_surf_from_figure_on_canvas(self, fig):
        matplotlib.use("Agg")
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.screen.blit(surf, (6 * WIDTH / 5, HEIGHT / 5))
        pygame.display.flip()


def read_training_data():
    training_data = []
    with open("apple_position.in", "r") as f:
        for line in f.readlines():
            training_data.append((int(line.split()[0]), int(line.split()[1])))
    return training_data
