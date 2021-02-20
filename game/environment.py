import matplotlib
import pygame
from graphics.sprite import BasicSprite
from PIL import Image
from ui import BASE_SIZE, BLACK, HEIGHT, WHITE, WIDTH


class Environment:
    def __init__(self, do_display):
        pygame.init()
        w = 3 * WIDTH
        h = HEIGHT

        self.font = pygame.font.SysFont("Comic Sans MS", 30)

        self.walls = pygame.sprite.Group()
        # wall_left = BasicSprite(0, 0, (BASE_SIZE, HEIGHT), WHITE)
        # wall_top = BasicSprite(0, 0, (WIDTH, BASE_SIZE), WHITE)
        wall_bottom = BasicSprite(0, HEIGHT - BASE_SIZE, (WIDTH, BASE_SIZE), WHITE)
        wall_right = BasicSprite(WIDTH - BASE_SIZE, 0, (BASE_SIZE, HEIGHT), WHITE)
        # for wall in [wall_left, wall_right, wall_top, wall_bottom]:
        for wall in [wall_right, wall_bottom]:
            self.walls.add(wall)

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
