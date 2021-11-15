import matplotlib
import pygame
from graphics.sprite import BasicSprite
from ui import BASE_SIZE, BLACK, HEIGHT, WHITE, WIDTH


class Environment:
    def __init__(self):
        pygame.init()
        w = 3 * WIDTH
        h = HEIGHT

        self.font = pygame.font.SysFont("Comic Sans MS", 30)

        self.walls = pygame.sprite.Group()
        wall_left = BasicSprite(BASE_SIZE / 2, HEIGHT / 2, (BASE_SIZE, HEIGHT), WHITE)
        wall_top = BasicSprite(WIDTH / 2, BASE_SIZE / 2, (WIDTH, BASE_SIZE), WHITE)
        wall_bottom = BasicSprite(
            WIDTH / 2,
            HEIGHT - BASE_SIZE / 2,
            (WIDTH, BASE_SIZE),
            WHITE,
        )
        wall_right = BasicSprite(
            WIDTH - BASE_SIZE / 2,
            HEIGHT / 2,
            (BASE_SIZE, HEIGHT),
            WHITE,
        )
        for wall in [wall_left, wall_right, wall_top, wall_bottom]:
            self.walls.add(wall)

        self.screen = pygame.display.set_mode((w, h))

    def draw_everything(self, snake, apple, text=None, flip=True):
        if text:
            textsurface = self.font.render(text, False, WHITE)
        self.screen.fill(BLACK)
        self.screen.blit(textsurface, (WIDTH + 50, 50))

        for coords in snake.get_body_position_list():
            body = BasicSprite(
                coords[0] * BASE_SIZE,
                coords[1] * BASE_SIZE,
            )
            body.draw(self.screen)

        apple_coords = apple.get_position()
        apple_sprite = BasicSprite(
            apple_coords[0] * BASE_SIZE,
            apple_coords[1] * BASE_SIZE,
            colour=(155, 0, 0),
        )
        apple_sprite.draw(self.screen)

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
