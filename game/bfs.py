import time
import pygame
import ui
import game
import itertools


class BFS(game.Game):
    def __init__(self):
        super().__init__()
        pygame.display.set_caption('Snake: BFS mode')
        self.moves = []
        self.grid = []
        for y in range(1, ui.Y_GRID - 1):
            grid = []
            for x in range(1, ui.X_GRID - 1):
                grid.append(
                    pygame.Rect(x * ui.BASE_SIZE, y * ui.BASE_SIZE,
                                ui.BASE_SIZE, ui.BASE_SIZE))
            self.grid.append(grid)

    def play(self):
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        score = 0
        new_apple = True

        while not self.snake.dead:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True

            if new_apple:
                self.get_moves_bfs()
                new_apple = False

            for move in self.moves:
                if move in ui.CONTROLS:
                    self.snake.change_direction(move)

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.apple.new_random()
                score += 1
                new_apple = True

            self.snake.update(self.walls)
            textsurface = myfont.render('Score: {}'.format(score), False,
                                        ui.WHITE)

            # Draw Everything
            self.screen.fill(ui.BLACK)
            self.screen.blit(textsurface, (ui.WIDTH + 50, 50))
            self.walls.draw(self.screen)
            self.draw_grid()
            self.snake.draw(self.screen)
            self.apple.draw(self.screen)

            print([int(i/ui.BASE_SIZE) for i in self.apple.rect.center])

            pygame.display.flip()
            time.sleep(50)
            time.sleep(150.0 / 1000.0)

        pygame.quit()

    def bfs_paths(self, start, goal):
        queue = [(start, [start])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next in self.grid[vertex] - set(path):
                if next == goal:
                    yield path + [next]
                else:
                    queue.append((next, path + [next]))

    def get_moves_bfs(self):
        pass

    def draw_grid(self):
        for rect in itertools.chain.from_iterable(self.grid):
            pygame.draw.rect(self.screen, ui.BROWN, rect, 3)
