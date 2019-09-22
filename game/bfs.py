import time
import pygame
import ui
import game
import itertools
from collections import deque


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

            self.make_grid_map()

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
            pygame.display.flip()
            time.sleep(150.0 / 1000.0)

        pygame.quit()

    def make_grid_map(self):
        self.grid_map = [[1 for _ in range(ui.X_GRID)]
                         for _ in range(ui.Y_GRID)]
        self.grid_map[0][:] = [0 for _ in range(ui.X_GRID)]
        self.grid_map[-1][:] = [0 for _ in range(ui.X_GRID)]
        for i in range(ui.Y_GRID):
            self.grid_map[i][0] = 0
            self.grid_map[i][-1] = 0
        for idx in range(len(self.snake.body_list)):
            pos = self.snake.get_position(idx)
            self.grid_map[pos[1]][pos[0]] = 0
        pos = self.apple.get_position()
        self.grid_map[pos[1]][pos[0]] = 2

    def BFS(self):
        height = len(map)
        width = len(map[0])
        start = self.snake.get_position()
        end = self.apple.get_position()
        queue = deque([[start]])
        visited = set((start))
        while queue:
            path = queue.popleft()
            nextp = path[-1]
            if nextp == end:
                return path
            neighbors = (nextp[0]+1, nextp[1]), \
                        (nextp[0]-1, nextp[1]), \
                        (nextp[0], nextp[1]+1), \
                        (nextp[0], nextp[1]-1)
            for neighbor in neighbors:
                xn = neighbor[0]
                yn = neighbor[1]
                ingrid = 0 <= xn < height and 0 <= yn < width
                isvisited = (neighbor[0], neighbor[1]) in visited
                if ingrid and (not isvisited) and map[xn][yn]:
                    queue.append(path + [(neighbor[0], neighbor[1])])
                    visited.add((neighbor[0], neighbor[1]))

    def walk_from_path(path):
        walk = []
        if path:
            if len(path) > 1:
                current = path[0]
                for place in path[1:]:
                    if place[0] - current[0] == 1:
                        walk.append('right')
                    if place[0] - current[0] == -1:
                        walk.append('left')
                    if place[1] - current[1] == 1:
                        walk.append('down')
                    if place[1] - current[1] == -1:
                        walk.append('up')
                    current = place
        return walk

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
