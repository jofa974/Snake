import itertools
import time
from collections import deque

import pygame

import ui
from components.apple import Apple
from components.snake import Snake

from . import brain


class BFS(brain):
    def __init__(self, do_display):
        super().__init__(do_display=do_display)
        self.moves = []
        self.grid = []
        for y in range(ui.Y_GRID):
            grid = []
            for x in range(ui.X_GRID):
                grid.append(
                    pygame.Rect(
                        x * ui.BASE_SIZE,
                        y * ui.BASE_SIZE,
                        ui.BASE_SIZE,
                        ui.BASE_SIZE,
                    )
                )
            self.grid.append(grid)

    def play(self):
        self.apple = Apple()
        self.snake = Snake()
        score = 0
        if self.do_display:
            self.env.set_caption("Snake: BFS mode")

        while not self.snake.dead:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.snake.dead = True

            if not self.moves:
                self.make_grid_map()
                path = self.BFS()
                self.get_moves_from_path(path)

            if self.moves:
                next_move = self.moves.pop(0)
            else:
                next_move = "forward"

            if next_move in ui.CONTROLS:
                self.snake.change_direction(next_move)

            self.snake.move()

            self.snake.detect_collisions()

            if self.snake.eat(self.apple):
                self.snake.grow()
                self.snake.update()
                self.apple.new_random()
                score += 1

            if self.do_display:
                score_text = "Score: {}".format(score)
                self.env.draw_everything(score_text, [self.snake, self.apple])
                time.sleep(50.0 / 1000.0)

        return score

    def make_grid_map(self):
        self.grid_map = [
            [True for _ in range(ui.X_GRID)] for _ in range(ui.Y_GRID)
        ]
        self.grid_map[0][:] = [False for _ in range(ui.X_GRID)]
        self.grid_map[-1][:] = [False for _ in range(ui.X_GRID)]
        for i in range(ui.Y_GRID):
            self.grid_map[i][0] = False
            self.grid_map[i][-1] = False

    def BFS(self):
        height = ui.Y_GRID
        width = ui.X_GRID
        start = self.snake.get_position(0)
        end = self.apple.get_position()
        queue = deque([[start]])
        visited = set((start))
        for i in range(1, len(self.snake.body_list)):
            visited.add((self.snake.get_position(i)))
        while queue:
            path = queue.popleft()
            nextp = path[-1]
            if nextp == end:
                return path
            neighbors = (
                (nextp[0] + 1, nextp[1]),
                (nextp[0] - 1, nextp[1]),
                (nextp[0], nextp[1] + 1),
                (nextp[0], nextp[1] - 1),
            )
            for neighbor in neighbors:
                xn = neighbor[0]
                yn = neighbor[1]
                ingrid = 0 <= xn < width and 0 <= yn < height
                isvisited = (neighbor[0], neighbor[1]) in visited
                if ingrid and (not isvisited) and self.grid_map[yn][xn]:
                    queue.append(path + [(neighbor[0], neighbor[1])])
                    visited.add((neighbor[0], neighbor[1]))

    def get_moves_from_path(self, path):
        direction = self.snake.speed
        self.moves = []
        if path and len(path) > 1:
            current = path[0]
            for place in path[1:]:
                if place[0] - current[0] == 1:
                    if direction[1] == 0:
                        self.moves.append("forward")
                    else:
                        self.moves.append(pygame.K_RIGHT)
                        direction = (ui.BASE_SPEED, 0)
                if place[0] - current[0] == -1:
                    if direction[1] == 0:
                        self.moves.append("forward")
                    else:
                        self.moves.append(pygame.K_LEFT)
                        direction = (-ui.BASE_SPEED, 0)
                if place[1] - current[1] == 1:
                    if direction[0] == 0:
                        self.moves.append("forward")
                    else:
                        self.moves.append(pygame.K_DOWN)
                        direction = (0, ui.BASE_SPEED)
                if place[1] - current[1] == -1:
                    if direction[0] == 0:
                        self.moves.append("forward")
                    else:
                        self.moves.append(pygame.K_UP)
                        direction = (0, -ui.BASE_SPEED)
                current = place

    def draw_grid(self):
        for rect in itertools.chain.from_iterable(self.grid):
            pygame.draw.rect(self.screen, ui.BROWN, rect, 3)
