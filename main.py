#!/usr/bin/env python3

import pygame
from components.apple import Apple
from components.snake import Snake
from components.walls import Wall
import ui
import argparse
from game import human, random


def main(mode):
    pygame.init()
    screen = pygame.display.set_mode((2*ui.WIDTH, ui.HEIGHT))
    pygame.display.set_caption('Snake: {} mode'.format(mode))

    apple = Apple()
    snake = Snake()
    walls = pygame.sprite.Group()
    wall_left = Wall(0, 0, Wall.WALL_WIDTH, ui.HEIGHT)
    wall_right = Wall(ui.WIDTH - Wall.WALL_WIDTH, 0, Wall.WALL_WIDTH,
                      ui.HEIGHT)
    wall_top = Wall(0, 0, ui.WIDTH, Wall.WALL_WIDTH)
    wall_bottom = Wall(0, ui.HEIGHT - Wall.WALL_WIDTH, ui.WIDTH,
                       Wall.WALL_WIDTH)
    for wall in [wall_left, wall_right, wall_top, wall_bottom]:
        walls.add(wall)

    if mode == "human":
        score = human.play(screen, snake, apple, walls)
    elif mode == "random":
        score = random.play(screen, snake, apple, walls)
    else:
        raise NotImplementedError(
            "This game mode has not been implemented yet")

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake game options')
    parser.add_argument('-m',
                        '--mode',
                        default='human',
                        help='Define play mode (default: human)')
    args = parser.parse_args()
    score = main(args.mode)
    print("GAME OVER! Your score is {}".format(score))
