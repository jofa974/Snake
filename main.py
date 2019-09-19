import pygame
from components.apple import Apple
from components.snake import Snake
from components.walls import Wall
import ui
import time


def main():
    pygame.init()
    screen = pygame.display.set_mode((ui.WIDTH, ui.HEIGHT))
    pygame.display.set_caption('Snake')

    all_walls = pygame.sprite.Group()

    wall_left = Wall(0, 0, Wall.WALL_WIDTH, ui.HEIGHT)
    all_walls.add(wall_left)
    wall_right = Wall(ui.WIDTH - Wall.WALL_WIDTH, 0, Wall.WALL_WIDTH,
                      ui.HEIGHT)
    all_walls.add(wall_right)
    wall_top = Wall(0, 0, ui.WIDTH, Wall.WALL_WIDTH)
    all_walls.add(wall_top)
    wall_bottom = Wall(0, ui.HEIGHT - Wall.WALL_WIDTH, ui.WIDTH,
                       Wall.WALL_WIDTH)
    all_walls.add(wall_bottom)

    apple = Apple()

    snake = Snake()

    snake.init_walls((wall_left, wall_right, wall_top, wall_bottom))

    while not snake.dead:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake.dead = True
            if event.type == pygame.KEYDOWN and event.key in ui.CONTROLS:
                snake.change_direction(event.key)

        if snake.eat(apple):
            snake.grow()
            apple.new_random()

        snake.update()

        # Draw Everything
        screen.fill(ui.BLACK)
        snake.draw(screen)
        apple.draw(screen)
        all_walls.draw(screen)
        pygame.display.flip()

        time.sleep(100.0 / 1000.0)

    pygame.quit()
    quit()


if __name__ == '__main__':
    main()
