import pygame
from components.apple import Apple
from components.snake import Snake
from components.walls import Wall
import ui
import time
import argparse



def main(mode):
    pygame.init()
    screen = pygame.display.set_mode((ui.WIDTH, ui.HEIGHT))
    pygame.display.set_caption('Snake: {} mode'.format(mode))

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

    score = 0

    # TODO if human then play human option
    # TODO else define computer mode

    while not snake.dead:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake.dead = True
            if event.type == pygame.KEYDOWN and event.key in ui.CONTROLS:
                snake.change_direction(event.key)

        if snake.eat(apple):
            snake.grow()
            apple.new_random()
            score += 1

        snake.update(all_walls)

        # Draw Everything
        screen.fill(ui.BLACK)
        snake.draw(screen)
        apple.draw(screen)
        all_walls.draw(screen)
        pygame.display.flip()

        time.sleep(150.0 / 1000.0)

    pygame.quit()
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake game options')
    parser.add_argument('--mode',
                        default='human',
                        help='Define play mode (default: human)')
    args = parser.parse_args()
    score = main(args.mode)
    print("GAME OVER! Your score is {}".format(score))
