import pygame
from components.apple import Apple
from components.snake import Snake
from components.walls import Wall
import ui


def main():
    pygame.init()
    screen = pygame.display.set_mode((ui.WIDTH, ui.HEIGHT))
    pygame.display.set_caption('Snake')
    clock = pygame.time.Clock()

    all_sprites = pygame.sprite.Group()

    wall_left = Wall(0, 0, Wall.WALL_WIDTH, ui.HEIGHT)
    all_sprites.add(wall_left)
    wall_right = Wall(ui.WIDTH-Wall.WALL_WIDTH, 0, Wall.WALL_WIDTH, ui.HEIGHT)
    all_sprites.add(wall_right)
    wall_top = Wall(0, 0, ui.WIDTH, Wall.WALL_WIDTH)
    all_sprites.add(wall_top)
    wall_bottom = Wall(0, ui.HEIGHT-Wall.WALL_WIDTH, ui.WIDTH, Wall.WALL_WIDTH)
    all_sprites.add(wall_bottom)

    apple = Apple()
    all_sprites.add(apple)

    snake = Snake()
    all_sprites.add(snake)

    snake.init_walls((wall_left, wall_right, wall_top, wall_bottom))

    while not snake.dead:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake.dead = True
            if event.type == pygame.KEYDOWN and event.key in ui.CONTROLS:
                snake.change_direction(event.key)

        if snake.eat(apple):
            snake.grow()
            all_sprites.add(snake.body_list[-1])
            apple.new_random()

        all_sprites.update()

        # Draw Everything
        screen.fill(ui.BLACK)
        all_sprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
    quit()


if __name__ == '__main__':
    main()
