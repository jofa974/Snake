import pygame
from components.apple import Apple
from components.snake import Snake
from components.walls import Wall
import ui


def main():
    pygame.init()
    screen = pygame.display.set_mode((ui.WIDTH, ui.HEIGHT))
    pygame.display.set_caption('HELLO')
    clock = pygame.time.Clock()

    crashed = False

    wall = Wall(0, 0, ui.WIDTH, ui.HEIGHT)
    apple = Apple()
    snake = Snake()
    allsprites = pygame.sprite.RenderPlain((wall, apple, snake))

    while not crashed:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN and event.key in Snake.controls:
                snake.change_direction(event.key)

        if snake.eat(apple):
            apple.new_random()

        allsprites.update()

        # Draw Everything
        screen.fill(ui.BLACK)
        allsprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
    quit()


if __name__ == '__main__':
    main()
