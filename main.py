import pygame
from components.apple import Apple
from components.snake import Snake
import ui


def main():
    pygame.init()
    screen = pygame.display.set_mode((ui.WIDTH, ui.HEIGHT))
    pygame.display.set_caption('HELLO')
    clock = pygame.time.Clock()

    crashed = False

    # Create The Backgound
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((125, 125, 125))

    apple = Apple()
    snake = Snake()
    allsprites = pygame.sprite.RenderPlain((apple, snake))

    while not crashed:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        if snake.eat(apple):
            apple.new_random()

        allsprites.update()

        # Draw Everything
        screen.blit(background, (0, 0))
        allsprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
    quit()


if __name__ == '__main__':
    main()
