import pygame
from components.apple import Apple

width = 800
height = 600


def main():
    pygame.init()
    game_display = pygame.display.set_mode((width, height))
    pygame.display.set_caption('HELLO')
    clock = pygame.time.Clock()

    crashed = False

    while not crashed:
        game_display.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                apple = Apple(pos)

        if Apple.instance:
            apple.show(game_display)
        pygame.display.update()

        clock.tick(60)

    pygame.quit()
    quit()


if __name__ == '__main__':
    main()
