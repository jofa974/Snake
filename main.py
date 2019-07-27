import pygame
from components.apple import Apple

width = 800
height = 600


def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('HELLO')
    clock = pygame.time.Clock()

    crashed = False

    # Create The Backgound
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    apple = Apple()
    allsprites = pygame.sprite.RenderPlain((apple))

    while not crashed:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                apple.move(pos)

        allsprites.update()
        # Draw Everything
        screen.blit(background, (0, 0))
        allsprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
    quit()


if __name__ == '__main__':
    main()
