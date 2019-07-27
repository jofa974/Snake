import pygame


class Apple():
    # Singleton pattern
    class __Apple():
        def __init__(self, position):
            self.position = position
            self.image = pygame.image.load(
                "/home/jonathan/Projects/PyGames/Snake/graphics/apple_simple.png"
            )

    instance = None

    def __init__(self, position):
        if not Apple.instance:
            Apple.instance = Apple.__Apple(position)
        else:
            Apple.instance.position = position

    def show(self, game_display):
        game_display.blit(self.instance.image, self.instance.position)
