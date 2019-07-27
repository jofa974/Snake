import pygame


class Apple():
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.image = pygame.image.load(
            "/home/jonathan/Projects/PyGames/Snake/graphics/apple_simple.png")

    def show(self, game_display):
        game_display.blit(self.image, (self.x, self.y))
