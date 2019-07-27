import pygame


class Apple():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.image.load(
            "/home/jonathan/Projects/PyGames/Snake/graphics/apple_simple.png")

    def show(self, game_display):
        game_display.blit(self.image, (self.x, self.y))
