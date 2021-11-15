import pygame
import ui


class BasicSprite(pygame.sprite.Sprite):
    def __init__(self, x, y, size=(ui.BASE_SIZE, ui.BASE_SIZE), colour=(0, 155, 0)):
        super().__init__()
        self.image = pygame.Surface(size)
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        self.rect.center = x, y

    def draw(self, surface):
        surface.blit(self.image, self.rect.center)

