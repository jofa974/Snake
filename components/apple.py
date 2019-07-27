import os, pygame
from pygame.compat import geterror
from pygame.locals import *

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, '../graphics')


# functions to create our resources
def load_image(name, colorkey=None, rescale=True):
    fullname = os.path.join(data_dir, name)
    try:
        image = pygame.image.load(fullname)
        if rescale:
            image = pygame.transform.scale(image, (20, 20))
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(geterror()))
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()


class Apple(pygame.sprite.Sprite):
    """Apple to feed the snake"""
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.image, self.rect = load_image("apple_alpha.png", -1)
#        self.image = pygame.transform.scale(image, (20, 20))

    def move(self, position):
        self.rect.midtop = position
