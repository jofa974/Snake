import os
import pygame
from pygame.compat import geterror
from pygame.locals import RLEACCEL
from . import data_dir


# functions to create our resources
def load_image(name, colorkey=None, rescale=True):
    fullname = os.path.join(data_dir, name)
    try:
        image = pygame.image.load(fullname)
        if rescale:
            image = pygame.transform.scale(image, (44, 44))
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(geterror()))
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()
