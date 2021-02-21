import math
import random

from ui import BASE_SIZE, X_GRID, Y_GRID


class Apple:
    """Apple to feed the snake"""

    def __init__(self, forbidden=[], xy=None):
        if xy is not None:
            self.new(xy[0], xy[1], forbidden)
        else:
            self.new_random()

    def new(self, x, y, forbidden=[]):
        if (x, y) in forbidden:
            self.new_random(forbidden)
        else:
            self.x = x
            self.y = y

    def new_random(self, forbidden=[]):
        x = random.randint(3, X_GRID - 2)
        y = random.randint(3, X_GRID - 2)
        while (x, y) in forbidden:
            x = random.randint(3, X_GRID - 2)
            y = random.randint(3, X_GRID - 2)
        self.x = x
        self.y = y

    def get_position(self):
        return self.x, self.y
