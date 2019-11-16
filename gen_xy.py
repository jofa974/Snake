#!/usr/bin/env python3

import random
from ui import X_GRID, Y_GRID, BASE_SIZE

if __name__ == "__main__":
    with open("apple_position.in", "w") as f:
        for _ in range(10000):
            x, y = (random.randint(3, X_GRID - 2),
                    random.randint(3, Y_GRID - 2))
            f.write('{} {} \n'.format(x, y))
