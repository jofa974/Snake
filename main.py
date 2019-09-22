#!/usr/bin/env python3

import argparse
from game import human, random


def main(mode):

    if mode == "human":
        game = human.Human()
    elif mode == "random":
        game = random.Random()
    else:
        raise NotImplementedError(
            "This game mode has not been implemented yet")

    game.play()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake game options')
    parser.add_argument('-m',
                        '--mode',
                        default='human',
                        help='Define play mode (default: human)')
    args = parser.parse_args()
    main(args.mode)
