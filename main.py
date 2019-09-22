#!/usr/bin/env python3

import argparse
from game import human, random, bfs


def main(mode):

    if mode == "human":
        game = human.Human()
        game.play()
    elif mode == "random":
        for _ in range(3):
            game = random.Random()
            game.play()
    elif mode == "BFS":
        for _ in range(1):
            game = bfs.BFS()
            game.play()
        game.show_stats()
    else:
        raise NotImplementedError(
            "This game mode has not been implemented yet")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake game options')
    parser.add_argument('-m',
                        '--mode',
                        default='human',
                        help='Define play mode (default: human)')
    args = parser.parse_args()
    main(args.mode)
