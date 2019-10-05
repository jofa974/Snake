#!/usr/bin/env python3

import argparse
from game import human, random, bfs, nn_ga
from stats.stats import show_stats
import numpy as np
import pygame


def main(args):
    if args.human:
        game = human.Human()
        game.play()
        pygame.quit()
    elif args.random:
        for _ in range(3):
            game = random.Random()
            game.play()
            pygame.quit()
    elif args.bfs:
        if args.interactive:
            game = bfs.BFS(display=True)
            nb_games = 1
            game.play()
        else:
            nb_games = 100
            all_score = np.zeros(nb_games)
            game = bfs.BFS(display=False)
            for i in range(nb_games):
                score = game.play()
                all_score[i] = score
            show_stats(all_score)
        pygame.quit()
    elif args.nnga:
        if args.interactive:
            game = nn_ga.NN_GA(display=True)
            nb_games = 1
            game.play(100)
        else:
            nb_games = 100
            all_score = np.zeros(nb_games)
            game = nn_ga.NN_GA(display=False)
            for i in range(nb_games):
                score = game.play()
                all_score[i] = score
            show_stats(all_score)
        pygame.quit()
    else:
        raise NotImplementedError(
            "This game mode has not been implemented yet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake game options')
    parser.add_argument('--human', action="store_true", help='Human play mode')
    parser.add_argument('--bfs', action="store_true", help='BFS play mode')
    parser.add_argument('--nnga', action="store_true", help='Neural Network Genetic algo play mode')
    parser.add_argument('--random',
                        action="store_true",
                        help='RANDOM play mode')
    parser.add_argument(
        '-i',
        '--interactive',
        action="store_true",
        help="Interactive mode: shows a snake game. Only for AI modes.")
    args = parser.parse_args()
    main(args)
