#!/usr/bin/env python3

import glob
import os
import argparse
from game import human, random, bfs, nn_ga
from stats.stats import show_stats, show_fitness
import numpy as np
import pygame
from neural_net.genetic_algorithm import generate_new_population
from pathlib import Path


def cleanup(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)


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
    elif args.nnga_learn:
        cleanup('genetic_data/*')
        nb_gen = args.nnga_learn[0]
        nb_games = args.nnga_learn[1]
        # all_score = np.zeros(nb_games)
        all_fitness = np.zeros([nb_gen, nb_games])
        for i in range(nb_gen):
            print("Generation: {}".format(i))
            if i > 0:
                path = Path('genetic_data')
                new_pop = generate_new_population(path, gen=i-1, nb_pop=nb_games, nb_best=40)
            else:
                new_pop = [None]*nb_games
            for j in range(nb_games):
                game = nn_ga.NN_GA(display=False, gen_id=(i, j), dna=new_pop[j])
                score, fitness = game.play(max_move=1000, dump=True)
                all_fitness[i][j] = fitness
        #all_score[i] = score
        show_fitness(all_fitness)
        pygame.quit()
    elif args.genetic:
        game = nn_ga.NN_GA(display=True, gen_id=args.genetic)
        nb_games = 1
        game.play(max_move=10000, dump=False)
        pygame.quit()
    else:
        raise NotImplementedError(
            "This game mode has not been implemented yet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake game options')
    parser.add_argument('--human', action="store_true", help='Human play mode')
    parser.add_argument('--bfs', action="store_true", help='BFS play mode')
    parser.add_argument('--genetic',
                        nargs='+',
                        type=int,
                        help='Neural Network Genetic algo play mode')
    parser.add_argument('--nnga_learn',
                        nargs='+',
                        type=int,
                        help='Neural Network Genetic algo learning mode')
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
