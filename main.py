#!/usr/bin/env python3
import argparse
import concurrent.futures
import glob
import itertools
import logging
import os
import time
from pathlib import Path

import numpy as np
import pygame

from game import bfs, human, nn_ga, random
from gen_xy import gen_xy
from neural_net.genetic_algorithm import generate_new_population
from stats.stats import show_fitness, show_stats


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
    elif args.bfs > 0:
        nb_games = args.bfs
        if nb_games == 1:
            game = bfs.BFS(display=True)
            game.play()
        else:
            all_score = np.zeros(nb_games)
            game = bfs.BFS(display=False)
            for i in range(nb_games):
                score = game.play()
                all_score[i] = score
            show_stats(all_score)
        pygame.quit()
    elif args.nnga_learn:
        cleanup("genetic_data/*")
        nb_gen = args.nnga_learn[0]
        nb_games = args.nnga_learn[1]
        all_fitness = np.zeros([nb_gen, nb_games])
        for i in range(nb_gen):
            print("Generation: {}".format(i))
            if (i % 15) == 0:
                print("Generating new random training input")
                gen_xy()
            if i > 0:
                path = Path("genetic_data")
                new_pop = generate_new_population(
                    path, gen=i - 1, nb_pop=nb_games, nb_best=int(nb_games * 0.20)
                )
            else:
                new_pop = [None] * nb_games
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                results = executor.map(
                    nn_ga.play_individual,
                    new_pop,
                    itertools.repeat(i, nb_games),
                    range(nb_games),
                )
                all_fitness[i][:] = np.array(list(results))
        # show_fitness(all_fitness)
        pygame.quit()
    elif args.genetic:
        game = nn_ga.NN_GA(display=True, gen_id=args.genetic)
        nb_games = 1
        game.play(max_move=10000, dump=False, learn=False)
        pygame.quit()
    else:
        raise NotImplementedError("This game mode has not been implemented yet")


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description="Snake game options")
    play_mode_group = parser.add_mutually_exclusive_group(required=True)
    play_mode_group.add_argument("--human", action="store_true", help="Human play mode")
    play_mode_group.add_argument(
        "--bfs",
        type=int,
        default=0,
        help="N games of BFS play mode. N=1 will play interactively. N>1 w.ill show statistics only",
    )
    play_mode_group.add_argument(
        "--genetic", nargs="+", type=int, help="Neural Network Genetic algo play mode"
    )
    play_mode_group.add_argument(
        "--nnga_learn",
        nargs="+",
        type=int,
        help="Neural Network Genetic algo learning mode",
    )
    play_mode_group.add_argument(
        "--random", action="store_true", help="RANDOM play mode."
    )
    args = parser.parse_args()

    # Logger
    format = "%(process)s - %(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # Main function
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
