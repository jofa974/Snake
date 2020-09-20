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

from brains import bfs, dqn_ann, dqn_cnn, human, nn_ga, random
from game import read_training_data
from neural_net.genetic_algorithm import generate_new_population
from stats.stats import plot_fitness, show_stats
from gen_xy import gen_xy


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
            game = bfs.BFS(do_display=True)
            game.play()
        else:
            all_score = np.zeros(nb_games)
            game = bfs.BFS(do_display=False)
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
            # if (i % 7) == 0:
            #     print("Generating new random training input")
            #     gen_xy()
            if i > 0:
                path = Path("genetic_data")
                new_pop = generate_new_population(
                    path, gen=i - 1, nb_pop=nb_games, nb_best=int(nb_games * 0.2),
                )
            else:
                new_pop = [None] * nb_games

            # Read training_data
            training_data = read_training_data()

            # Training
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                results = executor.map(
                    nn_ga.play_individual,
                    new_pop,
                    itertools.repeat(i, nb_games),
                    range(nb_games),
                    itertools.repeat(training_data, nb_games),
                )
                all_fitness[i][:] = np.array(list(results))
        pygame.quit()
    elif args.nnga_play:
        game = nn_ga.NN_GA(do_display=True, gen_id=args.nnga_play)
        nb_games = 1
        # Read training_data
        training_data = read_training_data()
        game.play(max_move=10000, dump=False, training_data=training_data)
        pygame.quit()
    elif args.dqn_ann:
        nb_games = 100
        game = dqn_ann.DQN_ANN(do_display=False)
        all_score = np.zeros(nb_games)
        for nb in range(nb_games):
            print("Game {}".format(nb))
            # if (nb % 10) == 0:
            #     print("Generating new random training input")
            #     gen_xy()
            training_data = read_training_data()
            score = game.play(max_move=10000, training_data=training_data)
            game.save()
            if score >= np.max(all_score):
                game.save_best()
            all_score[nb] = score
        show_stats(all_score)
        pygame.quit()
    elif args.dqn_ann_play:
        game = dqn_ann.DQN_ANN(do_display=True, learning=False)
        training_data = read_training_data()
        game.load_best()
        score = game.play(max_move=10000, training_data=training_data)
        pygame.quit()
    elif args.dqn_cnn:
        game = dqn_cnn.DQN_CNN(do_display=True)
        nb_games = 100
        all_score = np.zeros(nb_games)
        epsilon, eps_min, eps_decay = 1, 0.05, 0.92
        for nb in range(nb_games):
            print("Game {}".format(nb))
            # if nb > 0:
            #     game.load()
            training_data = read_training_data()
            epsilon = max(epsilon * eps_decay, eps_min)
            epsilon = 0
            score = game.play(
                max_move=10000, training_data=training_data, epsilon=epsilon
            )
            game.save()
            if score > np.max(all_score):
                game.save_best()
            all_score[nb] = score
        show_stats(all_score)
        pygame.quit()
    else:
        raise NotImplementedError("Game mode not implemented.")


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description="Snake game options")
    play_mode_group = parser.add_mutually_exclusive_group(required=True)
    play_mode_group.add_argument("--human", action="store_true", help="Human play mode")
    play_mode_group.add_argument(
        "--bfs",
        type=int,
        default=0,
        help="N games of BFS play mode. N=1 will play interactively.",
    )
    play_mode_group.add_argument(
        "--nnga_play",
        nargs="+",
        type=int,
        help="Neural Network Genetic algo play mode",
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
    play_mode_group.add_argument(
        "--dqn_ann", action="store_true", help="ANN Deep Q-learning mode."
    )
    play_mode_group.add_argument(
        "--dqn_ann_play",
        action="store_true",
        help="Play the best agent trained by the DQN-ANN.",
    )
    play_mode_group.add_argument(
        "--dqn_cnn", action="store_true", help="Convolutional Deep Q-learning mode.",
    )
    args = parser.parse_args()

    # Logger
    format = "%(process)s - %(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # Main function
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))

    if args.nnga_learn:
        plot_fitness(args.nnga_learn[0], args.nnga_learn[1])
