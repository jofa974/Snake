import concurrent.futures
import glob
import itertools
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pygame

import brains.bfs
import brains.dqn_ann
import brains.human
import brains.nn_ga
import brains.random
from neural_net.genetic_algorithm import generate_new_population
from stats.stats import plot_fitness, show_stats

from . import read_training_data

with open("inputs.json") as json_file:
    inputs = json.load(json_file)


def human():
    game = brains.human.Human()
    game.play()
    pygame.quit()


def random():
    for _ in range(3):
        game = brains.random.Random()
        game.play()
    pygame.quit()


def bfs():
    nb_games = inputs["BFS"]["games"]
    if nb_games == 1:
        game = brains.bfs.BFS(do_display=True)
        game.play()
        pygame.quit()
    else:
        all_score = np.zeros(nb_games)
        game = brains.bfs.BFS(do_display=False)
        for i in range(nb_games):
            score = game.play()
            all_score[i] = score
        pygame.quit()
        show_stats(all_score)


def dqn_ann():
    nb_games = inputs["DQN"]["games"]
    all_score = np.zeros(nb_games)
    training_data = read_training_data()
    if inputs["DQN"]["learn"]:
        game = brains.dqn_ann.DQN_ANN(do_display=False, learning=True)
        for nb in range(nb_games):
            print("Game {}".format(nb))
            score = game.play(
                max_move=inputs["DQN"]["max_move"], training_data=training_data
            )
            game.save()
            all_score[nb] = score
        show_stats(all_score)
        pygame.quit()
    else:
        game = brains.dqn_ann.DQN_ANN(do_display=True, learning=False)
        training_data = read_training_data()
        game.load()
        score = game.play(training_data=training_data)
        pygame.quit()


def nnga():
    nb_gen = inputs["NNGA"]["generations"]
    nb_individuals = inputs["NNGA"]["individuals"]
    breed_fraction = inputs["NNGA"]["breed_fraction"]
    workers = inputs["NNGA"]["workers"]
    all_fitness = np.zeros([nb_gen, nb_individuals])
    training_data = read_training_data()

    if inputs["NNGA"]["learn"]:
        if os.path.isdir("genetic_data"):
            shutil.rmtree("genetic_data")
        os.makedirs("genetic_data")
        for i in range(nb_gen):
            print("Generation: {}".format(i))
            if i > 0:
                path = Path("genetic_data")
                new_pop = generate_new_population(
                    path,
                    gen=i - 1,
                    nb_pop=nb_individuals,
                    nb_best=int(nb_individuals * breed_fraction),
                )
            else:
                new_pop = [None] * nb_individuals

            # Parallel Training
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers
            ) as executor:
                results = executor.map(
                    brains.nn_ga.play_individual,
                    new_pop,
                    itertools.repeat(i, nb_individuals),
                    range(nb_individuals),
                    itertools.repeat(training_data, nb_individuals),
                )
                all_fitness[i][:] = np.array(list(results))
        pygame.quit()
        plot_fitness(nb_gen=nb_gen, nb_games=nb_individuals)
    else:
        # Play the best individual of the last generation
        game = brains.nn_ga.NN_GA(
            do_display=True, gen_id=(inputs["NNGA"]["generations"] - 1, 0)
        )
        game.play(dump=False, training_data=training_data)
        pygame.quit()

    # elif args.dqn_ann:
    #     nb_games = 100
    #     game = dqn_ann.DQN_ANN(do_display=False)
    #     all_score = np.zeros(nb_games)
    #     for nb in range(nb_games):
    #         print("Game {}".format(nb))
    #         # if (nb % 10) == 0:
    #         #     print("Generating new random training input")
    #         #     gen_xy()
    #         training_data = read_training_data()
    #         score = game.play(max_move=10000, training_data=training_data)
    #         game.save()
    #         if score >= np.max(all_score):
    #             game.save_best()
    #         all_score[nb] = score
    #     show_stats(all_score)
    #     pygame.quit()
    # elif args.dqn_ann_play:
    #     game = dqn_ann.DQN_ANN(do_display=True, learning=False)
    #     training_data = read_training_data()
    #     game.load_best()
    #     score = game.play(max_move=10000, training_data=training_data)
    #     pygame.quit()
    # elif args.dqn_cnn:
    #     game = dqn_cnn.DQN_CNN(do_display=True)
    #     nb_games = 100
    #     all_score = np.zeros(nb_games)
    #     epsilon, eps_min, eps_decay = 1, 0.05, 0.92
    #     for nb in range(nb_games):
    #         print("Game {}".format(nb))
    #         # if nb > 0:
    #         #     game.load()
    #         training_data = read_training_data()
    #         epsilon = max(epsilon * eps_decay, eps_min)
    #         epsilon = 0
    #         score = game.play(
    #             max_move=10000, training_data=training_data, epsilon=epsilon
    #         )
    #         game.save()
    #         if score > np.max(all_score):
    #             game.save_best()
    #         all_score[nb] = score
    #     show_stats(all_score)
    #     pygame.quit()
    # else:
    #     raise NotImplementedError("Game mode not implemented.")
