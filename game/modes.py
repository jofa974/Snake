import concurrent.futures
import glob
import itertools
import json
import logging
import os
import shutil
import time
from pathlib import Path

import brains.bfs
import brains.dqn_ann
import brains.human
import brains.nn_ga
import brains.random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
from neural_net.genetic_algorithm import generate_new_population
from stats.stats import show_fitness, show_stats

from . import read_training_data

INPUTS = {}

with open("inputs.json") as json_file:
    INPUTS = json.load(json_file)


def bfs():
    nb_games = INPUTS["BFS"]["games"]
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
    nb_epochs = INPUTS["DQN"]["epochs"]
    all_score = np.zeros(nb_epochs)
    training_data = read_training_data()
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.set_xlim([1, nb_epochs])
    ax.set_ylim([0, 0.5])
    if INPUTS["DQN"]["learn"]:
        agent = brains.dqn_ann.DQN_ANN(
            batch_size=INPUTS["DQN"]["batch_sample_size"],
            memory_size=INPUTS["DQN"]["moves_per_epoch"],
            do_display=False,
            learning=True,
        )
        epsilon, eps_min, eps_decay = 1, 0.2, 0.999
        for epoch in range(nb_epochs):
            epsilon = max(epsilon * eps_decay, eps_min)
            agent.play(
                max_move=INPUTS["DQN"]["moves_per_epoch"],
                init_training_data=training_data,
                epsilon=epsilon,
            )
            loss = agent.learn()
            agent.save()
            ax.scatter(
                [epoch + 1], [loss], s=20, c="r",
            )
            plt.title(f"epsilon={epsilon}")
            plt.draw()
            plt.pause(0.00001)
        plt.savefig("last_training.eps")
        pygame.quit()
    else:
        agent = brains.dqn_ann.DQN_ANN(do_display=True, learning=False)
        training_data = read_training_data()
        agent.load()
        score = agent.play(max_move=1000000, init_training_data=training_data)
        pygame.quit()


def nnga():
    nb_gen = INPUTS["NNGA"]["generations"]
    nb_individuals = INPUTS["NNGA"]["individuals"]
    breed_fraction = INPUTS["NNGA"]["breed_fraction"]
    workers = INPUTS["NNGA"]["workers"]
    all_fitness = np.zeros([nb_gen, nb_individuals])
    training_data = read_training_data()

    if INPUTS["NNGA"]["learn"]:
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
                    itertools.repeat(
                        INPUTS["NNGA"]["neurons_per_hidden"], nb_individuals
                    ),
                )
                all_fitness[i][:] = np.array(list(results))
        pygame.quit()
        show_fitness(all_fitness)
    else:
        # Play the best individual of the last generation
        game = brains.nn_ga.NN_GA(
            do_display=True,
            gen_id=(INPUTS["NNGA"]["generations"] - 1, 0),
            hidden_nb=INPUTS["NNGA"]["neurons_per_hidden"],
        )
        game.play(dump=False, training_data=training_data)
        pygame.quit()

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
