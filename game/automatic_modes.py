import concurrent.futures
import glob
import itertools
import json
import logging
import os
import shutil
import time
from pathlib import Path

import brains.dqn_ann
import brains.nn_ga
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
from neural_net.genetic_algorithm import generate_new_population
from stats.stats import show_fitness, show_stats

INPUTS = {}

with open("inputs.json") as json_file:
    INPUTS = json.load(json_file)


def dqn_ann():
    raise NotImplementedError("Implemented dqn_ann demo")


def nnga():
    raise NotImplementedError("Implemented dqn_ann demo")
    # nb_gen = INPUTS["NNGA"]["generations"]
    # nb_individuals = INPUTS["NNGA"]["individuals"]
    # breed_fraction = INPUTS["NNGA"]["breed_fraction"]
    # workers = INPUTS["NNGA"]["workers"]
    # all_fitness = np.zeros([nb_gen, nb_individuals])
    # training_data = read_training_data()

    # if INPUTS["NNGA"]["learn"]:
    #     if os.path.isdir("genetic_data"):
    #         shutil.rmtree("genetic_data")
    #     os.makedirs("genetic_data")
    #     for i in range(nb_gen):
    #         print("Generation: {}".format(i))
    #         if i > 0:
    #             path = Path("genetic_data")
    #             new_pop = generate_new_population(
    #                 path,
    #                 gen=i - 1,
    #                 nb_pop=nb_individuals,
    #                 nb_best=int(nb_individuals * breed_fraction),
    #             )
    #         else:
    #             new_pop = [None] * nb_individuals

    #         # Parallel Training
    #         with concurrent.futures.ProcessPoolExecutor(
    #             max_workers=workers
    #         ) as executor:
    #             results = executor.map(
    #                 brains.nn_ga.play_individual,
    #                 new_pop,
    #                 itertools.repeat(i, nb_individuals),
    #                 range(nb_individuals),
    #                 itertools.repeat(training_data, nb_individuals),
    #                 itertools.repeat(
    #                     INPUTS["NNGA"]["neurons_per_hidden"], nb_individuals
    #                 ),
    #             )
    #             all_fitness[i][:] = np.array(list(results))
    #     pygame.quit()
    #     show_fitness(all_fitness)
