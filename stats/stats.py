import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle


def show_stats(all_score):
    plt.figure()
    plt.plot(all_score)
    plt.plot(all_score.mean()*np.ones(all_score.shape[0]))
    plt.gca().legend(("Scores", "Mean"))
    plt.xlabel("games")
    plt.ylabel("score")
    plt.show()


def show_fitness(all_fitness):
    fig = plt.figure()
    ax = plt.gca()
    im = plt.imshow(all_fitness.T, origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax)
    plt.ylabel("games")
    plt.xlabel("generation")
    plt.show()


def plot_fitness(nb_gen, nb_games):
    path = Path('genetic_data')
    all_fitness = np.zeros([nb_gen, nb_games])
    for gen in range(nb_gen):
        for ii in range(nb_games):
            f = path / Path('data_{}_{}.pickle'.format(gen, ii))
            d = pickle.load(open(f, "rb"))
            all_fitness[gen, ii] = d[0]
    show_fitness(all_fitness)

