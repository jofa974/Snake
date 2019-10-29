import matplotlib.pyplot as plt
import numpy as np


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
    im = plt.imshow(all_fitness, origin="lower")
    fig.colorbar(im, ax=ax)
    plt.xlabel("games")
    plt.ylabel("generation")
    plt.show()
