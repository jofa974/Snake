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
