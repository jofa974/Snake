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
    im = plt.imshow(all_fitness.T, origin="lower")
    fig.colorbar(im, ax=ax)
    plt.ylabel("games")
    plt.xlabel("generation")
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    import pickle
    from collections import OrderedDict
    gen = 9
    path = Path("../genetic_data/.")
    p = path.glob('data_' + str(gen) + '_*.pickle')
    files = sorted([x for x in p if x.is_file()])
    f_data = OrderedDict()
    for nb, f in enumerate(files):
        d = pickle.load(open(f, "rb"))
        f_data[nb] = d[0]
    sorted_x = sorted(f_data.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = OrderedDict(sorted_x)
    data = []*nb_best
    for nb in sorted_dict.keys():
        f = path / 'data_{}_{}.pickle'.format(gen, nb)
        d = pickle.load(open(f, "rb"))
        data[nb] = d[1:]
    import pdb; pdb.set_trace()
    
