import pickle
from collections import OrderedDict
import itertools
import numpy as np


def select_best_parents(path, gen, nb_best):
    p = path.glob('data_' + str(gen) + '_*.pickle')
    files = sorted([x for x in p if x.is_file()])
    f_data = OrderedDict()
    for nb, f in enumerate(files):
        d = pickle.load(open(f, "rb"))
        f_data[nb] = d[0]
    sorted_x = sorted(f_data.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = OrderedDict(sorted_x)
    data = []
    for nb in sorted_dict.keys():
        f = path / 'data_{}_{}.pickle'.format(gen, nb)
        d = pickle.load(open(f, "rb"))
        data.append(d[1:])
    return data[:nb_best]


def cross_over(data1, data2):
    return np.concatenate(
        (data1[0:len(data1) // 2], data2[len(data2) // 2:]),
        axis=None)


def generate_children(best_parents):
    children = []
    for combi in itertools.combinations(best_parents, 2):
        par1 = combi[0]
        par2 = combi[1]
        w1_shape = par1[0].shape
        w2_shape = par1[1].shape
        new_w1 = cross_over(par1[0].flatten(), par2[0].flatten()).reshape(w1_shape)
        new_w2 = cross_over(par1[1].flatten(), par2[1].flatten()).reshape(w2_shape)
        new_b = cross_over(par1[2].flatten(), par2[2].flatten())
        children.append((new_w1, new_w2, new_b))
    return children


def generate_new_population(path, gen, nb_best=4):
    best_parents = select_best_parents(path, gen, nb_best)
    children = generate_children(best_parents)
    best_parents.extend(children)
    return best_parents


if __name__ == '__main__':
    nb_best = 4
    from pathlib import Path
    # best_parents = list(select_best_parents(Path("../genetic_data/."), 0, 4))
    # children = generate_children(best_parents)
    print(generate_new_population(Path("../genetic_data/."), 0, nb_best=4))
