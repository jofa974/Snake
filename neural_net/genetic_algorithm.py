import pickle
from collections import OrderedDict
import itertools
import numpy as np
from copy import deepcopy


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
    return np.concatenate((data1[0:len(data1) // 2], data2[len(data2) // 2:]),
                          axis=None)


def mutate(data):
    old = deepcopy(data).flatten()
    d_shape = data.shape
    rate = 0.10
    for i in range(len(old)):
        r = np.random.rand()
        if r < rate:
            d = np.random.randn()
            old[i] = d
    return old.reshape(d_shape)


def generate_child(best_parents):
    for combi in itertools.combinations(best_parents, 2):
        par1 = combi[0]
        par2 = combi[1]
        w1_shape = par1[0].shape
        w2_shape = par1[1].shape
        new_w1 = cross_over(par1[0].flatten(),
                            par2[0].flatten()).reshape(w1_shape)
        new_w2 = cross_over(par1[1].flatten(),
                            par2[1].flatten()).reshape(w2_shape)
        new_b = cross_over(par1[2].flatten(), par2[2].flatten())
        new_w1 = mutate(new_w1)
        new_w2 = mutate(new_w2)
        new_b = mutate(new_b)
        yield (new_w1, new_w2, new_b)


def generate_new_population(path, gen, nb_pop, nb_best=4):
    new_pop = select_best_parents(path, gen, nb_best)
    while True:
        child = generate_child(new_pop)
        new_pop.append(next(child))
        if len(new_pop) == nb_pop:
            return new_pop


if __name__ == '__main__':
    nb_best = 10
    from pathlib import Path
    # best_parents = list(select_best_parents(Path("../genetic_data/."), 0, 4))
    # children = generate_child(best_parents)
    print(generate_new_population(Path("../genetic_data/."), 0, nb_best=4))
