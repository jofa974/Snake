import pickle
from collections import OrderedDict
import itertools
import numpy as np
from copy import deepcopy


def select_best_parents(pickled_data, nb_best):
    sorted_data = deepcopy(pickled_data)
    sorted_data.sort(key=lambda x: x[0], reverse=True)
    return sorted_data[:nb_best]


def cross_over(data1, data2):
    result = data1[:]
    randR = np.random.randint(len(data1))
    for i in range(len(data1)):
        if i > randR:
            result[i] = data2[i]
    return result
    # return np.concatenate((data1[0:len(data1) // 2], data2[len(data2) // 2:]),
    #                       axis=None)


def mutate(data):
    old = deepcopy(data).flatten()
    d_shape = data.shape
    rate = 0.05
    for i in range(len(old)):
        r = np.random.rand()
        if r < rate:
            d = np.random.normal(size=1) / 5
            old[i] = +d
            if old[i] < -1:
                old[i] = -1
            if old[i] > 1:
                old[i] = 1
    return old.reshape(d_shape)


def generate_child(best_parents):
    for combi in itertools.combinations(best_parents, 2):
        par1 = combi[0]
        par2 = combi[1]
        w1_shape = par1[0].shape
        w2_shape = par1[1].shape
        new_w1 = par1[0]
        new_w2 = par1[1]
        new_b = par1[2]
        # new_w1 = cross_over(par1[0].flatten(),
        #                     par2[0].flatten()).reshape(w1_shape)
        # new_w2 = cross_over(par1[1].flatten(),
        #                     par2[1].flatten()).reshape(w2_shape)
        # new_b = cross_over(par1[2].flatten(), par2[2].flatten())
        # new_w1 = mutate(new_w1)
        # new_w2 = mutate(new_w2)
        # new_b = mutate(new_b)
        yield (new_w1, new_w2, new_b)


def generate_new_population(path, gen, nb_pop, nb_best=4):
    new_pop = select_best_parents(path, gen, nb_best)
    child = generate_child(new_pop)
    while True:
        try:
            new_pop.append(next(child))
        except StopIteration:
            new_pop.append(new_pop[0])
        finally:
            if len(new_pop) == nb_pop:
                return new_pop


if __name__ == '__main__':
    nb_best = 10
    from pathlib import Path
    # best_parents = list(select_best_parents(Path("../genetic_data/."), 0, 4))
    # children = generate_child(best_parents)
    print(generate_new_population(Path("../genetic_data/."), 0, nb_best=4))
