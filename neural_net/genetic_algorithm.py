import itertools
import pickle
from copy import deepcopy

import numpy as np


def select_best_parents(pickled_data, nb_best):
    sorted_data = deepcopy(pickled_data)
    sorted_data.sort(key=lambda x: x[0], reverse=True)
    return [x[1:] for x in sorted_data[:nb_best]]


def cross_over(data1, data2):
    result = data1[:]
    randR = np.random.randint(len(data1))
    for i in range(len(data1)):
        if i > randR:
            result[i] = data2[i]
    return result
    # return np.concatenate((data1[0:len(data1) // 2], data2[len(data2) // 2:]),
    #                       axis=None)


def mutate(data, rate=0.15):
    old = deepcopy(data).flatten()
    d_shape = data.shape
    for i in range(len(old)):
        r = np.random.rand()
        if r < rate:
            d = np.random.normal(size=1) / 3
            old[i] += d
            # if old[i] < -1:
            #     old[i] = -1
            # if old[i] > 1:
            #     old[i] = 1
    return old.reshape(d_shape)


def generate_child(parent1, parent2):
    new_weights = cross_over(parent1[0], parent2[0])
    new_biases = cross_over(parent1[1], parent2[1])
    new_weights = mutate(new_weights)
    new_biases = mutate(new_biases)
    return (new_weights, new_biases)


def generate_new_population(path, gen, nb_pop, nb_best):
    p = path.glob("data_" + str(gen) + "_*.pickle")
    files = sorted([x for x in p if x.is_file()])
    gen_data = []
    for f in files:
        d = pickle.load(open(f, "rb"))
        gen_data.append(d)
    new_pop = select_best_parents(gen_data, nb_best)
    couples = itertools.combinations(new_pop, 2)
    while True:
        try:
            parent1, parent2 = next(couples)
            child = generate_child(parent1, parent2)
            new_pop.append(child)
        except StopIteration:
            new_pop.append(new_pop[0])
        finally:
            if len(new_pop) == nb_pop:
                return new_pop


if __name__ == "__main__":
    nb_best = 10
    from pathlib import Path

    print(generate_new_population(Path("../genetic_data/."), 0, nb_best=4))
