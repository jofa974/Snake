import pickle
from collections import OrderedDict
import itertools
import numpy as np
from copy import deepcopy

# TODO: Refactor this shit
def initialise_pop_with_best_parents(path, gen, nb_best):
    p = path.glob('data_' + str(gen) + '_*.pickle')
    files = sorted([x for x in p if x.is_file()])
    f_data = OrderedDict()
    for nb, f in enumerate(files):
        dd = pickle.load(open(f, "rb"))
        f_data[nb] = dd[0]
    sorted_x = sorted(f_data.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = OrderedDict(sorted_x)
    data = deepcopy(dd[1:])
    ####### TODO FIX HERE
    data = [data[:]*0]*len(files)
    id_best = []
    import pdb; pdb.set_trace()
    for nb in list(sorted_dict.keys())[:nb_best]:
        f = path / 'data_{}_{}.pickle'.format(gen, nb)
        dd = pickle.load(open(f, "rb"))
        data[nb, :, :, :] = dd[1], dd[2], dd[3]
        id_best.append(nb)
    return id_best, data


def cross_over(data1, data2):
    result = data1[:]
    randR = np.random.randint(len(data1))
    for i in range(len(data1)):
        if i > randR:
            result[i:i + 1] = data2[i:i + 1]
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
        new_w1 = cross_over(par1[0].flatten(),
                            par2[0].flatten()).reshape(w1_shape)
        new_w2 = cross_over(par1[1].flatten(),
                            par2[1].flatten()).reshape(w2_shape)
        new_b = cross_over(par1[2].flatten(), par2[2].flatten())
        import pdb
        pdb.set_trace()
        new_w1 = mutate(new_w1)
        new_w2 = mutate(new_w2)
        new_b = mutate(new_b)
        import pdb
        pdb.set_trace()
        yield (new_w1, new_w2, new_b)


def generate_new_population(path, gen, nb_pop, nb_best=4):
    id_best, new_pop = initialise_pop_with_best_parents(path, gen, nb_best)
    children = generate_child(new_pop[id_best])
    for ii in range(len(new_pop)):
        if ii not in id_best:

            new_pop[ii] = next(children)
    # while len(new_pop) < nb_pop:
    #     child = generate_child(new_pop)
    #     new_pop.append(next(child))
    return new_pop


if __name__ == '__main__':
    nb_best = 10
    from pathlib import Path
    # best_parents = list(initialise_pop_with_best_parents(Path("../genetic_data/."), 0, 4))
    # children = generate_child(best_parents)
    print(generate_new_population(Path("../genetic_data/."), 0, 100,
                                  nb_best=4))
