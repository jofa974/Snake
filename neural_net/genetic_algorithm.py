from pathlib import Path
import pickle
from collections import OrderedDict
import itertools
import numpy as np


def select_best_parents(gen, nb_best):
    p = Path('../genetic_data').glob('*_' + str(gen) + '_*.pickle')
    files = [x for x in p if x.is_file()]
    data = OrderedDict()
    for nb, f in enumerate(files):
        d = pickle.load(open(f, "rb"))
        data[d[0]] = d[1:]
    data = OrderedDict(sorted(data.items(), reverse=True))

    return list(data.values())[:nb_best]


def cross_over(best_parents):
    children = []
    for combi in itertools.combinations(best_parents, 2):
        par1 = combi[0]
        par2 = combi[1]
        w1 = np.concatenate(
            (par1[0][0:len(par1[0]) // 2], par2[0][len(par2[0]) // 2:]),
            axis=None)
        w2 = np.concatenate(
            (par1[1][0:len(par1[1]) // 2], par2[1][len(par2[1]) // 2:]),
            axis=None)
        b = np.concatenate(
            (par1[2][0:len(par1[2]) // 2], par2[2][len(par2[2]) // 2:]),
            axis=None)

        children.append((w1, w2, b))
    return children


if __name__ == '__main__':
    nb_best = 4
    best_parents = list(select_best_parents(8, 4))
    children = cross_over(best_parents)
    
