from pathlib import Path


def select_best_parents(gen, nb_best):
    p = Path('../genetic_data').glob('*_'+str(gen)+'_*.pickle')
    files = [x for x in p if x.is_file()]
    for f in files:
        fitness
    print(files)



if __name__ == '__main__':
    select_best_parents(8, 2)
