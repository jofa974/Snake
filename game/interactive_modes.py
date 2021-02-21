import json

import pygame
from brains import bfs, human, random

INPUTS = {}

with open("inputs.json") as json_file:
    INPUTS = json.load(json_file)


def human_game():
    game = human.Human()
    game.play()
    pygame.quit()


def random_game():
    for _ in range(3):
        game = random.Random()
        game.play()
    pygame.quit()


def bfs_game():
    # nb_games = INPUTS["BFS"]["games"]
    nb_games = 1
    if nb_games == 1:
        game = bfs.BFS()
        game.play()
        pygame.quit()
    # else:
    #     all_score = np.zeros(nb_games)
    #     game = brains.bfs.BFS(do_display=False)
    #     for i in range(nb_games):
    #         score = game.play()
    #         all_score[i] = score
    #     pygame.quit()
    #     show_stats(all_score)


# def nnga_game():
#     # Play the best individual of the last generation
#     game = brains.nn_ga.NN_GA(
#         learning=False,
#         gen_id=(INPUTS["NNGA"]["generations"] - 1, 0),
#         hidden_nb=INPUTS["NNGA"]["neurons_per_hidden"],
#     )
#     game.play(dump=False, training_data=training_data)
#     pygame.quit()
