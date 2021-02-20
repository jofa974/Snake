import brains
import pygame


def human():
    game = brains.human.Human()
    game.play()
    pygame.quit()


def random():
    for _ in range(3):
        game = brains.random.Random()
        game.play()
    pygame.quit()


def bfs():
    # nb_games = INPUTS["BFS"]["games"]
    nb_games = 1
    if nb_games == 1:
        game = brains.bfs.BFS()
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
