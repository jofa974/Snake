import pygame

from brains import dqn_ann
from game.environment import Environment

env = Environment()

agent = dqn_ann.DQN_ANN(learning=False)
agent.load()
score = agent.play(max_move=1000000, env=env)
pygame.quit()
