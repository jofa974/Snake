import pygame

from brains import dqn_ann, dqn_cnn
from game import read_training_data
from game.environment import Environment
from gen_xy import gen_xy

gen_xy()
training_data = read_training_data()

env = Environment()

# agent = dqn_cnn.DQN_CNN(learning=False)
agent = dqn_ann.DQN_ANN(learning=False)
agent.load()
score = agent.play(max_moves=1000000, env=env, init_training_data=training_data)
pygame.quit()
