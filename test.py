import brains.dqn_ann
import pygame

from game import read_training_data

agent = brains.dqn_ann.DQN_ANN(do_display=True, learning=False)
training_data = read_training_data()
agent.load()
score = agent.play(max_move=1000000, init_training_data=training_data)
pygame.quit()
