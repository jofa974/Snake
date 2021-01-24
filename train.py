import numpy as np
import pandas as pd
import pygame
import yaml

import brains.dqn_ann
from game import read_training_data

# with open("inputs.json") as json_file:
#     INPUTS = json.load(json_file)


# nb_epochs = INPUTS["DQN"]["epochs"]
# batch_sample_size = INPUTS["DQN"]["batch_sample_size"]
# moves_per_epoch = INPUTS["DQN"]["moves_per_epoch"]

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

nb_epochs = params["train"]["nb_epochs"]
batch_sample_size = params["train"]["batch_sample_size"]
moves_per_epoch = params["train"]["moves_per_epoch"]

all_score = np.zeros(nb_epochs)
training_data = read_training_data()


agent = brains.dqn_ann.DQN_ANN(
    batch_size=batch_sample_size,
    memory_size=moves_per_epoch,
    do_display=False,
    learning=True,
)
epsilon, eps_min, eps_decay = 1, 0.2, 0.999
losses = []
for epoch in range(nb_epochs):
    epsilon = max(epsilon * eps_decay, eps_min)
    agent.play(
        max_move=moves_per_epoch, init_training_data=training_data, epsilon=epsilon,
    )
    loss = agent.learn()
    losses.append(loss)
    agent.save()
pygame.quit()

# Results

df = pd.DataFrame({"epochs": np.arange(1, nb_epochs + 1), "loss": losses})
df.to_csv("loss.csv", index=False)
