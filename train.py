import numpy as np
import pandas as pd
import pygame
import yaml

import brains.dqn_ann
import brains.dqn_cnn
from game import read_training_data

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

nb_episodes = params["train"]["nb_episodes"]
nb_epochs = params["train"]["nb_epochs"]
batch_sample_size = params["train"]["batch_sample_size"]
moves_per_epoch = params["train"]["moves_per_epoch"]

all_score = np.zeros(nb_epochs)
training_data = read_training_data()


agent = brains.dqn_ann.DQN_ANN(
    batch_size=batch_sample_size,
    memory_size=10000,
    do_display=False,
    learning=True,
)
# epsilon, eps_min, eps_decay = 1.0, 0.2, 0.999
epsilon = 0.3
losses, mean_rewards = [], []
for ep in range(nb_episodes):
    print(f"episode: {ep}")
    # epsilon = max(epsilon * eps_decay, eps_min)
    agent.play(
        max_move=moves_per_epoch,
        init_training_data=training_data,
        epsilon=epsilon,
    )

    loss = 0.0
    for epoch in range(nb_epochs):
        loss += agent.learn() / nb_epochs

    losses.append(loss)
    mean_reward = agent.mean_reward()
    mean_rewards.append(mean_reward)
    agent.list_of_rewards = []
    agent.save()
pygame.quit()

# Results

df = pd.DataFrame(
    {"episode": np.arange(1, nb_episodes + 1), "loss": losses, "rewards": mean_rewards}
)
df[["episode", "loss"]].to_csv("loss.csv", index=False)
df[["episode", "rewards"]].to_csv("rewards.csv", index=False)
