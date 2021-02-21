import numpy as np
import pandas as pd
import yaml

import brains.dqn_ann
from game import read_training_data

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

nb_epochs = params["train"]["nb_epochs"]
batch_sample_size = params["train"]["batch_sample_size"]
moves_per_epoch = params["train"]["moves_per_epoch"]

all_score = np.zeros(nb_epochs)
training_data = read_training_data()


agent = brains.dqn_ann.DQN_ANN(
    batch_size=batch_sample_size, memory_size=moves_per_epoch, learning=True,
)
epsilon, eps_min, eps_decay = 1, 0.2, 0.999
losses, mean_rewards = [], []
for epoch in range(nb_epochs):
    print(f"epoch: {epoch}")
    epsilon = max(epsilon * eps_decay, eps_min)
    agent.play(
        max_move=moves_per_epoch, init_training_data=training_data, epsilon=epsilon,
    )
    loss = agent.learn()
    losses.append(loss)
    mean_reward = agent.mean_reward()
    mean_rewards.append(mean_reward)
    agent.list_of_rewards = []
    agent.save()

# Results

df = pd.DataFrame(
    {"epochs": np.arange(1, nb_epochs + 1), "loss": losses, "rewards": mean_rewards}
)
df[["epochs", "loss"]].to_csv("metrics/train/loss.csv", index=False)
df[["epochs", "rewards"]].to_csv("metrics/train/rewards.csv", index=False)
