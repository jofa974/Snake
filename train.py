import argparse
import logging
import time

import numpy as np
import pandas as pd
import yaml

import brains.dqn_ann
import brains.dqn_cnn
from game import read_training_data

if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description="Snake game options")
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=["dqn"],
        help="RL algorithm used for training.",
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        choices=["ann", "cnn"],
        help="Neural network architecture.",
    )

    args = parser.parse_args()

    # Logger
    format = "%(process)s - %(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # Main function
    start_time = time.time()

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
        params = params[f"{args.algorithm}_{args.network}"]["train"]

    nb_episodes = params["nb_episodes"]
    nb_epochs = params["nb_epochs"]
    batch_sample_size = params["batch_size"]
    max_moves = params["max_moves"]

    all_score = np.zeros(nb_epochs)
    training_data = read_training_data()

    agent = brains.dqn_ann.DQN_ANN(
        batch_size=batch_sample_size,
        memory_size=10000,
        learning=True,
    )
    # epsilon, eps_min, eps_decay = 1.0, 0.2, 0.999
    epsilon = 0.3
    losses, mean_rewards = [], []
    for ep in range(nb_episodes):
        print(f"episode: {ep}")
        # epsilon = max(epsilon * eps_decay, eps_min)
        agent.play(
            max_moves=max_moves,
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

    # Results

    df = pd.DataFrame(
        {
            "episode": np.arange(1, nb_episodes + 1),
            "loss": losses,
            "rewards": mean_rewards,
        }
    )
    df[["episode", "loss"]].to_csv(
        f"metrics/train/{args.algorithm}_{args.network}/loss.csv", index=False
    )
    df[["episode", "rewards"]].to_csv(
        f"metrics/train/{args.algorithm}_{args.network}/rewards.csv", index=False
    )

    print("--- %s seconds ---" % (time.time() - start_time))
