import argparse
import logging
import time

import numpy as np
import pandas as pd
import yaml

import brains.dqn_ann
import brains.dqn_cnn
from game import read_training_data
from gen_xy import gen_xy

AGENT_CLASSES = {"ann": brains.dqn_ann.DQN_ANN, "cnn": brains.dqn_cnn.DQN_CNN}

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
    format = "Training - %(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # Main function
    start_time = time.time()

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
        params = params[f"{args.algorithm}_{args.network}"]["train"]

    agent = AGENT_CLASSES[args.network](
        batch_size=params["batch_size"],
        memory_size=10000,
        learning=True,
    )
    epsilon, eps_min, eps_decay = 1.0, 0.2, 0.99
    # epsilon = 0.3
    losses, mean_rewards, cum_rewards = [], [], []
    for ep in range(params["nb_episodes"]):
        gen_xy()
        training_data = read_training_data()

        if ep % 20 == 0:
            epsilon = max(epsilon * eps_decay, eps_min)
            logging.info(f"episode: {ep} ; epsilon: {epsilon}")
        agent.play(
            max_moves=params["max_moves"],
            init_training_data=training_data,
            epsilon=epsilon,
        )

        losses.append(agent.learn(epochs=params["nb_epochs"]))
        mean_rewards.append(agent.mean_reward())
        cum_rewards.append(agent.cumulative_reward())
        agent.list_of_rewards = []
        agent.save()

    # Results

    df = pd.DataFrame(
        {
            "episode": np.arange(1, params["nb_episodes"] + 1),
            "loss": losses,
            "rewards": mean_rewards,
            "cum_rewards": cum_rewards,
        }
    )
    df[["episode", "loss"]].to_csv(
        f"metrics/train/{args.algorithm}_{args.network}/loss.csv", index=False
    )
    df[["episode", "rewards"]].to_csv(
        f"metrics/train/{args.algorithm}_{args.network}/rewards.csv", index=False
    )
    df[["episode", "cum_rewards"]].to_csv(
        f"metrics/train/{args.algorithm}_{args.network}/cum_rewards.csv", index=False
    )

    logging.info("--- %s seconds ---" % (time.time() - start_time))
