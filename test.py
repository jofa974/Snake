import argparse
import logging
import time

import numpy as np
import pandas as pd
import yaml
from dvclive import Live

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
    format = "Testing - %(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # Main function
    start_time = time.time()

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
        params = params[f"{args.algorithm}_{args.network}"]["test"]

    agent = AGENT_CLASSES[args.network](
        learning=False,
    )
    agent.load()

    live = Live()
    for ep in range(params["nb_episodes"]):
        gen_xy()
        training_data = read_training_data()

        score = agent.play(max_moves=100, init_training_data=training_data)
        live.log("scores", score)
        live.next_step()

    logging.info("--- %s seconds ---" % (time.time() - start_time))
