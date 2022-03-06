import argparse
import logging
import time

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
    epsilon, eps_min, eps_decay = 1.0, 0.2, 0.999
    # epsilon = 0.3
    live = Live()
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

        loss = agent.learn(epochs=params["nb_epochs"])
        live.log("epsilon", epsilon)
        live.log("loss", loss)
        live.log("mean_reward", agent.mean_reward())
        agent.save()
        agent.list_of_rewards = []
        live.next_step()

    logging.info("--- %s seconds ---" % (time.time() - start_time))
