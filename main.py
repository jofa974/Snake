#!/usr/bin/env python3
import argparse
import concurrent.futures
import glob
import itertools
import logging
import os
import time
from pathlib import Path

import numpy as np

from game import modes, read_training_data
from neural_net.genetic_algorithm import generate_new_population
from stats.stats import read_fitness, show_fitness, show_stats

GAME_MODES = {
    "human": modes.human,
    "random": modes.random,
    "bfs": modes.bfs,
    "nnga": modes.nnga,
    "dqn": modes.dqn_ann,
}


def main(args):
    GAME_MODES[args.mode]()


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description="Snake game options")
    play_mode_group = parser.add_mutually_exclusive_group(required=True)
    play_mode_group.add_argument(
        "--mode", type=str, choices=GAME_MODES.keys(), help="TBD",
    )
    args = parser.parse_args()

    # Logger
    format = "%(process)s - %(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # Main function
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
