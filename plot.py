#!/usr/bin/env python3
import argparse

from stats.stats import read_fitness, show_fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting options")
    parser.add_argument(
        "-p",
        "--plot_generations",
        nargs="+",
        type=int,
        help="Plot the fitness of the first N generations",
    )
    args = parser.parse_args()
    all_fitness = read_fitness(args.plot_generations[0], args.plot_generations[1])
    show_fitness(all_fitness)
