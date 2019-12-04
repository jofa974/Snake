#!/usr/bin/env python3
from stats.stats import plot_fitness
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting options')
    parser.add_argument('-p',
                        '--plot_generations',
                        nargs='+',
                        type=int,
                        help="Plot the fitness of the first N generations")
    args = parser.parse_args()
    plot_fitness(args.plot_generations[0], args.plot_generations[1])
