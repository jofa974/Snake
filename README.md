
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
# Table of Contents
- [Table of Contents](#table-of-contents)
- [Snake](#snake)
  - [Getting started](#getting-started)
  - [Training a DQL agent](#training-a-dql-agent)
  - [Play](#play)
    - [Human mode](#human-mode)
    - [Random mode](#random-mode)
    - [Breadth-first search algorithm](#breadth-first-search-algorithm)

# Snake

This is an implementation of the Snake game. The available game modes are:

- human: you play using the arrow keys.
- random: a very dumb agent that takes random actions.
- bfs: the agent takes the shortest path to the apple following a Breadth First Search algorithm.
- dqn: the agent takes actions based on a neural network optimized with deep Q-learning.

## Getting started

Install the requirements in a virtual environment. I suggest using `pyenv`:

```bash
pyenv virtualenv 3.9.7 Snake
pyenv local Snake
pip install -r requirements
pip install -e .
```

## Training a DQL agent

Two agents are available with different levels of knowledge about their environment:
- `dqn_ann.py`: it knows the distances to the walls, the position of the apple, if it's going towards it, how big it is.
- `dqn_cnn.py`: it knows the full picture of the environment, just like a human player.

Training the agents is done using the `dvc.yaml` pipeline file and the usual `dvc repro` and/or `dvc exp` commands. For more info about `dvc`: www.dvc.org

## Play

### Human mode

Use the command and do your best:

```bash
./main.py --mode=human
```

### Random mode

```bash
./main.py --mode=random
```

### Breadth-first search algorithm

```bash
./main.py --mode=bfs
```


