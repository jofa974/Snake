
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
# Table of Contents

- [Table of Contents](#table-of-contents)
- [Snake](#snake)
  - [How to play](#how-to-play)
    - [Clone repository](#clone-repository)
    - [Install dependencies](#install-dependencies)
    - [Read help menu](#read-help-menu)
  - [Ideas](#ideas)
    - [General interface](#general-interface)
      - [Create a Dash App](#create-a-dash-app)

# Snake

This is an implementation of the Snake game. The available game modes are:

- human: you play using the arrow keys.
- random: a very dumb agent that takes random actions.
- bfs: the agent takes the shortest path to the apple following a Breadth First Search algorithm.
- nnga: the agent takes actions based on a neural network optimized by genetic algorithm.
- dqn: the agent takes actions based on a neural network optimized with deep Q-learning.

## How to play

### Clone repository

```bash
git clone https://github.com/jofa974/Snake.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Read help menu

```bash
./main.py -h
```

## Ideas

### General interface

#### Create a Dash App
  - Remove all use of Pygame
  - Plot game using go.Heatmap or equivalent of imshow
