from abc import ABC, abstractmethod
from enum import IntEnum

from game import Environment


class Reward(IntEnum):
    EAT = 10
    DEAD = -100
    CLOSER = 1
    FURTHER = -1


class Brain(ABC):
    def __init__(self, do_display):
        self.do_display = do_display
        self.env = Environment(do_display=self.do_display)

    @abstractmethod
    def play(self, *args, **kwargs):
        raise NotImplementedError
