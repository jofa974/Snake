from abc import ABC, abstractmethod

from game import Environment


class brain(ABC):
    def __init__(self, do_display):
        self.do_display = do_display
        self.env = Environment(do_display=self.do_display)

    @abstractmethod
    def play(self, *args, **kwargs):
        raise NotImplementedError
