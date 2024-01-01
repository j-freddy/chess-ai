from abc import ABC, abstractmethod

import chess

class Player(ABC):
    def __init__(self, color: chess.Color):
        self.is_ai = True
        self.color = color

    @abstractmethod
    def choose_move(self, position: str) -> str:
        return NotImplemented
