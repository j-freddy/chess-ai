from abc import ABC, abstractmethod
from typing import TypeAlias
import chess

State: TypeAlias = str
Action: TypeAlias = chess.Move

class Model(ABC):
    @abstractmethod
    def predict(self, state: State) -> list[tuple[Action, float]]:
        return NotImplemented
