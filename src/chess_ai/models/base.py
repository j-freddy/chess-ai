from abc import ABC, abstractmethod

from chess_ai.chess_types import Action, State


class Model(ABC):
    @abstractmethod
    def predict(self, state: State) -> list[tuple[Action, float]]:
        """
        Return a prior probability for each legal move in @state.
        """
