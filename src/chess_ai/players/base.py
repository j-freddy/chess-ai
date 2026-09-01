from abc import ABC, abstractmethod

from chess_ai.chess_types import Action, State


class Player(ABC):
    @abstractmethod
    def choose_move(self, state: State) -> Action:
        """
        Return a move that is legal in @state.
        """

    def __str__(self) -> str:
        return type(self).__name__
