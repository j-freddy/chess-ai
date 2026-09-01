from typing import ClassVar

from chess_ai.players.base import Player


class AI(Player):
    """
    A player that chooses its own moves, and can therefore identify itself to a
    UCI GUI and be given a budget to think within.
    """

    NAME: ClassVar[str]
    AUTHOR: ClassVar[str]

    def __init__(self, time_budget: float = 5.0):
        self.time_budget = time_budget

    def __str__(self) -> str:
        return self.NAME
