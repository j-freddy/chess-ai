import chess

from chess_ai.chess_types import Action, State
from chess_ai.players import Player

ILLEGAL_MOVE = chess.Move.from_uci("e2e5")


class IllegalPlayer(Player):
    """
    A player that breaks the choose_move contract by returning a move that is
    not legal.
    """

    def choose_move(self, state: State) -> Action:
        return ILLEGAL_MOVE
