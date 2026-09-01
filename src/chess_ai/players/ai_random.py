import random

import chess

from chess_ai.chess_types import Action, State
from chess_ai.players.ai import AI


class AIRandom(AI):
    NAME = "RandomBot"
    AUTHOR = "Freddy Jiang"

    def choose_move(self, state: State) -> Action:
        board = chess.Board(state)
        return random.choice(list(board.legal_moves))
