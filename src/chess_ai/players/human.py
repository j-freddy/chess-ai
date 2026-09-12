import logging

import chess

from chess_ai.chess_types import Action, State
from chess_ai.players.base import Player, PlayerExit

logger = logging.getLogger(__name__)


class Human(Player):
    def choose_move(self, state: State) -> Action:
        board = chess.Board(state)

        while True:
            entered = input("Enter your move: ")

            if entered == "exit":
                raise PlayerExit

            try:
                return board.parse_san(entered)
            except chess.InvalidMoveError:
                logger.info("Invalid move, try again.")
            except chess.IllegalMoveError:
                logger.info("Illegal move, try again.")
            except chess.AmbiguousMoveError:
                logger.info("Ambiguous move, try again.")
