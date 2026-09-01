import chess

from chess_ai.chess_types import Action, State
from chess_ai.players.base import Player


class Human(Player):
    def choose_move(self, state: State) -> Action:
        board = chess.Board(state)

        while True:
            entered = input("Enter your move: ")

            try:
                return board.parse_san(entered)
            except chess.InvalidMoveError:
                print("Invalid move, try again.")
            except chess.IllegalMoveError:
                print("Illegal move, try again.")
            except chess.AmbiguousMoveError:
                print("Ambiguous move, try again.")
