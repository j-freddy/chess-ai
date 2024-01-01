import chess

from player import Player

class PlayerHuman(Player):
    def __init__(self, color: chess.Color):
        super().__init__(color)
        self.is_ai = False

    def choose_move(self, position: str) -> str:
        return input("Enter your move: ")
