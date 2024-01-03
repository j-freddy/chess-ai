import random
import chess

from player import Player

class AIRandom(Player):
    def choose_move(self, position: str) -> str:
        board = chess.Board(position)
        random_move = random.choice(list(board.legal_moves))
        random_move_uci = random_move.uci()
        return random_move_uci
