import chess
import numpy as np

from ai.model import Action, State
from ai.model import Model

class ModelNaive(Model):
    def __init__(self):
        super().__init__()
        
        self.checkmate_value = 10000
        self.piece_to_value: dict[chess.PieceType, float] = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        
        self._prior_offset = 1.0
    
    def _statically_score_move(self, move: chess.Move, fen: str) -> float:
        board = chess.Board(fen)
        board.push(move)

        if board.is_checkmate():
            return self.checkmate_value

        # Check captures
        board.pop()
        if board.is_capture(move):
            piece_type = board.piece_type_at(move.to_square)

            if piece_type is None:
                assert board.is_en_passant(move)
                return self.piece_to_value[chess.PAWN]

            return self.piece_to_value[piece_type]

        return 0.0
    
    def predict(self, state: State) -> list[tuple[Action, float]]:
        """
        Use static score evaluation to compute prior probabilities for each move
        in a given board state. Score is from the perspective of the current
        player w.r.t. @state.
        """

        board = chess.Board(state)
        actions = list(board.legal_moves)
        prior = np.empty(len(actions))

        for i in range(len(actions)):
            prior[i] = self._statically_score_move(actions[i], state)\
                + self._prior_offset

        normalised_prior = prior / np.sum(prior)
        return list(zip(actions, normalised_prior))
