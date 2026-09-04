import chess
import numpy as np

from chess_ai.chess_types import Action, State
from chess_ai.models.base import Model

CHECKMATE_VALUE = 10000
PIECE_TO_VALUE: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}
PRIOR_OFFSET = 1.0


def statically_score_move(move: chess.Move, fen: str) -> float:
    board = chess.Board(fen)
    board.push(move)

    if board.is_checkmate():
        return CHECKMATE_VALUE

    # Check captures
    board.pop()

    if board.is_capture(move):
        piece_type = board.piece_type_at(move.to_square)

        if piece_type is None:
            assert board.is_en_passant(move)
            return PIECE_TO_VALUE[chess.PAWN]

        return PIECE_TO_VALUE[piece_type]

    return 0.0


class ModelNaive(Model):
    def predict(self, state: State) -> list[tuple[Action, float]]:
        """
        Use static score evaluation to compute prior probabilities for each move
        in a given board state. Score is from the perspective of the current
        player w.r.t. @state.
        """

        board = chess.Board(state)
        actions = list(board.legal_moves)

        prior = np.array(
            [
                statically_score_move(action, state) + PRIOR_OFFSET
                for action in actions
            ]
        )
        normalised_prior = prior / np.sum(prior)

        return [
            (action, float(prob))
            for action, prob in zip(actions, normalised_prior, strict=True)
        ]
