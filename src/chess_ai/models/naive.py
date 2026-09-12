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


def score_move_on_board(board: chess.Board, move: chess.Move) -> float:
    """
    Score @move on @board, leaving @board unchanged.

    Taking the board rather than a FEN keeps this off the search's hot path:
    parsing a FEN once per position instead of once per candidate move is
    worth roughly an order of magnitude in playout throughput.
    """

    board.push(move)
    is_checkmate = board.is_checkmate()
    board.pop()

    if is_checkmate:
        return CHECKMATE_VALUE

    # Check captures
    if board.is_capture(move):
        piece_type = board.piece_type_at(move.to_square)

        if piece_type is None:
            assert board.is_en_passant(move)
            return PIECE_TO_VALUE[chess.PAWN]

        return PIECE_TO_VALUE[piece_type]

    return 0.0


def statically_score_move(move: chess.Move, fen: str) -> float:
    return score_move_on_board(chess.Board(fen), move)


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
                score_move_on_board(board, action) + PRIOR_OFFSET
                for action in actions
            ]
        )
        normalised_prior = prior / np.sum(prior)

        return [
            (action, float(prob))
            for action, prob in zip(actions, normalised_prior, strict=True)
        ]
