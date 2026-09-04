import chess
import numpy as np
import pytest

from chess_ai.models.naive import (
    CHECKMATE_VALUE,
    PIECE_TO_VALUE,
    PRIOR_OFFSET,
    ModelNaive,
    statically_score_move,
)


@pytest.mark.parametrize(
    "move, fen, score",
    [
        ("a1a8", "4k3/8/4K3/8/8/8/8/R7 w - - 0 1", CHECKMATE_VALUE),
        (
            "d8d5",
            "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
            PIECE_TO_VALUE[chess.PAWN],
        ),
        (
            "d3e4",
            "r3kb1r/ppqn1ppp/2p1p3/8/3Pn3/3Q1N1P/PPP2PP1/R1B2RK1 w kq - 0 12",
            PIECE_TO_VALUE[chess.KNIGHT],
        ),
        (
            "f5d3",
            "rn2kb1r/ppq2ppp/2p1pn2/5b2/3P4/2NB1N1P/PPP2PP1/R1BQ1RK1 b kq - 1 9",
            PIECE_TO_VALUE[chess.BISHOP],
        ),
        (
            "c5d6",
            "2k2r2/pp2qp2/2prpn1p/2P3p1/3P4/7P/PP2QPP1/R2R2K1 w - - 0 23",
            PIECE_TO_VALUE[chess.ROOK],
        ),
        (
            "d6e5",
            "2k2r2/pp3p2/2pqpn1p/4Q1p1/3P4/7P/PP3PP1/R2R2K1 b - - 1 24",
            PIECE_TO_VALUE[chess.QUEEN],
        ),
    ],
)
def test_statically_score_move(move, fen, score):
    assert statically_score_move(chess.Move.from_uci(move), fen) == score


def test_predict():
    # Custom position
    # Best move: mate
    # Then: capture knight followed by capture pawn according to static
    # evaluator
    fen = "rnbq1b1r/pp1pk3/3n2Q1/5p2/8/8/PP3PPP/RNB1KB1R w KQ - 0 13"

    model = ModelNaive()
    prior = model.predict(fen)

    _, prior_values = zip(*prior)
    prior_values = np.sort(np.array(prior_values))

    # Find how the prior values are scaled
    factor = (CHECKMATE_VALUE + PRIOR_OFFSET) / prior_values[-1]

    original_values = prior_values * factor - PRIOR_OFFSET

    assert original_values[-2] == pytest.approx(PIECE_TO_VALUE[chess.KNIGHT])
    assert original_values[-3] == pytest.approx(PIECE_TO_VALUE[chess.PAWN])
    assert original_values[-4] == pytest.approx(0)
    assert original_values[0] == pytest.approx(0)
