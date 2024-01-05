# pylint: disable=protected-access

import chess
import numpy as np
import pytest

from ai.ai_mcts import (
    AIMCTS,
    CHECKMATE_VALUE,
    piece_to_value,
    result,
    statically_score_move,
)

@pytest.mark.parametrize("fen, value", [
    (chess.Board.starting_fen, None),
    ("5k1r/6b1/p2BQ3/3Pp1p1/P3Pp2/8/4KPP1/1q6 b - - 0 35", 1.0),
    ("3r2k1/p4ppp/Q7/3p4/1N6/2N5/PP3nPP/R5RK w - - 1 29", -1.0),
    ("7k/8/6Q1/3BK3/8/8/8/8 b - - 20 81", 0.0),
])
def test_result_of_board_is_correctly_encoded(fen, value):
    board = chess.Board(fen)
    assert result(board) == value

# TODO
@pytest.mark.skip
def test_ucb_score():
    pass

# pylint: disable=line-too-long
@pytest.mark.parametrize("move, fen, score", [
    ("a1a8", "4k3/8/4K3/8/8/8/8/R7 w - - 0 1", CHECKMATE_VALUE),
    ("d8d5", "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2", piece_to_value[chess.PAWN]),
    ("d3e4", "r3kb1r/ppqn1ppp/2p1p3/8/3Pn3/3Q1N1P/PPP2PP1/R1B2RK1 w kq - 0 12", piece_to_value[chess.KNIGHT]),
    ("f5d3", "rn2kb1r/ppq2ppp/2p1pn2/5b2/3P4/2NB1N1P/PPP2PP1/R1BQ1RK1 b kq - 1 9", piece_to_value[chess.BISHOP]),
    ("c5d6", "2k2r2/pp2qp2/2prpn1p/2P3p1/3P4/7P/PP2QPP1/R2R2K1 w - - 0 23", piece_to_value[chess.ROOK]),
    ("d6e5", "2k2r2/pp3p2/2pqpn1p/4Q1p1/3P4/7P/PP3PP1/R2R2K1 b - - 1 24", piece_to_value[chess.QUEEN]),
])
def test_statically_score_move(move, fen, score):
    assert statically_score_move(chess.Move.from_uci(move), fen) == score

def test_compute_prior():
    # Custom position
    # Best move: mate
    # Then: capture knight followed by capture pawn according to static
    # evaluator
    fen = "rnbq1b1r/pp1pk3/3n2Q1/5p2/8/8/PP3PPP/RNB1KB1R w KQ - 0 13"

    ai = AIMCTS(chess.WHITE)
    prior = ai._compute_prior(fen)

    _, prior_values = zip(*prior)
    prior_values = np.sort(np.array(prior_values))

    # Find how the prior values are scaled
    factor = (CHECKMATE_VALUE + ai._prior_offset) / prior_values[-1]

    original_values = prior_values * factor - ai._prior_offset

    assert original_values[-2] == pytest.approx(piece_to_value[chess.KNIGHT])
    assert original_values[-3] == pytest.approx(piece_to_value[chess.PAWN])
    assert original_values[-4] == pytest.approx(0)
    assert original_values[0] == pytest.approx(0)

@pytest.mark.parametrize("fen, move", [
    ("rnbq1b1r/pp1pk3/3n2Q1/5p2/8/8/PP3PPP/RNB1KB1R w KQ - 0 13", "c1g5"),
    ("r1bqkbnr/pp1p1ppp/2n5/4p3/3PP3/8/PP3PPP/RNBQKBNR w KQkq - 0 5", "d4e5"),
    ("rnbqkb1r/pp1p4/6p1/3P1p1Q/4n3/8/PP3PPP/RNB1KB1R w KQkq - 0 11", "h5h8"),
])
def test_optimal_move_from_prior(fen, move):
    ai = AIMCTS(chess.WHITE)
    assert ai._optimal_move_from_prior(fen).uci() == move
