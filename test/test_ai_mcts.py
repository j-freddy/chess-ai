# pylint: disable=protected-access

import chess
import pytest

from ai.ai_mcts import AIMCTS, result


@pytest.mark.parametrize(
    "fen, value",
    [
        (chess.Board.starting_fen, None),
        ("5k1r/6b1/p2BQ3/3Pp1p1/P3Pp2/8/4KPP1/1q6 b - - 0 35", 1.0),
        ("3r2k1/p4ppp/Q7/3p4/1N6/2N5/PP3nPP/R5RK w - - 1 29", -1.0),
        ("7k/8/6Q1/3BK3/8/8/8/8 b - - 20 81", 0.0),
    ],
)
def test_result_of_board_is_correctly_encoded(fen, value):
    board = chess.Board(fen)
    assert result(board) == value


# TODO
@pytest.mark.skip
def test_ucb_score():
    pass


@pytest.mark.parametrize(
    "fen, move",
    [
        ("rnbq1b1r/pp1pk3/3n2Q1/5p2/8/8/PP3PPP/RNB1KB1R w KQ - 0 13", "c1g5"),
        (
            "r1bqkbnr/pp1p1ppp/2n5/4p3/3PP3/8/PP3PPP/RNBQKBNR w KQkq - 0 5",
            "d4e5",
        ),
        (
            "rnbqkb1r/pp1p4/6p1/3P1p1Q/4n3/8/PP3PPP/RNB1KB1R w KQkq - 0 11",
            "h5h8",
        ),
    ],
)
def test_optimal_move_from_prior(fen, move):
    ai = AIMCTS(chess.WHITE)
    assert ai._optimal_move_from_prior(fen).uci() == move


@pytest.mark.parametrize(
    "fen, move",
    [
        (
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
            "d8h4",
        ),
        ("8/5Qbk/p3p1pp/1p2P3/1P3P1P/3q4/3R1P1r/4K3 b - - 13 44", "h2h1"),
        ("8/3r1pbk/p3p1pp/1p1qP3/1P3P2/P3Q1B1/5P1P/4R1K1 b - - 6 33", None),
    ],
)
def test_find_mate_in_one(fen, move):
    ai = AIMCTS(chess.WHITE)

    if move is not None:
        assert ai._check_for_mate(fen) == chess.Move.from_uci(move)
    else:
        assert ai._check_for_mate(fen) is None
