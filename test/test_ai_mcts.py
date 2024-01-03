import chess
import pytest

from ai.ai_mcts import result

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
