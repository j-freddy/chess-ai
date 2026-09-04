import chess
import pytest

from chess_ai.players import AIMCTS, AIRandom


@pytest.mark.parametrize("ai_class", [AIRandom, AIMCTS])
def test_ai_chooses_legal_move(ai_class):
    ai = ai_class()
    board = chess.Board()
    assert ai.choose_move(board.fen()) in board.legal_moves


@pytest.mark.parametrize(
    "ai_class, name", [(AIRandom, "RandomBot"), (AIMCTS, "MirroredBot")]
)
def test_ai_identifies_itself(ai_class, name):
    ai = ai_class()
    assert ai.NAME == name
    assert str(ai) == name
