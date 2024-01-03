import chess
import pytest

from ai.ai_random import AIRandom

@pytest.mark.parametrize("AI", [AIRandom])
# pylint: disable=invalid-name
def test_ai_chooses_legal_move(AI):
    ai = AI(chess.WHITE)
    board = chess.Board()
    move = ai.choose_move(board.fen())
    assert chess.Move.from_uci(move) in board.legal_moves
