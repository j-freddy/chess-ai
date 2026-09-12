import chess

from chess_ai.players import AIRandom


def test_ai_random_chooses_a_legal_move():
    ai = AIRandom()
    board = chess.Board()

    assert ai.choose_move(board.fen()) in board.legal_moves


def test_ai_random_identifies_itself():
    ai = AIRandom()

    assert ai.NAME == "RandomBot"
    assert str(ai) == "RandomBot"
