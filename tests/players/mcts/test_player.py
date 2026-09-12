import chess
import pytest

from chess_ai.players import AIMCTS


def test_ai_mcts_chooses_a_legal_move():
    ai = AIMCTS()
    board = chess.Board()

    assert ai.choose_move(board.fen()) in board.legal_moves


def test_ai_mcts_identifies_itself():
    ai = AIMCTS()

    assert ai.NAME == "MCTSBot"
    assert str(ai) == "MCTSBot"


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
def test_optimal_move_from_prior(fen: str, move: str):
    assert AIMCTS()._optimal_move_from_prior(fen).uci() == move


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
def test_find_mate_in_one(fen: str, move: str | None):
    ai = AIMCTS()

    if move is not None:
        assert ai._check_for_mate(fen) == chess.Move.from_uci(move)
    else:
        assert ai._check_for_mate(fen) is None


def test_search_without_playouts_is_not_supported():
    with pytest.raises(NotImplementedError):
        AIMCTS().run(chess.STARTING_FEN, time_budget=0.1, num_playouts=0)
