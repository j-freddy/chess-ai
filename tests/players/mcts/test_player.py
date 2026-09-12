import random

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


# White to move mates with a1a8; the mirror has Black mating with a8a1. Each
# side has 20 legal moves, so the search has plenty of alternatives to prefer
# if it scores the mate wrongly.
MATE_IN_ONE_WHITE = ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8")
MATE_IN_ONE_BLACK = ("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1")


@pytest.mark.parametrize("fen, uci", [MATE_IN_ONE_WHITE, MATE_IN_ONE_BLACK])
def test_mate_is_scored_as_a_win_for_the_side_delivering_it(fen: str, uci: str):
    # Terminal positions are scored from White's perspective and have to be
    # flipped into the leaf player's perspective before being backed up.
    # Without the flip, mating is recorded as a win for the mated player.
    root, _ = AIMCTS().run(fen, time_budget=1.0)
    mating_child = root.children[chess.Move.from_uci(uci)]

    assert mating_child.num_visits > 0
    # The child scores from the mated player's perspective, so a loss for them
    assert mating_child.value() == -1.0


@pytest.mark.parametrize("fen, uci", [MATE_IN_ONE_WHITE, MATE_IN_ONE_BLACK])
def test_search_finds_mate_without_the_mate_in_one_short_circuit(
    fen: str, uci: str
):
    root, _ = AIMCTS().run(fen, time_budget=1.0)

    assert root.select_best_action() == chess.Move.from_uci(uci)


def test_choose_move_finds_mate_in_one(capsys: pytest.CaptureFixture[str]):
    fen, uci = MATE_IN_ONE_WHITE

    assert AIMCTS().choose_move(fen) == chess.Move.from_uci(uci)
    # Engine chatter must stay off stdout, which belongs to the UCI channel
    assert capsys.readouterr().out == ""


def test_playouts_are_stochastic():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    ai = AIMCTS(rng=random.Random(0))

    # A greedy playout would return the same value every time, making the
    # Monte Carlo average over repeated playouts meaningless
    assert len({ai.playout(fen) for _ in range(12)}) > 1


def test_playouts_are_reproducible_for_a_given_seed():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

    first = [AIMCTS(rng=random.Random(7)).playout(fen) for _ in range(3)]
    second = [AIMCTS(rng=random.Random(7)).playout(fen) for _ in range(3)]

    assert first == second


def test_run_rejects_a_position_with_no_legal_moves():
    with pytest.raises(ValueError):
        AIMCTS().run("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", time_budget=0.1)
