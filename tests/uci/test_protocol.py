import time

import chess
import pytest

from chess_ai.players import AIMCTS, AIRandom
from chess_ai.uci.protocol import (
    INFINITE_TIME_BUDGET,
    _time_budget_for_go,
    service_uci_command,
)

OPENING_FEN = (
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
)


def test_uci_reports_engine_identity(capsys: pytest.CaptureFixture[str]):
    service_uci_command("uci", chess.Board(), AIMCTS())
    out = capsys.readouterr().out

    assert "id name MCTSBot" in out
    assert "id author Freddy Jiang" in out
    assert "uciok" in out


def test_isready(capsys: pytest.CaptureFixture[str]):
    service_uci_command("isready", chess.Board(), AIRandom())

    assert "readyok" in capsys.readouterr().out


def test_ucinewgame_resets_the_board():
    board = chess.Board()
    board.push_san("e4")

    service_uci_command("ucinewgame", board, AIRandom())

    assert board.fen() == chess.STARTING_FEN


@pytest.mark.parametrize("moves", ["", " moves f8c5", " moves f8c5 e1g1"])
def test_position_fen_sets_the_board_and_applies_moves(moves: str):
    board = chess.Board()
    service_uci_command(f"position fen {OPENING_FEN}{moves}", board, AIRandom())

    expected = chess.Board(OPENING_FEN)
    for move in moves.split()[1:]:
        expected.push_uci(move)

    assert board.fen() == expected.fen()


@pytest.mark.parametrize("moves", ["", " moves e2e4", " moves e2e4 e7e5"])
def test_position_startpos_sets_the_board_and_applies_moves(moves: str):
    board = chess.Board(OPENING_FEN)
    service_uci_command(f"position startpos{moves}", board, AIRandom())

    expected = chess.Board()
    for move in moves.split()[1:]:
        expected.push_uci(move)

    assert board.fen() == expected.fen()


def test_position_rejects_an_unknown_subcommand():
    with pytest.raises(ValueError):
        service_uci_command("position elsewhere", chess.Board(), AIRandom())


def test_go_prints_a_legal_bestmove(capsys: pytest.CaptureFixture[str]):
    board = chess.Board()
    service_uci_command("go", board, AIRandom())

    _, uci = capsys.readouterr().out.split()

    assert chess.Move.from_uci(uci) in board.legal_moves


def test_empty_command_is_ignored():
    service_uci_command("   ", chess.Board(), AIRandom())


def test_quit_exits():
    with pytest.raises(SystemExit):
        service_uci_command("quit", chess.Board(), AIRandom())


FINISHED_POSITIONS = [
    # Checkmate
    "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1",
    # Stalemate
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
]


@pytest.mark.parametrize("fen", FINISHED_POSITIONS)
def test_go_reports_a_null_move_when_the_game_is_over(
    fen: str, capsys: pytest.CaptureFixture[str]
):
    service_uci_command("go", chess.Board(fen), AIMCTS())

    assert capsys.readouterr().out.strip() == "bestmove 0000"


def test_go_writes_nothing_but_bestmove_to_stdout(
    capsys: pytest.CaptureFixture[str],
):
    service_uci_command("go movetime 50", chess.Board(), AIMCTS())
    out = capsys.readouterr().out.strip().splitlines()

    assert len(out) == 1
    assert out[0].startswith("bestmove ")


@pytest.mark.parametrize(
    "command, turn, budget",
    [
        # No clock information, so fall back to the engine's own budget
        ("go", chess.WHITE, 5.0),
        ("go depth 12", chess.WHITE, 5.0),
        ("go ponder", chess.WHITE, 5.0),
        # An explicit per-move time wins over everything else
        ("go movetime 250", chess.WHITE, 0.25),
        ("go movetime 250 wtime 60000", chess.WHITE, 0.25),
        # Spread the remaining clock over the assumed moves left
        ("go wtime 60000 btime 30000", chess.WHITE, 2.0),
        ("go wtime 60000 btime 30000", chess.BLACK, 1.0),
        # movestogo overrides the estimate
        ("go wtime 60000 btime 60000 movestogo 5", chess.WHITE, 12.0),
        # Most of the increment is spendable on top of the slice
        ("go wtime 30000 btime 30000 winc 5000 binc 5000", chess.WHITE, 5.0),
        # Never sink the whole clock into one move
        ("go wtime 1000 btime 1000 movestogo 1", chess.WHITE, 0.8),
        # An unbounded search is not serviceable without "stop"
        ("go infinite", chess.WHITE, INFINITE_TIME_BUDGET),
    ],
)
def test_time_budget_for_go(command: str, turn: chess.Color, budget: float):
    board = chess.Board()
    board.turn = turn

    assert _time_budget_for_go(command.split(), board, 5.0) == pytest.approx(
        budget
    )


def test_go_restores_the_configured_time_budget():
    ai = AIMCTS(time_budget=5.0)
    service_uci_command("go movetime 50", chess.Board(), ai)

    assert ai.time_budget == 5.0


def test_go_honours_movetime():
    ai = AIMCTS(time_budget=30.0)
    start = time.time()
    service_uci_command("go movetime 100", chess.Board(), ai)

    assert time.time() - start < 5.0
