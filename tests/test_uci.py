import chess
import pytest

from chess_ai.players import AIMCTS, AIRandom
from chess_ai.uci.protocol import service_uci_command

OPENING_FEN = (
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
)


def test_uci_reports_engine_identity(capsys):
    service_uci_command("uci", chess.Board(), AIMCTS())
    out = capsys.readouterr().out

    assert "id name MirroredBot" in out
    assert "id author Freddy Jiang" in out
    assert "uciok" in out


def test_isready(capsys):
    service_uci_command("isready", chess.Board(), AIRandom())
    assert "readyok" in capsys.readouterr().out


def test_ucinewgame_resets_the_board():
    board = chess.Board()
    board.push_san("e4")

    service_uci_command("ucinewgame", board, AIRandom())

    assert board.fen() == chess.STARTING_FEN


@pytest.mark.parametrize("moves", ["", " moves f8c5", " moves f8c5 e1g1"])
def test_position_fen_sets_the_board_and_applies_moves(moves):
    board = chess.Board()
    service_uci_command(f"position fen {OPENING_FEN}{moves}", board, AIRandom())

    expected = chess.Board(OPENING_FEN)
    for move in moves.split()[1:]:
        expected.push_uci(move)

    assert board.fen() == expected.fen()


@pytest.mark.parametrize("moves", ["", " moves e2e4", " moves e2e4 e7e5"])
def test_position_startpos_sets_the_board_and_applies_moves(moves):
    board = chess.Board(OPENING_FEN)
    service_uci_command(f"position startpos{moves}", board, AIRandom())

    expected = chess.Board()
    for move in moves.split()[1:]:
        expected.push_uci(move)

    assert board.fen() == expected.fen()


def test_position_rejects_an_unknown_subcommand():
    with pytest.raises(ValueError):
        service_uci_command("position elsewhere", chess.Board(), AIRandom())


def test_go_prints_a_legal_bestmove(capsys):
    board = chess.Board()
    service_uci_command("go", board, AIRandom())

    _, uci = capsys.readouterr().out.split()
    assert chess.Move.from_uci(uci) in board.legal_moves


def test_empty_command_is_ignored():
    service_uci_command("   ", chess.Board(), AIRandom())


def test_quit_exits():
    with pytest.raises(SystemExit):
        service_uci_command("quit", chess.Board(), AIRandom())
