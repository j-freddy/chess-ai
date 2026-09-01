from unittest import mock

import chess
import pytest

from chess_ai.players import Human

AMBIGUOUS_FEN = "4k3/8/8/8/8/2N5/8/4K1N1 w - - 0 1"


@pytest.mark.parametrize(
    "entered, expected_uci",
    [("e4", "e2e4"), ("e2e4", "e2e4"), ("Nf3", "g1f3")],
)
def test_human_returns_the_entered_move(entered, expected_uci):
    with mock.patch("builtins.input", return_value=entered):
        move = Human().choose_move(chess.STARTING_FEN)

    assert move.uci() == expected_uci


@pytest.mark.parametrize(
    "state, entered, message, recovery",
    [
        (chess.STARTING_FEN, "foo", "Invalid move, try again.", "e4"),
        (chess.STARTING_FEN, "e5", "Illegal move, try again.", "e4"),
        (AMBIGUOUS_FEN, "Ne2", "Ambiguous move, try again.", "Nce2"),
    ],
)
def test_human_reprompts_until_the_move_is_legal(
    state, entered, message, recovery, capsys
):
    with mock.patch("builtins.input", side_effect=[entered, recovery]):
        move = Human().choose_move(state)

    assert move == chess.Board(state).parse_san(recovery)
    assert message in capsys.readouterr().out
