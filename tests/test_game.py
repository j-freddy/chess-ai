from unittest import mock

import chess
import pytest

from chess_ai.game import Game, IllegalMoveError
from chess_ai.players import AIRandom, Human
from tests.conftest import CHEAT_MOVE, CheatingPlayer


@pytest.mark.parametrize("user_inputs", [["f3", "e5", "g4", "Qh4"]])
def test_two_humans_can_play_a_game(user_inputs):
    with mock.patch("builtins.input", side_effect=user_inputs):
        game = Game(player_white=Human(), player_black=Human())
        assert game.play() == "0-1"


@pytest.mark.parametrize(
    "user_inputs", [["f3", "e5", "g4", "foo", "Qh5", "Qh4"]]
)
def test_two_humans_can_play_a_game_with_invalid_moves(user_inputs, capsys):
    with mock.patch("builtins.input", side_effect=user_inputs):
        game = Game(player_white=Human(), player_black=Human())
        game.play()

    captured = capsys.readouterr()
    assert "Invalid move, try again." in captured.out
    assert "Illegal move, try again." in captured.out


def test_two_ais_can_play_a_game():
    game = Game(player_white=AIRandom(), player_black=AIRandom())
    assert game.play() in ("1-0", "0-1", "1/2-1/2")


def test_current_player_follows_the_side_to_move():
    white, black = AIRandom(), AIRandom()
    game = Game(player_white=white, player_black=black)

    assert game.current_player is white
    game.board.push_san("e4")
    assert game.current_player is black


def test_game_stops_when_a_player_returns_an_illegal_move():
    cheat = CheatingPlayer()
    game = Game(player_white=cheat, player_black=AIRandom())

    with pytest.raises(IllegalMoveError) as excinfo:
        game.play()

    assert excinfo.value.player is cheat
    assert excinfo.value.move == CHEAT_MOVE
    assert excinfo.value.state == chess.STARTING_FEN
    # The illegal move was not applied
    assert game.board.fen() == chess.STARTING_FEN
