from unittest import mock
import chess
import pytest

from game import Game
from player_human import PlayerHuman
from ai.ai_random import AIRandom

@pytest.mark.parametrize("user_input", ["e4", "e2e4", "invalidmove"])
def test_human_player_can_enter_input_move(user_input):
    with mock.patch("builtins.input", return_value=user_input):
        player = PlayerHuman(chess.WHITE)
        move = player.choose_move(chess.STARTING_FEN)
        assert move == user_input

@pytest.mark.parametrize("user_inputs", [["f3", "e5", "g4", "Qh4"]])
def test_two_humans_can_play_a_game(user_inputs):
    with mock.patch("builtins.input", side_effect=user_inputs):
        game = Game(
            player_white=PlayerHuman(chess.WHITE),
            player_black=PlayerHuman(chess.BLACK),
        )
        game.play()

@pytest.mark.parametrize("user_inputs", [["f3", "e5", "g4", "foo", "Qh5", "Qh4"]])
def test_two_humans_can_play_a_game_with_invalid_moves(user_inputs, capsys):
    with mock.patch("builtins.input", side_effect=user_inputs):
        game = Game(
            player_white=PlayerHuman(chess.WHITE),
            player_black=PlayerHuman(chess.BLACK),
        )
        game.play()

    captured = capsys.readouterr()
    assert "Invalid move, try again." in captured.out
    assert "Illegal move, try again." in captured.out

def test_two_ais_can_play_a_game():
    game = Game(
        player_white=AIRandom(chess.WHITE),
        player_black=AIRandom(chess.BLACK),
    )
    game.play()
