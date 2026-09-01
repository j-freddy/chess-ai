import pytest

from chess_ai import cli
from tests.conftest import CheatingPlayer


def test_two_ais_can_play_from_the_command_line():
    assert cli.main(["-white", "airandom", "-black", "airandom"]) == 0


def test_cli_reports_an_illegal_move_instead_of_crashing(monkeypatch, capsys):
    monkeypatch.setitem(cli.ID_TO_PLAYER_CLASS, "cheat", CheatingPlayer)

    assert cli.main(["-white", "cheat", "-black", "airandom"]) == 1
    assert "Game stopped: CheatingPlayer returned an illegal move (e2e5)" in (
        capsys.readouterr().out
    )


def test_cli_rejects_an_unknown_player():
    with pytest.raises(SystemExit):
        cli.main(["-white", "nosuchplayer", "-black", "airandom"])
