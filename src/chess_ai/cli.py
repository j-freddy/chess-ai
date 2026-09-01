import argparse

import chess

from chess_ai.game import Game, IllegalMoveError
from chess_ai.players import AIMCTS, AIRandom, Human, Player

ID_TO_PLAYER_CLASS: dict[str, type[Player]] = {
    "human": Human,
    "airandom": AIRandom,
    "aimcts": AIMCTS,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    player_class_ids = ", ".join(ID_TO_PLAYER_CLASS)

    parser = argparse.ArgumentParser(prog="chess-ai")
    parser.add_argument(
        "-white",
        type=str,
        required=True,
        choices=ID_TO_PLAYER_CLASS,
        metavar="WHITE",
        help=f"White player. Options: {player_class_ids}",
    )
    parser.add_argument(
        "-black",
        type=str,
        required=True,
        choices=ID_TO_PLAYER_CLASS,
        metavar="BLACK",
        help=f"Black player. Options: {player_class_ids}",
    )
    parser.add_argument(
        "-startpos",
        type=str,
        help="Starting position in FEN. Default: standard position.",
        default=chess.STARTING_FEN,
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    game = Game(
        player_white=ID_TO_PLAYER_CLASS[args.white](),
        player_black=ID_TO_PLAYER_CLASS[args.black](),
        start_pos=args.startpos,
    )

    try:
        game.play()
    except IllegalMoveError as err:
        print(f"Game stopped: {err}")
        return 1

    return 0
