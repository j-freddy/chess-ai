import argparse
import logging

import chess

from chess_ai.game import Game, IllegalMoveError
from chess_ai.players import AI, AIMCTS, AIRandom, Human, Player

logger = logging.getLogger(__name__)

ID_TO_PLAYER_CLASS: dict[str, type[Player]] = {
    "human": Human,
    "airandom": AIRandom,
    "aimcts": AIMCTS,
}


class TimeBudgetForHumanError(ValueError):
    """
    Raised when a time budget is given for a human player.
    """


def build_player(player_id: str, time_budget: float | None) -> Player:
    player_class = ID_TO_PLAYER_CLASS[player_id]

    if not issubclass(player_class, AI):
        if time_budget is not None:
            raise TimeBudgetForHumanError(
                f"Cannot set a time budget for human player {player_id!r}."
            )
        return player_class()

    if time_budget is None:
        return player_class()

    return player_class(time_budget=time_budget)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    player_class_ids = ", ".join(ID_TO_PLAYER_CLASS)

    parser = argparse.ArgumentParser(prog="python -m chess_ai")
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
        "-whitetime",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Time budget per move for White, in seconds. Only valid for AI players.",
    )
    parser.add_argument(
        "-blacktime",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Time budget per move for Black, in seconds. Only valid for AI players.",
    )
    parser.add_argument(
        "-startpos",
        type=str,
        help="Starting position in FEN. Default: standard position.",
        default=chess.STARTING_FEN,
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)

    try:
        player_white = build_player(args.white, args.whitetime)
        player_black = build_player(args.black, args.blacktime)
    except TimeBudgetForHumanError as err:
        logger.info(f"Argument error: {err}")
        return 1

    game = Game(
        player_white=player_white,
        player_black=player_black,
        start_pos=args.startpos,
        player_white_id=args.white,
        player_black_id=args.black,
    )

    try:
        game.play()
    except IllegalMoveError as err:
        logger.info(f"Game stopped: {err}")
        return 1

    return 0
