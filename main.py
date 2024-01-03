import argparse
import chess

from ai.ai_mcts import AIMCTS
from ai.ai_random import AIRandom
from game import Game
from player_human import PlayerHuman

id_to_player_class = {
    "human": PlayerHuman,
    "airandom": AIRandom,
    "aimcts": AIMCTS,
}

def player_class_ids() -> str:
    return ", ".join(id_to_player_class.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-white",
        type=str,
        help=f"White player. Options: {player_class_ids()}"
    )
    parser.add_argument(
        "-black",
        type=str,
        help=f"Black player. Options: {player_class_ids()}",
    )

    args = parser.parse_args()

    player_white_class = id_to_player_class[args.white]
    player_black_class = id_to_player_class[args.black]

    return player_white_class, player_black_class

if __name__=="__main__":
    PlayerWhite, PlayerBlack = parse_args()

    game = Game(
        player_white=PlayerWhite(chess.WHITE),
        player_black=PlayerBlack(chess.BLACK),
    )
    game.play()
