import chess

from ai.ai_mcts import AIMCTS
from ai.ai_random import AIRandom
from game import Game
from player_human import PlayerHuman

if __name__=="__main__":
    game = Game(
        player_white=AIRandom(chess.WHITE),
        player_black=AIMCTS(chess.BLACK),
    )
    game.play()
