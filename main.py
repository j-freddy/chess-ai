import chess

from ai_random import AIRandom
from game import Game
from player_human import PlayerHuman

if __name__=="__main__":
    game = Game(
        player_white=PlayerHuman(chess.WHITE),
        player_black=AIRandom(chess.BLACK),
    )
    game.play()
