import chess

from chess_ai.chess_types import Action, State
from chess_ai.players import Player


class IllegalMoveError(Exception):
    """
    Raised when a player breaks the choose_move contract by returning a move
    that is not legal in the position it was given.
    """

    def __init__(self, player: Player, move: Action, state: State):
        self.player = player
        self.move = move
        self.state = state

        super().__init__(
            f"{player} returned an illegal move ({move.uci()}) in {state}"
        )


class Game:
    def __init__(
        self,
        player_white: Player,
        player_black: Player,
        start_pos: str = chess.STARTING_FEN,
    ):
        self.board = chess.Board(start_pos)
        self.player_white = player_white
        self.player_black = player_black

    @property
    def current_player(self) -> Player:
        return self.player_white if self.board.turn else self.player_black

    def play(self) -> str:
        """
        Play until the game ends, and return the result.

        Raises IllegalMoveError if a player returns a move that is not legal.
        """

        print(self.board)

        while not self.board.is_game_over():
            player = self.current_player
            state = self.board.fen()
            move = player.choose_move(state)

            if move not in self.board.legal_moves:
                raise IllegalMoveError(player, move, state)

            print(f"{player} plays {self.board.san(move)}")
            self.board.push(move)
            print(self.board)

        result = self.board.result()
        print(f"Game over: {result}")
        return result
