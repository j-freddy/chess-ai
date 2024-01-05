import chess

from player import Player

class Game:
    def __init__(
        self,
        player_white: Player,
        player_black: Player,
        start_pos: str=chess.STARTING_FEN,
    ):
        self.board = chess.Board(start_pos)
        self.player_white = player_white
        self.player_black = player_black

        self.current_player = self.player_white if self.board.turn\
            else self.player_black

    def _make_move(self, move: str) -> bool:
        if self.current_player.is_ai:
            print(f"AI moves {move}")
            # Throws exception if move is invalid
            self.board.push_san(move)
            # Move must be valid if no exception thrown
            return True

        # Human player: validate move

        is_valid_move = False

        try:
            self.board.push_san(move)
            is_valid_move = True
        except chess.InvalidMoveError:
            print("Invalid move, try again.")
        except chess.IllegalMoveError:
            print("Illegal move, try again.")
        except chess.AmbiguousMoveError:
            print("Ambiguous move, try again.")

        return is_valid_move

    def play(self):
        print(self.board)

        while not self.board.is_game_over():
            is_valid_move = False

            while not is_valid_move:
                move = self.current_player.choose_move(self.board.fen())
                is_valid_move = self._make_move(move)

            # Switch players
            self.current_player = self.player_black\
                if self.current_player == self.player_white\
                else self.player_white
            assert self.current_player.color == self.board.turn

            print(self.board)

        print(f"Game over: {self.board.result()}")
