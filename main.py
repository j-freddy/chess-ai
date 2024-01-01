import chess

if __name__=="__main__":
    board = chess.Board()

    while not board.is_game_over():
        print(board)

        is_valid_move = False

        while not is_valid_move:
            move = input("Enter your move: ")

            try:
                board.push_san(move)
                is_valid_move = True
            except chess.InvalidMoveError:
                print("Invalid move, try again.")
            except chess.IllegalMoveError:
                print("Illegal move, try again.")
            except chess.AmbiguousMoveError:
                print("Ambiguous move, try again.")
