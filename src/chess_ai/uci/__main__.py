import chess

from chess_ai.players import AIMCTS
from chess_ai.uci.protocol import service_uci_command


def main() -> None:
    board = chess.Board()
    ai = AIMCTS()

    while True:
        service_uci_command(
            command=input().strip(),
            board=board,
            ai=ai,
        )


if __name__ == "__main__":
    main()
