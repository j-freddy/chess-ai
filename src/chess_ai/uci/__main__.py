import chess

from chess_ai.players import AIMCTS
from chess_ai.uci.protocol import service_uci_command


def main() -> None:
    board = chess.Board()
    ai = AIMCTS()

    while True:
        try:
            command = input()
        except (EOFError, KeyboardInterrupt):
            # The GUI closed the pipe without sending "quit"
            return

        service_uci_command(
            command=command.strip(),
            board=board,
            ai=ai,
        )


if __name__ == "__main__":
    main()
