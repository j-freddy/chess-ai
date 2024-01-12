"""
Implement the UCI protocol as publiced by Stefan-Meyer Kahlen (ShredderChess)

See uci-protocol.txt. Summary below.

GUI to engine:
    [DONE] uci
    [SKIP] debug [ on | off ]
    isready
    [SKIP] setoption name  [value ]
    [SKIP] register
    [DONE] ucinewgame
    [DONE] position [fen  | startpos ]  moves  ....
    go ....
    [SKIP] stop
    [SKIP] ponderhit
    [DONE] quit

Engine to GUI:
    [DONE] id name author
    [DONE] uciok
    [DONE] readyok
    [DONE] bestmove  [ ponder  ]
    [SKIP] copyprotection
    [SKIP] registration
    [SKIP] info ....
    [SKIP] option ....
"""

import sys
import chess

from ai.ai_mcts import AIMCTS
from player import Player

# pylint: disable=redefined-outer-name
def service_uci_command(command: str, board: chess.Board, ai: Player):
    tokens = command.split()

    match tokens[0]:
        case "uci":
            print("id name MirroredBot")
            print("id author Freddy Jiang")
            print("uciok")

        case "isready":
            print("readyok")

        case "ucinewgame":
            board.reset()

        # position [fen  | startpos ]  moves  ....
        case "position":
            match tokens[1]:
                case "fen":
                    # FEN has 6 words
                    fen = "".join(tokens[2:8])
                    board.set_fen(fen)
                    tokens_moves = tokens[9:]
                case "startpos":
                    board.reset()
                    tokens_moves = tokens[3:]
                case _:
                    raise ValueError("Invalid position command")

            for move in tokens_moves:
                board.push_uci(move)

        # go [searchmoves  ....] ponder wtime btime winc binc movestogo depth
        # nodes mate movetime infinite
        case "go":
            move = ai.choose_move(board.fen())
            print(f"bestmove {move}")

        case "quit":
            sys.exit()

if __name__=="__main__":
    board = chess.Board()
    ai = AIMCTS()

    while True:
        service_uci_command(
            command=input().strip(),
            board=board,
            ai=ai,
        )
