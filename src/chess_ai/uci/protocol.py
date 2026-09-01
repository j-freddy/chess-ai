"""
Implement the UCI protocol as published by Stefan-Meyer Kahlen (ShredderChess)

See docs/uci-protocol.txt. Summary below.

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

from chess_ai.players import AI

# A FEN is made up of 6 space-separated fields
NUM_FEN_FIELDS = 6


def _set_position(board: chess.Board, tokens: list[str]) -> None:
    """
    Service: position [fen  | startpos ]  moves  ....
    """

    if len(tokens) < 2:
        raise ValueError("Invalid position command: missing fen or startpos")

    match tokens[1]:
        case "fen":
            fen_end = 2 + NUM_FEN_FIELDS
            board.set_fen(" ".join(tokens[2:fen_end]))
            # Skip the "moves" keyword that follows the FEN
            tokens_moves = tokens[fen_end + 1 :]

        case "startpos":
            board.reset()
            # Skip the "moves" keyword that follows "startpos"
            tokens_moves = tokens[3:]

        case _:
            raise ValueError(f"Invalid position command: {' '.join(tokens)}")

    for move in tokens_moves:
        board.push_uci(move)


def service_uci_command(command: str, board: chess.Board, ai: AI) -> None:
    tokens = command.split()

    if not tokens:
        return

    match tokens[0]:
        case "uci":
            print(f"id name {ai.NAME}")
            print(f"id author {ai.AUTHOR}")
            print("uciok")

        case "isready":
            print("readyok")

        case "ucinewgame":
            board.reset()

        case "position":
            _set_position(board, tokens)

        # go [searchmoves  ....] ponder wtime btime winc binc movestogo depth
        # nodes mate movetime infinite
        case "go":
            move = ai.choose_move(board.fen())
            print(f"bestmove {move.uci()}")

        case "quit":
            sys.exit()
