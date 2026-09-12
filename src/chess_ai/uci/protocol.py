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
    [DONE] go .... (movetime, wtime/btime/winc/binc/movestogo and infinite are
           honoured; searchmoves, ponder, depth, nodes and mate are ignored)
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
from itertools import pairwise

import chess

from chess_ai.players import AI

# A FEN is made up of 6 space-separated fields
NUM_FEN_FIELDS = 6

# UCI has no way to say "there is no move", so a null move reports a position
# that is already over.
NULL_MOVE_UCI = "0000"

# "go" parameters that carry an integer value
GO_INT_PARAMS = frozenset(
    {
        "wtime",
        "btime",
        "winc",
        "binc",
        "movestogo",
        "movetime",
        "depth",
        "nodes",
        "mate",
    }
)

# How many more moves to assume are left when the GUI does not say
MOVES_TO_GO_ESTIMATE = 30
# Share of the increment to spend on top of the per-move slice
INCREMENT_FRACTION = 0.8
# Never sink more than this share of the remaining clock into one move
MAX_CLOCK_FRACTION = 0.8
# The engine is single threaded and cannot service "stop" mid-search, so treat
# "go infinite" as a long search rather than an endless one
INFINITE_TIME_BUDGET = 60.0
MIN_TIME_BUDGET = 0.05


def _parse_go_params(tokens: list[str]) -> dict[str, int]:
    """
    Collect the integer-valued parameters of a "go" command.
    """

    params: dict[str, int] = {}

    for keyword, value in pairwise(tokens):
        if keyword not in GO_INT_PARAMS:
            continue

        try:
            params[keyword] = int(value)
        except ValueError:
            continue

    return params


def _time_budget_for_go(
    tokens: list[str], board: chess.Board, default: float
) -> float:
    """
    Return how many seconds to think for, given a "go" command. Clock values
    sent by the GUI are in milliseconds.
    """

    params = _parse_go_params(tokens)

    if "movetime" in params:
        return max(MIN_TIME_BUDGET, params["movetime"] / 1000)

    is_white = board.turn == chess.WHITE
    clock_key = "wtime" if is_white else "btime"

    if clock_key in params:
        remaining = params[clock_key] / 1000
        increment = params.get("winc" if is_white else "binc", 0) / 1000
        moves_to_go = max(1, params.get("movestogo", MOVES_TO_GO_ESTIMATE))

        budget = remaining / moves_to_go + increment * INCREMENT_FRACTION

        return max(MIN_TIME_BUDGET, min(budget, remaining * MAX_CLOCK_FRACTION))

    if "infinite" in tokens:
        return INFINITE_TIME_BUDGET

    return default


def _service_go(tokens: list[str], board: chess.Board, ai: AI) -> None:
    """
    Service: go ....
    """

    if not board.legal_moves:
        print(f"bestmove {NULL_MOVE_UCI}", flush=True)
        return

    configured_budget = ai.time_budget
    ai.time_budget = _time_budget_for_go(tokens, board, configured_budget)

    try:
        move = ai.choose_move(board.fen())
    finally:
        ai.time_budget = configured_budget

    print(f"bestmove {move.uci()}", flush=True)


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
            print(f"id name {ai.NAME}", flush=True)
            print(f"id author {ai.AUTHOR}", flush=True)
            print("uciok", flush=True)

        case "isready":
            print("readyok", flush=True)

        case "ucinewgame":
            board.reset()

        case "position":
            _set_position(board, tokens)

        # go [searchmoves  ....] ponder wtime btime winc binc movestogo depth
        # nodes mate movetime infinite
        case "go":
            _service_go(tokens, board, ai)

        case "quit":
            sys.exit()
