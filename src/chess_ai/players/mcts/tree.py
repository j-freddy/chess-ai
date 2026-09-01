from __future__ import annotations

import math
from collections.abc import Sequence

import chess

from chess_ai.chess_types import Action, State


def outcome_value(board: chess.Board) -> float | None:
    """
    Return the result of a finished game from White's perspective, or None if
    the game is still in progress.
    """

    match board.result():
        case "1-0":
            return 1.0
        case "0-1":
            return -1.0
        case "1/2-1/2":
            return 0.0
        case _:
            return None


def ucb_score(parent: Node, child: Node) -> float:
    prior_score = (
        child.prior * math.sqrt(parent.num_visits) / (child.num_visits + 1)
    )
    value_score = 0.0 if child.num_visits == 0 else -child.value()
    return value_score + prior_score


class Node:
    def __init__(self, prior: float, current_player: chess.Color):
        self.prior = prior
        self.current_player = current_player
        self.children: dict[Action, Node] = {}
        self.num_visits = 0
        self.value_sum = 0.0
        # Board state represented as a FEN string
        self.state: State | None = None

    def value(self) -> float:
        if self.num_visits == 0:
            return 0.0
        return self.value_sum / self.num_visits

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self) -> tuple[Action, Node]:
        """
        Select the child with the highest UCB score.
        """

        if not self.children:
            raise ValueError("Cannot select a child of an unexpanded node")

        return max(
            self.children.items(),
            key=lambda item: ucb_score(self, item[1]),
        )

    def expand(
        self,
        state: State,
        actions: Sequence[Action],
        action_probs: Sequence[float],
    ) -> None:
        """
        Expand this node and track the prior probability (e.g. given by a policy
        network).
        """

        self.state = state

        for action, prob in zip(actions, action_probs, strict=True):
            if prob != 0:
                self.children[action] = Node(
                    prior=prob,
                    # self.current_player is chess.Color which is a bool
                    current_player=self.current_player ^ True,
                )

    def __str__(self) -> str:
        sb = f"{self!r}\n"
        sb += str(chess.Board(self.state))

        for action, child in self.children.items():
            sb += f"\n{action}: {child!r}"

        return sb

    def __repr__(self) -> str:
        return (
            f"Node(prior={self.prior}, "
            f"current_player={self.current_player}, "
            f"num_visits={self.num_visits}, "
            f"value_sum={self.value_sum}, "
            f"state={self.state})"
        )
