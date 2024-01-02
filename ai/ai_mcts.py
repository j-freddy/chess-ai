from __future__ import annotations
import chess
import math
import numpy as np
from typing import TypeAlias

from player import Player

State: TypeAlias = str
Action: TypeAlias = chess.Move

def ucb_score(parent: Node, child: Node) -> float:
    prior_score = child.prior * math.sqrt(parent.visit_count) /\
        (child.visit_count + 1)
    value_score = 0 if child.visit_count == 0 else -child.value()
    return value_score + prior_score

class Node:
    def __init__(self, prior: float, current_player: chess.Color):
        self.prior = prior
        self.current_player = current_player
        self.children: dict[Action, Node] = {}
        self.num_visits = 0
        self.value_sum = 0.0
        # Board state represented as a FEN string
        self.state: State = None

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

        best_score = -np.inf
        best_action: Action = None
        best_child: Node = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child
    
    def expand(self, state: State, current_player: chess.Color, action_probs):
        """
        Expand this node and track the prior probability (e.g. given by a policy
        network).
        """

        self.current_player = current_player
        self.state = state

        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(
                    prior=prob,
                    # self.current_player is chess.Color which is a bool
                    current_player=self.current_player ^ 1
                )

class AIMCTS(Player):
    def backprop(self, search_path, value, current_player: chess.Color):
        return NotImplemented

    def run(
        self,
        state: State,
        current_player: chess.Color,
        num_simulations=10,
    )-> Node:
        """
        Perform Monte Carlo tree search: run @self.num_simulations simulations
        starting from board state @state.
        """

        root = Node(0, current_player)

        # Stage: EXPAND
        current_board = chess.Board(state)
        actions = list(current_board.legal_moves)
        action_probs = np.ones(len(actions)) / len(actions)
        root.expand(state, current_player, action_probs)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # Stage: SELECT
            while node.is_expanded():
                action, node = node.select_child()
                search_path.append(node)
            
            parent = search_path[-2]
            state = parent.state

            # TODO
            assert False

        return root

    def choose_move(self, position: str) -> str:
        root = self.run(position, self.color)

        return NotImplemented

if __name__ == "__main__":
    ai = AIMCTS(chess.WHITE)
    ai.choose_move(chess.STARTING_FEN)
