import chess
import math
import numpy as np
from typing import TypeAlias

from player import Player

State: TypeAlias = str
Action: TypeAlias = chess.Move

class Node:
    def __init__(self, prior: float, current_player: chess.Color):
        self.prior = prior
        self.current_player = current_player
        self.children: dict[Action, Node] = {}
        self.num_visits = 0
        self.value_sum = 0.0
        # Board state represented as a FEN string
        self.board_state: State = None

    def value(self) -> float:
        if self.num_visits == 0:
            return 0.0
        return self.value_sum / self.num_visits

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self):
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
    
    def expand(self, state, current_player, action_probs):
        """
        Expand this node and track the prior probability given by the policy
        network
        """

        return NotImplemented
    
def ucb_score(parent: Node, child: Node) -> float:
    prior_score = child.prior * math.sqrt(parent.visit_count) /\
        (child.visit_count + 1)
    value_score = 0 if child.visit_count == 0 else -child.value()
    return value_score + prior_score

class AIMCTS(Player):
    def choose_move(self, position: str) -> str:
        return NotImplemented
