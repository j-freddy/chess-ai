from __future__ import annotations
import chess
import math
import numpy as np
from typing import TypeAlias

from player import Player

State: TypeAlias = str
Action: TypeAlias = chess.Move

def result(board: chess.Board) -> float:
    r = board.result()

    if r == "1-0":
        return 1.0
    elif r == "0-1":
        return -1.0
    elif r == "1/2-1/2":
        return 0.0
    
    return None

def ucb_score(parent: Node, child: Node) -> float:
    prior_score = child.prior * math.sqrt(parent.num_visits) /\
        (child.num_visits + 1)
    value_score = 0 if child.num_visits == 0 else -child.value()
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
    
    def expand(self,
        state: State,
        actions: list[Action],
        action_probs: np.ndarray[float],
    ):
        """
        Expand this node and track the prior probability (e.g. given by a policy
        network).
        """

        self.state = state

        for i in range(len(actions)):
            action, prob = actions[i], action_probs[i]

            if prob != 0:
                self.children[action] = Node(
                    prior=prob,
                    # self.current_player is chess.Color which is a bool
                    current_player=self.current_player ^ True
                )
    
    def __str__(self) -> str:
        sb = f"{repr(self)}\n"
        sb += str(chess.Board(self.state))

        for action, child in self.children.items():
            sb += f"\n{action}: {repr(child)}"

        return sb

    def __repr__(self) -> str:
        return f"Node(prior={self.prior}, current_player={self.current_player}, num_visits={self.num_visits}, value_sum={self.value_sum}, state={self.state})"

class AIMCTS(Player):
    def playout(self, state: State) -> float:
        """
        Play a random game from the given board state and return the result.
        """

        board = chess.Board(state)

        while not board.is_game_over():
            action = np.random.choice(list(board.legal_moves))
            board.push(action)

        return result(board)

    def run(
        self,
        state: State,
        current_player: chess.Color,
        num_simulations=100,
        num_playouts=1,
    ) -> Node:
        """
        Perform Monte Carlo tree search: run @self.num_simulations simulations
        starting from board state @state.
        """

        root = Node(0, current_player)

        # Stage: EXPAND
        current_board = chess.Board(state)
        actions = list(current_board.legal_moves)
        action_probs = np.ones(len(actions)) / len(actions)


        root.expand(state, actions, action_probs)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # Stage: SELECT
            while node.is_expanded():
                action, node = node.select_child()
                search_path.append(node)
            
            parent = search_path[-2]
            state = parent.state

            # We are now at a leaf node
            # Make a move
            board_at_leaf_node = chess.Board(state)
            board_at_leaf_node.push(action)
            next_state = board_at_leaf_node.fen()

            # Get value of next state from perspective of White
            value = result(board_at_leaf_node)

            if value is None:
                # Game has not ended
                # Stage: EXPAND
                actions = list(board_at_leaf_node.legal_moves)
                action_probs = np.ones(len(actions)) / len(actions)
                node.expand(next_state, actions, action_probs)
                
                # Stage: PLAYOUT
                acc_value = 0.0
                for _ in range(num_playouts):
                    acc_value += self.playout(next_state)
                
                value = acc_value / num_playouts

            # Get value from perspective of other player
            if node.current_player == chess.BLACK:
                value *= -1

            self.backprop(search_path, value, parent.current_player ^ True)

        return root

    def backprop(self, search_path, value, current_player: chess.Color):
        for node in reversed(search_path):
            node.value_sum += value if node.current_player == current_player\
                else -value
            node.num_visits += 1

    def choose_move(self, position: str) -> str:
        root = self.run(position, self.color, num_simulations=100, num_playouts=1)
        print(root)
        action, _ = root.select_child()
        return action.uci()
