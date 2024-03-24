from __future__ import annotations
import math
import time
from typing import Optional
import chess
import numpy as np

from ai.model import Action, Model, State
from ai.model_naive import ModelNaive
from player import Player

def result(board: chess.Board) -> float:
    r = board.result()

    if r == "1-0":
        return 1.0
    if r == "0-1":
        return -1.0
    if r == "1/2-1/2":
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
        # pylint: disable=line-too-long
        return f"Node(prior={self.prior}, current_player={self.current_player}, num_visits={self.num_visits}, value_sum={self.value_sum}, state={self.state})"

class AIMCTS(Player):
    def __init__(
        self,
        color: chess.Color=chess.WHITE,
        model: Model=ModelNaive(),
    ):
        super().__init__(color)
        self.model = model

    def _check_for_mate(self, state: State) -> Optional[chess.Move]:
        """
        Return move that leads to checkmate if it exists.
        """

        board = chess.Board(state)
        actions = list(board.legal_moves)

        for action in actions:
            board.push(action)
            if board.is_checkmate():
                return action

            board.pop()

        return None

    def _optimal_move_from_prior(self, state: State) -> Action:
        """
        Return the move with the highest prior probability.
        """

        prior = self.model.predict(state)
        return max(prior, key=lambda x: x[1])[0]

    def playout(self, state: State) -> float:
        """
        Play a random game from the given board state and return the result.
        """

        board = chess.Board(state)

        while not board.is_game_over():
            move = self._optimal_move_from_prior(board.fen())
            board.push(move)

        return result(board)

    def run(
        self,
        state: State,
        time_budget: float,
        num_playouts=1,
        model=None,
    ) -> tuple[Node, int]:
        """
        Perform Monte Carlo tree search: run simulations starting from board
        state until time budget is exhausted.
        
        Args:
        - state (State): FEN string representing the current board state
        - time_budget (float): time budget in seconds
        - num_playouts (int): for each simulation, the number of playouts per
            each expanded leaf node
        
        Returns:
        - root (Node): the root node of the MCTS tree
        - num_simuls (int): the number of simulations performed
        """
        
        time_start = time.time()

        current_board = chess.Board(state)
        current_player = current_board.turn

        root = Node(0, current_player)

        # Stage: EXPAND
        actions = list(current_board.legal_moves)
        prior = self.model.predict(state)
        _, action_probs = zip(*prior)

        root.expand(state, actions, action_probs)
        
        # Record max time needed for a single simulation
        max_time_per_simul = 0
        num_simuls = 0

        while time.time() - time_start < time_budget - max_time_per_simul:
            time_start_simul = time.time()
            
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
                prior = self.model.predict(board_at_leaf_node.fen())
                actions, action_probs = zip(*prior)
                node.expand(next_state, actions, action_probs)

                # Stage: PLAYOUT
                if num_playouts == 0:
                    # TODO: Model prediction should return an evaluation as well
                    # as the prior probabilities
                    assert False
                else:
                    assert num_playouts > 0
                
                    acc_value = 0.0
                    for _ in range(num_playouts):
                        acc_value += self.playout(next_state)

                    value = acc_value / num_playouts

                # Get value from perspective of other player
                if node.current_player == chess.BLACK:
                    value *= -1

            self.backprop(search_path, value, parent.current_player ^ True)
            
            max_time_per_simul = max(
                max_time_per_simul,
                time.time() - time_start_simul,
            )
            num_simuls += 1

        return root, num_simuls

    def backprop(
        self,
        search_path: list[Node],
        value: float,
        current_player: chess.Color,
    ):
        for node in reversed(search_path):
            node.value_sum += value if node.current_player == current_player\
                else -value
            node.num_visits += 1

    def choose_move(self, position: str, time_budget: float=5) -> str:
        # Short-circuit if checkmate exists
        maybe_mate_move = self._check_for_mate(position)

        if maybe_mate_move is not None:
            print("Found mate in 1. Not performing MCTS.")
            return maybe_mate_move.uci()

        root, num_simuls = self.run(
            position,
            time_budget=time_budget,
            num_playouts=1,
        )

        print(root)
        print(f"Number of simulations: {num_simuls}")
        
        action, _ = root.select_child()
        return action.uci()
