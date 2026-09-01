import time

import chess

from chess_ai.chess_types import Action, State
from chess_ai.models.base import Model
from chess_ai.models.naive import ModelNaive
from chess_ai.players.ai import AI
from chess_ai.players.mcts.tree import Node, outcome_value


class AIMCTS(AI):
    NAME = "MirroredBot"
    AUTHOR = "Freddy Jiang"

    def __init__(
        self,
        model: Model | None = None,
        time_budget: float = 5.0,
    ):
        super().__init__(time_budget)
        self.model = ModelNaive() if model is None else model

    def _check_for_mate(self, state: State) -> Action | None:
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

        value = outcome_value(board)
        assert value is not None
        return value

    def run(
        self,
        state: State,
        time_budget: float,
        num_playouts: int = 1,
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

        if num_playouts < 1:
            # TODO: Model prediction should return an evaluation as well as the
            # prior probabilities, so a search without playouts is possible
            raise NotImplementedError(
                "Evaluating a leaf node without playouts is not supported"
            )

        time_start = time.time()

        current_board = chess.Board(state)
        current_player = current_board.turn

        root = Node(0, current_player)

        # Stage: EXPAND
        actions: tuple[Action, ...] | list[Action] = list(
            current_board.legal_moves
        )

        if not actions:
            raise ValueError(f"No legal moves in position: {state}")

        prior = self.model.predict(state)
        _, action_probs = zip(*prior, strict=True)

        root.expand(state, actions, action_probs)

        # Record max time needed for a single simulation
        max_time_per_simul = 0.0
        num_simuls = 0

        while time.time() - time_start < time_budget - max_time_per_simul:
            time_start_simul = time.time()

            node = root
            search_path = [node]
            action: Action | None = None

            # Stage: SELECT
            while node.is_expanded():
                action, node = node.select_child()
                search_path.append(node)

            # The root is always expanded, so the loop above ran at least once
            assert action is not None

            parent = search_path[-2]
            # A node is only ever selected after being expanded, which
            # is where its state is recorded
            assert parent.state is not None

            # We are now at a leaf node
            # Make a move
            board_at_leaf_node = chess.Board(parent.state)
            board_at_leaf_node.push(action)
            next_state = board_at_leaf_node.fen()

            # Get value of next state from perspective of White
            value = outcome_value(board_at_leaf_node)

            if value is None:
                # Game has not ended
                # Stage: EXPAND
                prior = self.model.predict(next_state)
                actions, action_probs = zip(*prior, strict=True)
                node.expand(next_state, actions, action_probs)

                # Stage: PLAYOUT
                acc_value = sum(
                    self.playout(next_state) for _ in range(num_playouts)
                )
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
    ) -> None:
        for node in reversed(search_path):
            node.value_sum += (
                value if node.current_player == current_player else -value
            )
            node.num_visits += 1

    def choose_move(self, state: State) -> Action:
        # Short-circuit if checkmate exists
        maybe_mate_move = self._check_for_mate(state)

        if maybe_mate_move is not None:
            print("Found mate in 1. Not performing MCTS.")
            return maybe_mate_move

        root, num_simuls = self.run(state, time_budget=self.time_budget)

        print(root)
        print(f"Number of simulations: {num_simuls}")

        action, _ = root.select_child()
        return action
