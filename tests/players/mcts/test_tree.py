import chess
import pytest

from chess_ai.players.mcts.tree import Node, outcome_value, ucb_score


@pytest.mark.parametrize(
    "fen, value",
    [
        (chess.Board.starting_fen, None),
        ("5k1r/6b1/p2BQ3/3Pp1p1/P3Pp2/8/4KPP1/1q6 b - - 0 35", 1.0),
        ("3r2k1/p4ppp/Q7/3p4/1N6/2N5/PP3nPP/R5RK w - - 1 29", -1.0),
        ("7k/8/6Q1/3BK3/8/8/8/8 b - - 20 81", 0.0),
    ],
)
def test_outcome_value_of_board_is_correctly_encoded(
    fen: str, value: float | None
):
    assert outcome_value(chess.Board(fen)) == value


# TODO
@pytest.mark.skip
def test_ucb_score():
    pass


def test_select_best_action_prefers_the_most_visited_child():
    root = Node(prior=0.0, current_player=chess.WHITE)
    actions = [chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4", "a2a3")]
    root.expand(chess.STARTING_FEN, actions, [0.2, 0.3, 0.5])

    # a2a3 keeps the highest UCB score because it is barely explored, but e2e4
    # is the move the search actually spent its time on
    root.num_visits = 40
    root.children[actions[0]].num_visits = 30
    root.children[actions[0]].value_sum = -6.0
    root.children[actions[1]].num_visits = 9
    root.children[actions[1]].value_sum = -1.0
    root.children[actions[2]].num_visits = 1

    assert ucb_score(root, root.children[actions[2]]) > ucb_score(
        root, root.children[actions[0]]
    )
    assert root.select_best_action() == actions[0]


def test_select_best_action_breaks_ties_on_value():
    root = Node(prior=0.0, current_player=chess.WHITE)
    actions = [chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4")]
    root.expand(chess.STARTING_FEN, actions, [0.5, 0.5])

    root.num_visits = 20
    for action in actions:
        root.children[action].num_visits = 10

    # Children score from the mover's perspective, so the lower value is the
    # better outcome for the parent
    root.children[actions[0]].value_sum = 5.0
    root.children[actions[1]].value_sum = -5.0

    assert root.select_best_action() == actions[1]


def test_select_best_action_rejects_an_unexpanded_node():
    with pytest.raises(ValueError):
        Node(prior=0.0, current_player=chess.WHITE).select_best_action()
