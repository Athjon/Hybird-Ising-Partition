"""Independent small oracles for the experimental objective diagnostic."""
import itertools

import numpy as np
import torch

from benchmarks.hypergraph.ogp_objective_diagnostic import (
    balanced_assignments, costs, expected_km1, formula_audit,
    make_instance, swap_descent, topk_round,
)
from src.partition.hyper_utils import build_clique_expanded_graph


def test_km1_expectation_and_gradient_against_full_categorical_enumeration():
    checks = formula_audit()
    assert checks['finite_difference_gradcheck']
    for result in checks['exact_expectation_checks']:
        assert result['value_error'] < 1e-12
        assert result['gradient_max_error'] < 1e-12
    assert not checks['isolated_argmax_numpy_cut_requires_grad']


def test_km1_is_not_cut_net_for_three_occupied_blocks():
    p = torch.eye(3, dtype=torch.float64)
    assert expected_km1(p, [[0, 1, 2]]).item() == 2.0


def test_balanced_enumeration_matches_brute_force_quotient():
    expected = {x for x in itertools.product((0, 1), repeat=8) if sum(x) == 4 and x[0] == 0}
    actual = {tuple(x) for x in balanced_assignments(8)}
    assert actual == expected
    assert len(actual) == 35


def test_vectorized_clique_cost_matches_repository_graph_expansion():
    edges = np.array([[0, 1, 2, 3], [1, 2, 4, 5]])
    states = np.array(list(itertools.product((0, 1), repeat=6)))
    graph = build_clique_expanded_graph(edges.tolist(), num_nodes=6).to_dense().numpy()
    expected = np.array([sum(graph[i, j] for i in range(6) for j in range(i + 1, 6) if x[i] != x[j]) for x in states])
    np.testing.assert_allclose(costs(states, edges)[1], expected, rtol=1e-6)


def test_rounding_and_descent_preserve_feasibility_and_do_not_increase_cut():
    edges = make_instance(8, 'random', 71)
    states = topk_round(np.random.default_rng(13).random((5, 8)))
    assert np.all(states.sum(axis=1) == 4)
    for state in states:
        refined = swap_descent(state, edges)
        assert refined.sum() == 4
        assert costs(refined[None], edges)[0][0] <= costs(state[None], edges)[0][0]
        neighbors = []
        for u in np.flatnonzero(refined == 0):
            for v in np.flatnonzero(refined == 1):
                neighbor = refined.copy()
                neighbor[[u, v]] = neighbor[[v, u]]
                neighbors.append(neighbor)
        assert costs(np.array(neighbors), edges)[0].min() >= costs(refined[None], edges)[0][0]
