"""Regression oracles for the repaired native hypergraph FEM entry point."""
import itertools

import numpy as np
import pytest
import torch

from src.hyper_solver import FemCoarsenSolver, _Q4PUBOWrapper
from src.partition.hyper_objective import (
    BalanceInfeasibleError, BalanceSearchError, HypergraphObjective,
    round_balanced_probabilities,
)
from src.partition.hyper_quotient import connectivity_cost


@pytest.mark.parametrize('q', [2, 3])
def test_native_expectation_and_logit_gradient_match_complete_enumeration(q):
    edges = [[0, 1, 2, 3], [0, 2], [1], []]
    weights = [1.7, 0.6, 3.0, 8.0]
    objective = HypergraphObjective(edges, np.ones(4), q, hyperedge_weights=weights, imbalance_weight=0)
    generator = torch.Generator().manual_seed(7)
    h = torch.randn(2, 4, q, generator=generator, dtype=torch.float64, requires_grad=True)
    p = h.softmax(-1)
    exact = h.sum(dim=(1, 2)) * 0
    for assignment in itertools.product(range(q), repeat=4):
        cost = connectivity_cost(assignment, edges, weights)
        exact = exact + cost * p[:, torch.arange(4), torch.tensor(assignment)].prod(-1)
    actual = objective.expectation(None, p)
    np.testing.assert_allclose(actual.detach(), exact.detach(), atol=1e-12)
    ga = torch.autograd.grad(actual.sum(), h, retain_graph=True)[0]
    ge = torch.autograd.grad(exact.sum(), h)[0]
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)
    assert ga.abs().max() > 1e-3


def test_pubo_batch_scores_and_gradient_remain_differentiable():
    wrapper = _Q4PUBOWrapper([[0, 1, 2]], [1, 2, 1], 4, 3, hyperedge_weights=[2.5])
    h = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    values = wrapper.expectation(None, h.softmax(-1))
    assert values.shape == (2,)
    assert torch.autograd.grad(values.sum(), h)[0].abs().max() > 0
    assert wrapper.inference(None, h.softmax(-1)).shape == (2, 3)


def test_weighted_clique_expectation_matches_pair_enumeration():
    p = torch.tensor([[[0.2, 0.8], [0.5, 0.5], [0.9, 0.1], [0.7, 0.3]]], dtype=torch.float64)
    objective = HypergraphObjective([[0, 1, 2, 3]], [1] * 4, 2, hyperedge_weights=[7], map_type='clique')
    expected = sum(7 / 3 * (1 - (p[:, i] * p[:, j]).sum(-1)) for i, j in itertools.combinations(range(4), 2))
    torch.testing.assert_close(objective.cut_expectation(p), expected)


def test_star_auxiliaries_do_not_contribute_to_resource_balance():
    objective = HypergraphObjective([[0, 1, 2, 3]], [1] * 4, 2, hyperedge_weights=[0], map_type='star')
    assert objective.num_variables == 5
    p = torch.full((2, 5, 2), 0.5, dtype=torch.float64)
    p[0, 4] = torch.tensor([1., 0.])
    p[1, 4] = torch.tensor([0., 1.])
    torch.testing.assert_close(objective.expectation(None, p), torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize('n,q', [(8, 2), (12, 3), (12, 4)])
def test_uniform_probabilities_are_rounded_to_exact_unit_balance(n, q):
    labels = round_balanced_probabilities(np.full((n, q), 1 / q), np.ones(n), q, epsilon=0)
    np.testing.assert_array_equal(np.bincount(labels, minlength=q), np.full(q, n // q))


def test_weighted_projection_recovers_a_case_single_moves_cannot_repair():
    weights = [3, 3, 2, 2, 2]
    p = np.eye(2)[[0, 1, 0, 1, 0]]
    labels = round_balanced_probabilities(p, weights, 2, epsilon=0)
    np.testing.assert_allclose(np.bincount(labels, weights=weights, minlength=2), [6, 6])


def test_exact_packing_budget_failure_is_not_infeasibility():
    weights = [4, 2, 1, 5, 9, 3, 2, 2, 2]
    p = np.full((9, 3), 1 / 3)
    labels = round_balanced_probabilities(p, weights, 3, epsilon=0)
    np.testing.assert_allclose(np.bincount(labels, weights=weights, minlength=3), [10, 10, 10])
    with pytest.raises(BalanceSearchError, match='disabled|size limit'):
        round_balanced_probabilities(p, weights, 3, epsilon=0, exact_max_nodes=0)
    with pytest.raises(BalanceSearchError, match='budget'):
        round_balanced_probabilities(p, weights, 3, epsilon=0, search_budget=1)


def test_proven_infeasibility_is_distinct_from_unknown():
    with pytest.raises(BalanceInfeasibleError, match='integer block capacities'):
        round_balanced_probabilities(np.full((3, 2), 0.5), [4, 4, 4], 2, epsilon=0)
    with pytest.raises(BalanceInfeasibleError, match='complete capacity search'):
        round_balanced_probabilities(np.full((4, 2), 0.5), [4, 4, 3, 1], 2, epsilon=0)
    with pytest.raises(BalanceInfeasibleError, match='node exceeds'):
        round_balanced_probabilities(np.full((2, 2), 0.5), [7, 1], 2, epsilon=0)


def test_capacity_semantics_do_not_require_symmetric_lower_bounds():
    labels = round_balanced_probabilities(np.eye(3), [4, 4, 2], 3, epsilon=0.2)
    np.testing.assert_array_equal(labels, [0, 1, 2])


@pytest.mark.parametrize('scale', [1.0, 1e-6, 1e-12, 1e6])
def test_resource_rescaling_preserves_capacity_and_soft_penalty(scale):
    labels = round_balanced_probabilities(np.full((4, 2), 0.5), np.ones(4) * scale, 2, epsilon=0)
    np.testing.assert_array_equal(np.bincount(labels, minlength=2), [2, 2])
    p = torch.tensor([[[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]], dtype=torch.float64)
    base = HypergraphObjective([[0, 1, 2]], [1, 3, 2], 2)
    scaled = HypergraphObjective([[0, 1, 2]], np.array([1, 3, 2]) * scale, 2)
    torch.testing.assert_close(base.expectation(None, p), scaled.expectation(None, p))


def test_best_fit_handles_large_packing_without_an_exact_search():
    weights = [3] * 10 + [2] * 15
    labels = round_balanced_probabilities(np.full((25, 10), 0.1), weights, 10,
                                          epsilon=0, exact_max_nodes=0)
    np.testing.assert_array_equal(np.bincount(labels, weights=weights, minlength=10), [6] * 10)


@pytest.mark.parametrize('method,map_type', [('fem', 'native'), ('pubo', 'native'), ('fem', 'clique'), ('fem', 'star')])
def test_actual_fem_entry_point_is_balanced_and_scores_native_weighted_cut(method, map_type):
    edges = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 4]]
    edge_weights = [3.0, 2.0, 0.5]
    solver = FemCoarsenSolver()
    labels = solver.initial_partition(edges, torch.ones(8), 2, method=method, map_type=map_type,
        hyperedge_weights=edge_weights, epsilon=0, num_trials=4, num_steps=50, seed=3, anneal='inverse')
    assert labels.shape == (8,)
    np.testing.assert_array_equal(np.bincount(labels, minlength=2), [4, 4])
    assert solver.last_result['native_cut'] == connectivity_cost(labels, edges, edge_weights)
    assert solver.last_result['native_cut'] == min(solver.last_result['candidate_native_cuts'])


def test_native_is_default_and_supports_weighted_multiway_partition():
    solver = FemCoarsenSolver()
    labels = solver.initial_partition([[0, 1], [1, 2]], [4, 4, 2], 3,
        epsilon=0.2, num_trials=2, num_steps=5)
    assert solver.last_result['map_type'] == 'native'
    assert np.bincount(labels, weights=[4, 4, 2], minlength=3).max() <= 4


def test_selects_best_feasible_native_candidate_not_first_or_surrogate_score(monkeypatch):
    from fem import FEM
    probabilities = torch.tensor([
        [[1., 0.], [0., 1.], [1., 0.], [0., 1.]],
        [[1., 0.], [1., 0.], [0., 1.], [0., 1.]],
    ], dtype=torch.float64)

    def fake_solve(case):
        case.solver.probabilities = probabilities
        return probabilities.argmax(-1), torch.tensor([-100., 100.])

    monkeypatch.setattr(FEM, 'solve', fake_solve)
    solver = FemCoarsenSolver()
    labels = solver.initial_partition([[0, 1], [2, 3]], [1] * 4, 2, num_trials=2, num_steps=1, epsilon=0)
    assert connectivity_cost(labels, [[0, 1], [2, 3]]) == 0
    assert solver.last_result['selected_trial'] == 1
