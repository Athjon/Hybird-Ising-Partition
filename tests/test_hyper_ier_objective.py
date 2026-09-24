"""Native joint-selector objective oracles, including correlated move pins."""

import itertools

import numpy as np
import pytest
import torch

from src.partition.hyper_ier_objective import JointMoveObjective


def fixture(scale=1):
    assignment = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 2, 2])
    nodes = np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * scale
    edges = [[0, 1, 3, 4, 6, 8], [0, 0, 1, 4, 9], [2, 5, 8, 11],
             [7, 9, 10], [6], [5, 5], [], [0, 4, 11], [3, 6, 9]]
    weights = [2.3, .8, 1.7, .5, 20, 3, 7, 1.1, 0]
    moves = [{0: 1, 1: 2, 2: 0}, {3: 2, 4: 0, 5: 1}, {6: 1, 8: 0}]
    return assignment, nodes, edges, weights, moves


def direct_apply(assignment, moves, selectors):
    labels = assignment.copy()
    for active, move in zip(selectors, moves):
        if active:
            for vertex, label in move.items():
                labels[vertex] = label
    return labels


def direct_cost(labels, edges, weights):
    return sum(weight * max(0, len({int(labels[v]) for v in edge}) - 1)
               for edge, weight in zip(edges, weights))


def test_all_discrete_subsets_match_native_oracle_and_preserve_weighted_q3_capacity():
    assignment, nodes, edges, weights, moves = fixture()
    objective = JointMoveObjective(assignment, edges, moves, 3, nodes, weights, epsilon=0)
    assert objective.num_moves == 3 and objective.num_nodes == 12
    assert objective.num_variables == 3
    selectors = torch.tensor(list(itertools.product((0, 1), repeat=3)))
    expected = []
    for selector in selectors.numpy():
        labels = direct_apply(assignment, moves, selector)
        np.testing.assert_array_equal(objective.apply(selector), labels)
        np.testing.assert_array_equal(np.bincount(labels, weights=nodes, minlength=3), [5, 5, 5])
        expected.append(direct_cost(labels, edges, weights))
    torch.testing.assert_close(objective.energy(None, selectors), torch.tensor(expected, dtype=torch.float64))
    categorical = torch.nn.functional.one_hot(selectors, num_classes=2).to(torch.float64)
    torch.testing.assert_close(objective.energy(None, categorical), torch.tensor(expected, dtype=torch.float64))
    torch.testing.assert_close(objective.inference(None, categorical), selectors)
    exact = objective.exact_best()
    assert exact['complete'] and exact['evaluated'] == 8
    assert exact['cost'] == pytest.approx(min(expected))
    assert direct_cost(exact['assignment'], edges, weights) == pytest.approx(min(expected))


def test_expectation_and_logit_gradient_match_complete_selector_enumeration():
    assignment, nodes, edges, weights, moves = fixture()
    objective = JointMoveObjective(assignment, edges, moves, 3, nodes, weights, epsilon=0)
    generator = torch.Generator().manual_seed(20260924)
    logits = torch.randn(2, 3, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    p = logits.softmax(dim=-1)
    expected = logits.sum(dim=(1, 2)) * 0
    for selector in itertools.product((0, 1), repeat=3):
        cost = direct_cost(direct_apply(assignment, moves, selector), edges, weights)
        probability = p[:, torch.arange(3), torch.tensor(selector)].prod(dim=-1)
        expected = expected + cost * probability
    actual = objective.expectation(None, p)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(gradient, oracle_gradient, atol=1e-12, rtol=1e-12)
    assert gradient.abs().max() > 1e-3


def test_same_atom_pins_are_correlated_not_independent_vertex_marginals():
    # Atom flips all four vertices. Pins 0/1 remain together with certainty,
    # while independent .5/.5 vertex marginals would falsely predict cut=.5.
    objective = JointMoveObjective([0, 0, 1, 1], [[0, 1]],
                                    [{0: 1, 1: 1, 2: 0, 3: 0}], 2, epsilon=0)
    p = torch.tensor([[[.5, .5]]], dtype=torch.float64, requires_grad=True)
    actual = objective.expectation(None, p)
    torch.testing.assert_close(actual, torch.zeros(1, dtype=torch.float64))
    naive_independent_vertex_cost = 1 - .5 ** 2 - .5 ** 2
    assert naive_independent_vertex_cost == .5
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), p)[0], torch.zeros_like(p))


def test_zero_moves_are_a_differentiable_constant_with_valid_upper_only_capacity():
    objective = JointMoveObjective([0, 1, 2], [[0, 1, 2], [2], []], [], 3,
                                    node_weights=[4, 4, 2], hyperedge_weights=[2.5, 9, 8], epsilon=.2)
    p = torch.empty((2, 0, 2), dtype=torch.float64, requires_grad=True)
    values = objective.expectation(None, p)
    torch.testing.assert_close(values, torch.tensor([5., 5.], dtype=torch.float64))
    assert torch.autograd.grad(values.sum(), p)[0].shape == (2, 0, 2)
    assert objective.inference(None, p).shape == (2, 0)
    torch.testing.assert_close(objective.energy(None, torch.empty((2, 0), dtype=torch.long)), values)
    np.testing.assert_array_equal(objective.apply([]), [0, 1, 2])
    exact = objective.exact_best()
    assert exact['cost'] == 5 and exact['evaluated'] == 1 and exact['complete']


@pytest.mark.parametrize('scale', [1e-12, 1, 1e9])
def test_move_balance_and_all_subsets_are_invariant_to_resource_units(scale):
    assignment, nodes, edges, weights, moves = fixture(scale)
    objective = JointMoveObjective(assignment, edges, moves, 3, nodes, weights, epsilon=0)
    for selector in itertools.product((0, 1), repeat=3):
        labels = objective.apply(selector)
        np.testing.assert_allclose(np.bincount(labels, weights=nodes, minlength=3), [5 * scale] * 3,
                                   rtol=1e-12, atol=0)


@pytest.mark.parametrize('moves,match', [
    ([{}], 'nonempty'),
    ([{0: 0}], 'actually change'),
    ([{0: 1}], 'preserve every block load'),
    ([{0: 1, 2: 0}, {0: 1, 3: 0}], 'disjoint'),
    ([{0: 2}], 'target label'),
    ([{4: 1}], 'move vertex'),
])
def test_illegal_move_atoms_are_rejected(moves, match):
    with pytest.raises(ValueError, match=match):
        JointMoveObjective([0, 0, 1, 1], [[0, 1]], moves, 2, epsilon=0)


def test_initial_capacity_violation_is_rejected_without_repair():
    with pytest.raises(ValueError, match='initial assignment exceeds'):
        JointMoveObjective([0, 0, 0, 1], [[0, 1]], [], 2, epsilon=0)


def test_individually_tolerated_load_residuals_cannot_accumulate_across_atoms():
    delta = 4e-10
    nodes = [1, 1, 1, 1 + delta, 1 + delta, 1 - 2 * delta]
    with pytest.raises(ValueError, match='residuals can accumulate'):
        JointMoveObjective([0, 0, 0, 1, 1, 1], [], [{0: 1, 3: 0}, {1: 1, 4: 0}],
                           2, node_weights=nodes, epsilon=0)


def test_selector_validation_and_exact_search_guard():
    assignment, nodes, edges, weights, moves = fixture()
    objective = JointMoveObjective(assignment, edges, moves, 3, nodes, weights, epsilon=0)
    with pytest.raises(ValueError, match='binary'):
        objective.apply([0, .5, 1])
    with pytest.raises(ValueError, match='length'):
        objective.apply([0, 1])
    with pytest.raises(ValueError, match='shaped'):
        objective.expectation(None, torch.ones(2, 12, 2))
    with pytest.raises(ValueError, match='max_moves'):
        objective.exact_best(max_moves=2)


def test_sparse_groups_scale_with_touched_atoms_not_all_selectors():
    count = 50
    objective = JointMoveObjective([0] * count + [1] * count,
                                    [[i, i + count] for i in range(count)],
                                    [{i: 1, i + count: 0} for i in range(count)], 2, epsilon=0)
    assert objective.num_moves == count
    assert sum(group['indices'].size for group in objective._groups) == count
    p = torch.full((2, count, 2), .5, dtype=torch.float64)
    torch.testing.assert_close(objective.expectation(None, p), torch.full((2,), float(count), dtype=torch.float64))
    summary = objective.interaction_summary()
    assert summary['edges_touching_atoms'] == {'0': 0, '1': 50, '2': 0, '3+': 0}
    # All fifty terms are actually constant despite touching one atom each.
    assert 'not actual polynomial degree' in summary['interpretation']


def test_interaction_summary_distinguishes_trivial_and_zero_weight_edges():
    assignment, nodes, edges, weights, moves = fixture()
    objective = JointMoveObjective(assignment, edges, moves, 3, nodes, weights, epsilon=0)
    summary = objective.interaction_summary()
    assert summary['total_hyperedges'] == 9
    assert summary['nontrivial_hyperedges'] == 6
    assert summary['excluded_empty_or_singleton_hyperedges'] == 3
    assert summary['edges_touching_atoms'] == {'0': 1, '1': 0, '2': 3, '3+': 2}
    assert summary['positive_weight_edges_touching_atoms'] == {'0': 1, '1': 0, '2': 2, '3+': 2}
    assert summary['max_atoms_touching_one_hyperedge'] == 3
