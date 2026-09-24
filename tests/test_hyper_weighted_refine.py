"""Weighted objective and upper-capacity regressions for native flow/V-cycle."""

import numpy as np
import pytest

from src.hyper_solver import HyperRefineSolver, _refine_flow, vcycle_uncoarsen
from src.partition.hyper_quotient import connectivity_cost, quotient_hypergraph


WEIGHTED_EDGES = [[0, 2, 5], [1, 3, 4, 5], [4, 5], [0, 2, 3],
                  [1, 2, 3, 5], [2, 4], [2, 5], [1, 2, 4]]
EDGE_WEIGHTS = [1, 4, 13, 2, 7, 6, 2, 5]


def test_weighted_flow_does_not_accept_unweighted_improvement():
    # Ignoring edge weights flips vertex 5 and raises native cost 27 -> 37.
    start = np.array([0, 0, 0, 1, 1, 1])
    result = HyperRefineSolver().refine(
        start, WEIGHTED_EDGES, 2, hyperedge_weights=EDGE_WEIGHTS,
        max_imbalance=1 / 3,
    )
    assert connectivity_cost(result, WEIGHTED_EDGES, EDGE_WEIGHTS) <= 27
    assert np.bincount(result, minlength=2).max() <= 4
    np.testing.assert_array_equal(start, [0, 0, 0, 1, 1, 1])


def test_upper_capacity_does_not_impose_absolute_imbalance_lower_bound():
    # Loads 4,4,2 satisfy U=4, although max absolute relative deviation is .4.
    start = np.array([0, 1, 2])
    result = _refine_flow(start, [], 3, node_weights=[4, 4, 2], max_imbalance=.2)
    np.testing.assert_array_equal(result, start)


def test_weighted_packing_trap_uses_checked_exact_fallback():
    start = np.array([0, 1, 0, 1, 0])
    weights = np.array([3, 3, 2, 2, 2])
    result = _refine_flow(start, [], 2, node_weights=weights, max_imbalance=0)
    np.testing.assert_array_equal(np.bincount(result, weights=weights, minlength=2), [6, 6])


def test_disabled_repair_never_silently_returns_over_capacity():
    with pytest.raises(RuntimeError, match='does not prove the instance infeasible'):
        _refine_flow(np.array([0, 1, 0, 1, 0]), [], 2,
                     node_weights=[3, 3, 2, 2, 2], max_imbalance=0,
                     repair_balance=False)


def test_hybrid_explicitly_rejects_nonunit_edge_weights():
    with pytest.raises(NotImplementedError, match='nonunit hyperedge_weights'):
        HyperRefineSolver().refine(
            np.array([0, 0, 0, 1, 1, 1]), WEIGHTED_EDGES, 2,
            hyperedge_weights=EDGE_WEIGHTS, mode_cycle=('mcts', 'flow'),
        )


class RecordingRefiner:
    def __init__(self):
        self.calls = []

    def refine(self, assignment, hyperedges, q, node_weights=None, hyperedge_weights=None):
        self.calls.append((np.asarray(assignment).copy(), [list(edge) for edge in hyperedges],
                           np.asarray(node_weights).copy(), np.asarray(hyperedge_weights).copy()))
        return np.asarray(assignment).copy()


def weighted_hierarchy():
    edges = [[0], [0, 1, 2], [2, 3], [3, 4, 5], [0, 5], [0, 5]]
    nodes = np.array([2, 1, 1, 1, 1, 2])
    weights = np.array([999, 10, 1, 11, 3, 7])
    mapping = np.array([0, 0, 1, 2, 3, 3])
    normalized, _, normalized_weights = quotient_hypergraph(edges, np.arange(6), nodes, weights)
    coarse_edges, coarse_nodes, coarse_weights = quotient_hypergraph(edges, mapping, nodes, weights)
    hierarchy = [
        {'hyperedges': normalized, 'node_weights': nodes,
         'hyperedge_weights': normalized_weights,
         'groups': [[i] for i in range(6)], 'num_nodes': 6, 'remap': mapping},
        {'hyperedges': coarse_edges, 'node_weights': coarse_nodes,
         'hyperedge_weights': coarse_weights,
         'groups': [[0, 1], [2], [3], [4, 5]], 'num_nodes': 4,
         'remap': np.array([0, 0, 1, 1])},
    ]
    return edges, nodes, weights, hierarchy


def test_empty_hierarchy_preserves_explicit_original_weights():
    refiner = RecordingRefiner()
    edges, nodes, weights, _ = weighted_hierarchy()
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = vcycle_uncoarsen(labels, [], edges, 2, refiner, False,
                             node_weights=nodes, hyperedge_weights=weights)
    np.testing.assert_array_equal(result, labels)
    assert len(refiner.calls) == 1
    np.testing.assert_array_equal(refiner.calls[0][2], nodes)
    np.testing.assert_array_equal(refiner.calls[0][3], weights)


def test_every_hierarchy_level_and_final_pass_receive_own_weights():
    edges, nodes, weights, hierarchy = weighted_hierarchy()
    refiner = RecordingRefiner()
    result = vcycle_uncoarsen(np.array([0, 1]), hierarchy, edges, 2, refiner, False,
                             node_weights=nodes, hyperedge_weights=weights)
    np.testing.assert_array_equal(result, [0, 0, 0, 1, 1, 1])
    assert [len(call[0]) for call in refiner.calls] == [4, 6, 6]
    for call, level in zip(refiner.calls[:2], reversed(hierarchy)):
        np.testing.assert_array_equal(call[2], level['node_weights'])
        np.testing.assert_array_equal(call[3], level['hyperedge_weights'])
    np.testing.assert_array_equal(refiner.calls[-1][2], nodes)
    np.testing.assert_array_equal(refiner.calls[-1][3], weights)
    costs = [connectivity_cost(call[0], call[1], call[3]) for call in refiner.calls]
    assert costs == [11, 11, 11]


def test_inferred_original_edge_weights_align_after_singleton_removal():
    edges, nodes, weights, hierarchy = weighted_hierarchy()
    refiner = RecordingRefiner()
    vcycle_uncoarsen(np.array([0, 1]), hierarchy, edges, 2, refiner, False)
    np.testing.assert_array_equal(refiner.calls[-1][2], nodes)
    # Singleton weight cannot be recovered, but its contribution is zero.
    np.testing.assert_array_equal(refiner.calls[-1][3][1:], weights[1:])
    assert connectivity_cost(refiner.calls[-1][0], edges, refiner.calls[-1][3]) == 11


def test_ambiguous_original_edge_weight_alignment_requires_explicit_weights():
    edges, _, _, hierarchy = weighted_hierarchy()
    with pytest.raises(ValueError, match='cannot safely align'):
        vcycle_uncoarsen(np.array([0, 1]), hierarchy, list(reversed(edges)), 2,
                        RecordingRefiner(), False)


def test_empty_hierarchy_with_real_flow_preserves_capacity_and_weighted_cost():
    edges, nodes, weights, _ = weighted_hierarchy()
    labels = np.array([0, 0, 0, 1, 1, 1])
    refiner = HyperRefineSolver()
    refiner.update_params(max_imbalance=.5)
    result = vcycle_uncoarsen(labels, [], edges, 2, refiner, False,
                             node_weights=nodes, hyperedge_weights=weights)
    assert np.bincount(result, weights=nodes, minlength=2).max() <= 6
    assert connectivity_cost(result, edges, weights) <= connectivity_cost(labels, edges, weights)


@pytest.mark.parametrize('weights', [[1], [1, -1], [1, np.nan]])
def test_invalid_edge_weights_are_rejected(weights):
    with pytest.raises(ValueError, match='hyperedge_weights'):
        _refine_flow(np.array([0, 1]), [[0], [0, 1]], 2, hyperedge_weights=weights)
