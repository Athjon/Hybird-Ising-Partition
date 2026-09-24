"""Small exact oracles used to separate reachability from energy barriers."""
import numpy as np

from benchmarks.hypergraph.native_search_probe import (
    canonical_binary, exact_states, exact_swap_barrier, profile_restricted_optimum,
)


def test_weight_profile_can_exclude_global_optimum():
    # Two weight-2 vertices prefer the same block. Local swaps from the given
    # strict-capacity state preserve one heavy vertex per block forever.
    nodes = np.array([1., 1., 1., 1., 2., 2.])
    start = np.array([0, 0, 1, 1, 0, 1])
    states, codes, values = exact_states([[4, 5]], nodes, np.ones(1), 2)
    restricted = profile_restricted_optimum(states, values, start, nodes, 2)
    assert values.min() == 0
    assert restricted['restricted_optimum'] == 1
    assert restricted['reachable_labeled_states'] == 12
    assert len(states) == 14
    assert np.array_equal(states @ (2 ** np.arange(6)), codes)
    assert np.all((states == 0) @ nodes == 4)


def test_swap_barrier_includes_intermediate_energies():
    # Three balanced bisection classes on four vertices form a triangle; two
    # minima are directly adjacent, so a high third energy is no barrier.
    codes = np.array([3, 5, 6, 9, 10, 12])
    values = np.array([0., 0., 9., 9., 0., 0.])
    result = exact_swap_barrier(codes, values, 4, 3, 5)
    assert result['barrier_above_optimum'] == 0
    assert result['path_energies'] == [0., 0.]
    assert result['canonical_path_masks'] == [canonical_binary(3, 4), canonical_binary(5, 4)]


def test_exact_native_q3_includes_duplicates_and_empty_edges():
    edges = [[0, 0, 1, 2], [], [2]]
    states, _, values = exact_states(edges, np.ones(3), np.array([2., 5., 7.]), 3)
    assert len(states) == 6
    assert np.all(values == 4)
