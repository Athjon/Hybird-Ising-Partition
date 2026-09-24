"""Independent feasibility and scope checks for balanced IER atom proposals."""

import itertools
import math
from collections import Counter

import numpy as np
import pytest

from src.partition.hyper_ier_candidates import _local_pool, _native_delta, generate_balanced_moves


def _verify_every_subset(start, nodes, q, epsilon, moves):
    """Use direct sums, not the implementation's cached load arithmetic."""
    used = set()
    total = math.fsum(nodes)
    cap = (1 + epsilon) * total / q
    tol = 1e-10 * max(total, cap)
    for move in moves:
        assert len(move) in (2, 3)
        assert not used.intersection(move)
        used.update(move)
        for vertex, label in move.items():
            assert isinstance(vertex, int) and isinstance(label, int)
            assert 0 <= vertex < len(start) and 0 <= label < q
            assert start[vertex] != label
        for label in range(q):
            incoming = math.fsum(nodes[v] for v, target in move.items() if target == label)
            outgoing = math.fsum(nodes[v] for v in move if start[v] == label)
            assert abs(incoming - outgoing) <= 1e-12 * max(incoming, outgoing)
    for subset in itertools.product((False, True), repeat=len(moves)):
        state = list(start)
        for enabled, move in zip(subset, moves):
            if enabled:
                for vertex, target in move.items():
                    state[vertex] = target
        loads = [math.fsum(weight for weight, label in zip(nodes, state) if label == block)
                 for block in range(q)]
        assert max(loads, default=0.) <= cap + tol


@pytest.mark.parametrize('scale', [1e-12, 1., 1e6])
@pytest.mark.parametrize('q', [2, 3])
def test_all_binary_subsets_preserve_strict_weighted_capacity(q, scale):
    start = np.repeat(np.arange(q), 3)
    nodes = np.tile([1., 1., 2.], q) * scale
    edges = [[0, 3, 3], [1, 4], [2, 5], list(range(3*q)), [], [0]]
    weights = [2.5, .5, 3., .75, 100., 9.]
    before = start.copy()
    for seed in range(5):
        moves = generate_balanced_moves(start, edges, q, nodes, weights, 0.,
                                        max_moves=4, seed=seed)
        assert moves
        _verify_every_subset(start, nodes, q, 0., moves)
    np.testing.assert_array_equal(start, before)


@pytest.mark.parametrize('scale', [1e-12, 1., 1e6])
def test_triple_is_constructed_when_pair_swaps_cannot_cross_weight_profile(scale):
    start, nodes = [0, 0, 1], np.array([1., 1., 2.]) * scale
    edges = [[0, 2], [1, 2]]
    assert generate_balanced_moves(start, edges, 2, nodes, epsilon=0., allow_triples=False) == []
    moves = generate_balanced_moves(start, edges, 2, nodes, epsilon=0., seed=7)
    assert moves == [{0: 1, 1: 1, 2: 0}]
    _verify_every_subset(start, nodes, 2, 0., moves)
    after = [moves[0].get(v, label) for v, label in enumerate(start)]
    assert sum(nodes[v] == scale for v, label in enumerate(start) if label == 0) == 2
    assert sum(nodes[v] == scale for v, label in enumerate(after) if label == 0) == 0


@pytest.mark.parametrize('pool_strategy', ['local', 'global'])
def test_seed_is_local_reproducible_and_changes_atom_choices(pool_strategy):
    start = [0] * 8 + [1] * 8
    edges = [list(range(16))]
    kwargs = dict(max_moves=3, boundary_pool=10, allow_triples=False, pool_strategy=pool_strategy)
    np.random.seed(831)
    before = np.random.get_state()
    first = generate_balanced_moves(start, edges, 2, seed=13, **kwargs)
    assert first == generate_balanced_moves(start, edges, 2, seed=13, **kwargs)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    signatures = {tuple(tuple(sorted(move.items())) for move in
                        generate_balanced_moves(start, edges, 2, seed=seed, **kwargs))
                  for seed in range(8)}
    assert len(signatures) > 1


def test_boundary_pool_is_total_and_prioritizes_positive_cut_net_vertices():
    start = [0] * 5 + [1] * 5
    edges = [[0, 1, 5, 6], list(range(10))]
    for seed in range(5):
        moves = generate_balanced_moves(start, edges, 2, hyperedge_weights=[3., 0.],
                                        boundary_pool=4, max_moves=9, seed=seed)
        used = {v for move in moves for v in move}
        assert len(used) <= 4
        assert used <= {0, 1, 5, 6}
        _verify_every_subset(start, np.ones(10), 2, .03, moves)


def test_local_pool_keeps_atoms_in_one_cut_net_component_and_changes_seed_component():
    start = np.tile([0, 0, 1, 1], 8)
    edges = [list(range(4*i, 4*i+4)) for i in range(8)]
    components = set()
    for seed in range(8):
        moves = generate_balanced_moves(start, edges, 2, epsilon=0., max_moves=8,
                                        boundary_pool=4, allow_triples=False,
                                        pool_strategy='local', seed=seed)
        assert moves == generate_balanced_moves(start, edges, 2, epsilon=0., max_moves=8,
                                                boundary_pool=4, allow_triples=False,
                                                pool_strategy='local', seed=seed)
        used = {vertex for move in moves for vertex in move}
        assert len(moves) == 2 and len(used) == 4
        assert any(used == set(edge) for edge in edges)
        components.add(min(used) // 4)
        # Both distinct atoms participate in a common native objective term.
        assert any(sum(bool(set(move).intersection(edge)) for move in moves) == 2 for edge in edges)
    assert len(components) > 1


def test_local_pool_expands_cut_net_incidence_to_an_interacting_neighborhood():
    start = np.tile([0, 1], 10)
    edges = [[v, (v + 1) % len(start)] for v in range(len(start))]
    for seed in range(5):
        moves = generate_balanced_moves(start, edges, 2, epsilon=0., max_moves=10,
                                        boundary_pool=6, allow_triples=False, seed=seed)
        assert len(moves) == 3
        assert len({v for move in moves for v in move}) == 6
        assert any(sum(bool(set(move).intersection(edge)) for move in moves) >= 2 for edge in edges)
        _verify_every_subset(start, np.ones(len(start)), 2, 0., moves)


def test_local_pool_global_anchors_cover_missing_blocks_within_total_cap():
    # A component contains only blocks 0/1; isolated block 2 still gets a
    # representative when the pool is large enough to represent all blocks.
    labels = np.array([0, 0, 1, 1, 2, 2])
    edges, cut_edges = [[0, 1, 2, 3]], [0]
    incident = [[0], [0], [0], [0], [], []]
    boundary = np.array([True, True, True, True, False, False])
    for seed in range(4):
        pool = _local_pool(edges, cut_edges, incident, boundary, labels, 4,
                           np.random.default_rng(seed))
        assert len(pool) == len(set(pool)) == 4
        assert set(labels[pool]) == {0, 1, 2}


def test_empty_zero_weight_isolated_and_q1_inputs():
    assert generate_balanced_moves([], [], 2) == []
    assert generate_balanced_moves([0, 0], [[], [0, 0]], 1) == []
    assert generate_balanced_moves([0, 1], [], 2, max_moves=0) == []
    assert generate_balanced_moves([0, 1], [], 2, boundary_pool=1) == []
    moves = generate_balanced_moves([0, 0, 1, 1], [[], [0, 0]], 2,
                                    node_weights=np.zeros(4), epsilon=0., seed=6)
    assert moves
    _verify_every_subset([0, 0, 1, 1], np.zeros(4), 2, 0., moves)
    isolated = generate_balanced_moves([0, 0, 1, 1], [], 2, epsilon=0., seed=1)
    assert isolated
    _verify_every_subset([0, 0, 1, 1], np.ones(4), 2, 0., isolated)


def test_positive_cost_atoms_are_not_discarded_and_edges_keep_weights():
    # Every nontrivial balanced exchange cuts both previously internal nets.
    start, edges = [0, 0, 1, 1], [[0, 1, 1], [2, 3], [], [0]]
    weights = [2.5, .75, 99., 23.]
    moves = generate_balanced_moves(start, edges, 2, hyperedge_weights=weights,
                                    epsilon=0., allow_triples=False, seed=4)
    assert moves
    state = [moves[0].get(v, label) for v, label in enumerate(start)]
    cost = sum(w * max(0, len({state[v] for v in edge}) - 1)
               for edge, w in zip(edges, weights))
    assert cost == 3.25
    _verify_every_subset(start, np.ones(4), 2, 0., moves)


def test_joint_native_delta_matches_all_assignments_without_shared_net_double_counting():
    start = [0, 1, 2, 0]
    raw_edges = [[0, 1, 1, 2], [0, 1], [1, 2, 3], [3], [], [0, 1]]
    weights = [2.5, .75, 1.5, 9., 20., .25]
    edges = [sorted(set(edge)) for edge in raw_edges]
    counts = [Counter(start[v] for v in edge) for edge in edges]
    incident = [[e for e, edge in enumerate(edges) if vertex in edge] for vertex in range(4)]

    def cost(state):
        return math.fsum(w * max(0, len({state[v] for v in edge}) - 1)
                         for edge, w in zip(raw_edges, weights))

    for after in itertools.product(range(3), repeat=4):
        changes = {vertex: label for vertex, label in enumerate(after) if label != start[vertex]}
        assert _native_delta(changes, start, incident, counts, weights) == cost(after) - cost(start)


@pytest.mark.parametrize('scale', [1e-12, 1., 1e6])
def test_matching_has_no_absolute_resource_tolerance_floor(scale):
    # Slack permits the start, but these resources are not equal exchanges.
    moves = generate_balanced_moves([0, 1], [[0, 1]], 2,
                                    node_weights=np.array([1., 1.000001]) * scale,
                                    epsilon=.03, allow_triples=False)
    assert moves == []


def test_cumulative_positive_error_bound_when_initial_state_uses_tolerance():
    # Block 1 has already spent nearly all allowed capacity tolerance.  Several
    # individually near-balanced swaps would accumulate beyond that tolerance.
    near = 5e-13
    nodes = np.array([1. + near] * 4 + [1.] * 4 + [1.6008e-9])
    start = [0] * 4 + [1] * 5
    moves = generate_balanced_moves(start, [list(range(8))], 2, nodes,
                                    epsilon=0., boundary_pool=8, max_moves=4,
                                    allow_triples=False, seed=2)
    assert len(moves) == 1
    _verify_every_subset(start, nodes, 2, 0., moves)
    total = math.fsum(nodes)
    initial = [math.fsum(w for w, label in zip(nodes, start) if label == block)
               for block in range(2)]
    positive = [0., 0.]
    for move in moves:
        for block in range(2):
            change = math.fsum(nodes[v] for v, target in move.items() if target == block)
            change -= math.fsum(nodes[v] for v in move if start[v] == block)
            positive[block] += max(0., change)
    assert max(a+b for a, b in zip(initial, positive)) <= total / 2 + 1e-10 * total


@pytest.mark.parametrize('changes', [
    {'assignment': [[0, 1]]}, {'assignment': [0., 1.]}, {'assignment': [False, True]},
    {'assignment': [0, 2]}, {'q': True}, {'q': 2.5}, {'q': 0},
    {'node_weights': [[1., 1.]]}, {'node_weights': [1.]},
    {'node_weights': [1., -1.]}, {'node_weights': [1., np.nan]},
    {'hyperedge_weights': [[1.]]}, {'hyperedge_weights': [-1.]},
    {'hyperedges': [[0, 2]]}, {'hyperedges': [[0, 1.5]]}, {'hyperedges': [[False, 1]]},
    {'epsilon': -1.}, {'epsilon': np.nan}, {'epsilon': [0.]},
    {'max_moves': -1}, {'boundary_pool': 1.5}, {'seed': -1}, {'allow_triples': 1},
    {'pool_strategy': 'unsupported'},
])
def test_invalid_inputs_are_rejected(changes):
    kwargs = dict(assignment=[0, 1], hyperedges=[[0, 1]], q=2)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        generate_balanced_moves(**kwargs)


def test_infeasible_initial_partition_rejected_even_if_no_moves_requested():
    with pytest.raises(ValueError, match='capacity'):
        generate_balanced_moves([0, 0, 1], [], 2, epsilon=0., max_moves=0)
