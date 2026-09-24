"""Small, self-contained checks for hypergraph quotient coarsening."""

import itertools

import numpy as np
import pytest

from src.hyper_solver import KahyparLikeSolver, HyperRefineSolver, vcycle_uncoarsen
from src.partition.hyper_quotient import (
    balanced_packing_feasible, connectivity_cost, quotient_hypergraph,
)


def test_quotient_preserves_weighted_cut_and_balance_under_composition():
    edges = [[0, 1, 2], [1, 2, 3], [0, 0, 4], [4, 5], [1, 5], [3]]
    edge_weights = [2.0, 1.5, 3.0, 4.0, 0.5, 7.0]
    node_weights = [1.0, 2.0, 1.5, 1.0, 2.5, 1.0]
    first = np.array([0, 0, 1, 2, 3, 3])
    second = np.array([0, 1, 1, 2])
    composed = second[first]

    first_edges, first_nodes, first_weights = quotient_hypergraph(
        edges, first, node_weights, edge_weights,
    )
    two_step_edges, two_step_nodes, two_step_weights = quotient_hypergraph(
        first_edges, second, first_nodes, first_weights,
    )
    direct_edges, direct_nodes, direct_weights = quotient_hypergraph(
        edges, composed, node_weights, edge_weights,
    )
    assert two_step_edges == direct_edges
    np.testing.assert_array_equal(two_step_weights, direct_weights)
    np.testing.assert_array_equal(two_step_nodes, direct_nodes)

    for labels in itertools.product(range(3), repeat=len(direct_nodes)):
        coarse = np.asarray(labels)
        fine = coarse[composed]
        assert connectivity_cost(fine, edges, edge_weights) == pytest.approx(
            connectivity_cost(coarse, direct_edges, direct_weights)
        )
        np.testing.assert_allclose(
            np.bincount(fine, weights=node_weights, minlength=3),
            np.bincount(coarse, weights=direct_nodes, minlength=3),
        )


def test_capacity_cap_prevents_infeasible_supervertex():
    solver = KahyparLikeSolver()
    result = solver.coarsen(
        [[0, 1], [0, 1], [0, 2], [1, 3]], 4, 2,
        node_weights=[2.0, 2.0, 1.0, 1.0],
        coarsen_to=2, seed=0, enforce_balance_cap=True, epsilon=0.0,
    )
    assert max(result['coarse_node_weights']).item() <= 3.0
    assert result['original_to_coarse'][0] != result['original_to_coarse'][1]


def test_exact_packing_checker_agrees_with_enumeration():
    rng = np.random.default_rng(31)
    for _ in range(100):
        n = int(rng.integers(3, 9))
        q = int(rng.integers(2, 4))
        weights = rng.integers(1, 6, size=n)
        capacity = int(rng.integers(3, 12))
        brute_force = any(
            np.bincount(labels, weights=weights, minlength=q).max() <= capacity
            for labels in itertools.product(range(q), repeat=n)
        )
        assert balanced_packing_feasible(weights, q, capacity) == brute_force


def test_global_guard_blocks_locally_legal_but_infeasible_merge():
    solver = KahyparLikeSolver()
    common = dict(
        node_weights=[4, 4, 2, 2], coarsen_to=3, seed=0,
        enforce_balance_cap=True, epsilon=0.0,
    )
    locally_capped = solver.coarsen([[2, 3]], 4, 2, **common)
    assert locally_capped['coarse_node_weights'].tolist() == [4.0, 4.0, 4.0]
    assert not balanced_packing_feasible(
        locally_capped['coarse_node_weights'], 2, 6.0,
    )

    guarded = solver.coarsen(
        [[2, 3]], 4, 2, enforce_global_feasibility=True, **common,
    )
    assert len(guarded['coarse_groups']) == 4
    assert balanced_packing_feasible(guarded['coarse_node_weights'], 2, 6.0)

    with_alternative = solver.coarsen(
        [[2, 3], [2, 3], [0, 2], [1, 3]], 4, 2,
        enforce_global_feasibility=True, **common,
    )
    assert len(with_alternative['coarse_groups']) == 3
    assert balanced_packing_feasible(with_alternative['coarse_node_weights'], 2, 6.0)


def test_global_guard_rejects_uncheckable_inputs():
    solver = KahyparLikeSolver()
    with pytest.raises(ValueError, match='no feasible balanced partition'):
        solver.coarsen(
            [[0, 1]], 3, 2, node_weights=[4, 4, 4],
            enforce_global_feasibility=True, epsilon=0.0,
        )
    with pytest.raises(ValueError, match='requires use_lsh=False'):
        solver.coarsen(
            [[0, 1]], 2, 2, enforce_global_feasibility=True,
            use_lsh=True, epsilon=0.0,
        )
    with pytest.raises(ValueError, match='at most'):
        solver.coarsen(
            [[0, 1]], 3, 2, enforce_global_feasibility=True,
            global_feasibility_max_nodes=2,
        )


@pytest.mark.parametrize('score_mode', ['hem', 'boundary'])
def test_random_guarded_quotients_remain_balance_feasible(score_mode):
    solver = KahyparLikeSolver()
    checked = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 9))
        q = int(rng.integers(2, 4))
        weights = rng.integers(1, 5, size=n).astype(float)
        capacity = 1.25 * float(weights.sum()) / q
        if not balanced_packing_feasible(weights, q, capacity):
            continue
        edges = [
            rng.choice(n, size=int(rng.integers(2, min(n, 5))), replace=False).tolist()
            for _ in range(12)
        ]
        result = solver.coarsen(
            edges, n, q, node_weights=weights, coarsen_to=3,
            score_mode=score_mode, enforce_global_feasibility=True,
            epsilon=0.25, seed=seed, num_pilots=2,
        )
        assert balanced_packing_feasible(
            result['coarse_node_weights'].numpy(), q, capacity,
        )
        checked += 1
    assert checked >= 20


def test_weighted_solver_quotient_matches_original_cut():
    edges = [[0, 1, 2], [1, 2, 3], [2, 3], [0, 3]]
    weights = [2.0, 3.0, 0.5, 1.5]
    result = KahyparLikeSolver().coarsen(
        edges, 4, 2, coarsen_to=2, seed=2,
        hyperedge_weights=weights, node_weights=[1.0, 2.0, 1.0, 2.0],
    )
    assert len(result['coarse_groups']) == 2
    for labels in itertools.product(range(2), repeat=2):
        coarse = np.asarray(labels)
        fine = coarse[result['original_to_coarse']]
        assert connectivity_cost(fine, edges, weights) == pytest.approx(
            connectivity_cost(
                coarse, result['coarse_hyperedges'],
                result['coarse_hyperedge_weights'],
            )
        )


def test_boundary_score_uses_pilot_disagreement():
    # The two repeated large hyperedges make 2 and 3 attractive to HEM.
    # A strong boundary weight down-ranks pairs that pilots separate.
    edges = [[0, 4], [0, 2, 3], [0, 2, 3], [2, 3, 4, 5],
             [1, 2, 4], [1, 3, 4], [2, 3, 4, 5], [2, 4], [1, 2, 3, 4]]
    solver = KahyparLikeSolver()
    hem = solver.coarsen(
        edges, 6, 2, coarsen_to=3, seed=0,
        enforce_balance_cap=True, epsilon=0.0,
    )
    boundary = solver.coarsen(
        edges, 6, 2, coarsen_to=3, seed=0,
        score_mode='boundary', pilot_assignments=[[0, 0, 0, 1, 1, 1]],
        boundary_weight=0.8, epsilon=0.0,
    )
    assert hem['original_to_coarse'][0] != hem['original_to_coarse'][1]
    assert boundary['original_to_coarse'][0] == boundary['original_to_coarse'][1]


def test_zero_boundary_weight_recovers_capped_hem():
    edges = [[0, 1, 2], [0, 2, 3], [2, 3, 4], [3, 4, 5], [0, 5]]
    solver = KahyparLikeSolver()
    common = dict(coarsen_to=3, seed=7, enforce_balance_cap=True, epsilon=0.0)
    hem = solver.coarsen(edges, 6, 2, score_mode='hem', **common)
    boundary = solver.coarsen(
        edges, 6, 2, score_mode='boundary', boundary_weight=0.0,
        pilot_assignments=[[0, 0, 0, 1, 1, 1]], **common,
    )
    np.testing.assert_array_equal(
        hem['original_to_coarse'], boundary['original_to_coarse'],
    )


def test_lsh_level_is_recorded_for_projection():
    edges = [[0, 1, 2], [0, 1, 3], [2, 3, 4], [3, 4, 5]]
    result = KahyparLikeSolver().coarsen(
        edges, 6, 2, coarsen_to=2, seed=0, use_lsh=True,
        score_mode='hem', enforce_balance_cap=True,
    )
    assert result['hierarchy_stack'][0]['num_nodes'] == 6
    assert sorted(v for group in result['coarse_groups'] for v in group) == list(range(6))
    for labels in itertools.product(range(2), repeat=len(result['coarse_groups'])):
        assignment = np.asarray(labels)
        assert connectivity_cost(
            assignment[result['original_to_coarse']], edges,
        ) == pytest.approx(connectivity_cost(
            assignment, result['coarse_hyperedges'],
            result['coarse_hyperedge_weights'],
        ))

    coarse_assignment = KahyparLikeSolver().initial_partition_greedy(
        result['coarse_hyperedges'], result['coarse_node_weights'], 2,
        hyperedge_weights=result['coarse_hyperedge_weights'], seed=0,
    )
    fine_assignment = vcycle_uncoarsen(
        coarse_assignment, result['hierarchy_stack'], edges, 2,
        HyperRefineSolver(), verbose=False,
    )
    assert fine_assignment.shape == (6,)


@pytest.mark.parametrize('score_mode,use_lsh', [
    ('hem', False), ('hem', True), ('boundary', False),
])
def test_random_weighted_coarsening_preserves_pullback_objective(score_mode, use_lsh):
    """Stress the full multi-level map, including isolated and repeated pins."""
    solver = KahyparLikeSolver()
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 10))
        edges = [
            rng.integers(0, n, size=int(rng.integers(1, 6))).tolist()
            for _ in range(int(rng.integers(4, 14)))
        ]
        edge_weights = rng.integers(1, 7, size=len(edges)).astype(float)
        node_weights = rng.integers(1, 5, size=n).astype(float)
        result = solver.coarsen(
            edges, n, 3, coarsen_to=3, seed=seed,
            node_weights=node_weights, hyperedge_weights=edge_weights,
            score_mode=score_mode, use_lsh=use_lsh,
            enforce_balance_cap=False, num_pilots=2,
        )
        mapping = result['original_to_coarse']
        coarse_n = len(result['coarse_groups'])
        assert sorted(v for group in result['coarse_groups'] for v in group) == list(range(n))
        assert sorted(set(mapping)) == list(range(coarse_n))
        np.testing.assert_allclose(
            np.bincount(mapping, weights=node_weights, minlength=coarse_n),
            result['coarse_node_weights'].numpy(),
        )

        for labels in itertools.product(range(3), repeat=coarse_n):
            coarse = np.asarray(labels)
            fine = coarse[mapping]
            assert connectivity_cost(fine, edges, edge_weights) == pytest.approx(
                connectivity_cost(
                    coarse, result['coarse_hyperedges'],
                    result['coarse_hyperedge_weights'],
                )
            )
            np.testing.assert_allclose(
                np.bincount(fine, weights=node_weights, minlength=3),
                np.bincount(
                    coarse, weights=result['coarse_node_weights'].numpy(), minlength=3,
                ),
            )

        current_edges, current_node_weights, current_edge_weights = quotient_hypergraph(
            edges, np.arange(n), node_weights, edge_weights,
        )
        for level in result['hierarchy_stack']:
            assert level['num_nodes'] == len(current_node_weights)
            assert level['hyperedges'] == current_edges
            next_edges, next_nodes, next_weights = quotient_hypergraph(
                level['hyperedges'], level['remap'][:level['num_nodes']],
                level['node_weights'], level['hyperedge_weights'],
            )
            current_edges = next_edges
            current_node_weights = next_nodes
            current_edge_weights = next_weights
        assert current_edges == result['coarse_hyperedges']
        np.testing.assert_allclose(current_edge_weights, result['coarse_hyperedge_weights'])
