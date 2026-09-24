"""Joint optimization, opt-in integration, and capacity contracts for FEM-IER."""
import itertools

import numpy as np
import pytest

from src.hyper_solver import HyperRefineSolver, KahyparLikeSolver, vcycle_uncoarsen
from src.partition.hyper_ier import refine_fem_ier
from src.partition.hyper_quotient import connectivity_cost


def cooperative_instance():
    # Atoms 0/1 individually cost +4, but their joint change costs -4.
    # Atom 2 costs +20, making all-on worse as well. The optimal subset must
    # therefore be selected, rather than found among single/all/off baselines.
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    edges = [[0, 1], [3, 4], [0, 5], [1, 5], [3, 2], [4, 2], [6, 8], [7, 9]]
    weights = np.array([3., 3., 1., 1., 1., 1., 10., 10.])
    moves = [{0: 1, 3: 0}, {1: 1, 4: 0}, {6: 1, 7: 0}]
    return labels, edges, weights, moves


def test_actual_fem_selects_cooperative_subset(monkeypatch):
    labels, edges, weights, moves = cooperative_instance()
    monkeypatch.setattr('src.partition.hyper_ier.generate_balanced_moves', lambda *args, **kw: moves)
    result, report = refine_fem_ier(labels, edges, 2, hyperedge_weights=weights,
        epsilon=0, rounds=1, num_trials=8, num_steps=100, seed=7)
    step = report['history'][0]
    assert report['initial_native_cut'] == 4
    assert step['deterministic_best_native_cut'] == 4
    assert step['backend_best_native_cut'] == 0
    assert step['backend_beats_deterministic']
    assert step['selected_source'].startswith('fem_')
    assert step['selected_selector'] == [1, 1, 0]
    assert connectivity_cost(result, edges, weights) == 0
    assert np.array_equal(np.bincount(result, minlength=2), [5, 5])


def test_exact_backend_matches_independent_subset_enumeration(monkeypatch):
    labels, edges, weights, moves = cooperative_instance()
    monkeypatch.setattr('src.partition.hyper_ier.generate_balanced_moves', lambda *args, **kw: moves)
    oracle = []
    for bits in itertools.product((0, 1), repeat=3):
        state = labels.copy()
        for bit, move in zip(bits, moves):
            if bit:
                for vertex, block in move.items():
                    state[vertex] = block
        oracle.append(connectivity_cost(state, edges, weights))
    result, report = refine_fem_ier(labels, edges, 2, hyperedge_weights=weights,
                                   epsilon=0, rounds=1, backend='exact')
    assert report['final_native_cut'] == min(oracle)
    assert connectivity_cost(result, edges, weights) == min(oracle)


def test_zero_moves_and_zero_rounds_are_safe_noops():
    labels = np.array([0, 1])
    for options in ({'rounds': 0}, {'rounds': 2}):
        result, report = refine_fem_ier(labels, [], 2, epsilon=0, **options)
        assert np.array_equal(result, labels)
        assert report['final_native_cut'] == 0


def test_infeasible_start_and_unknown_backend_are_explicit():
    with pytest.raises(ValueError, match='feasible'):
        refine_fem_ier(np.array([0, 0]), [[0, 1]], 2, epsilon=0)
    with pytest.raises(ValueError, match='backend'):
        refine_fem_ier(np.array([0, 1]), [[0, 1]], 2, backend='unknown')


def test_refiner_mode_and_vcycle_support_nonunit_weights():
    nodes = np.array([1., 1., 2., 1., 1., 2., 1., 1.])
    edges = [[0, 1, 2], [3, 4, 5], [2, 6], [5, 7], [0, 3, 6, 7]]
    weights = np.array([3., 2., 1., 4., 2.5])
    q, epsilon = 2, .2
    coarsener = KahyparLikeSolver()
    coarse = coarsener.coarsen(edges, len(nodes), q, node_weights=nodes,
        hyperedge_weights=weights, coarsen_to=5, epsilon=epsilon, seed=3,
        enforce_balance_cap=True, enforce_global_feasibility=True)
    # Find a feasible coarse start without depending on a heuristic initializer.
    coarse_nodes = np.asarray(coarse['coarse_node_weights'])
    cap = (1+epsilon)*nodes.sum()/q
    start = next(np.array(bits) for bits in itertools.product(range(q), repeat=len(coarse_nodes))
                 if max(np.bincount(bits, weights=coarse_nodes, minlength=q)) <= cap)
    before = connectivity_cost(start, coarse['coarse_hyperedges'], coarse['coarse_hyperedge_weights'])
    solver = HyperRefineSolver()
    solver.update_params(mode_cycle=('fem_ier', 'flow'), max_imbalance=epsilon,
        ier_rounds=1, ier_num_trials=4, ier_num_steps=30, ier_max_moves=4,
        ier_boundary_pool=8, seed=4, flow_passes=2)
    result = vcycle_uncoarsen(start, coarse['hierarchy_stack'], edges, q, solver,
                              verbose=False, node_weights=nodes, hyperedge_weights=weights)
    assert connectivity_cost(result, edges, weights) <= before
    assert max(np.bincount(result, weights=nodes, minlength=q)) <= cap + 1e-10
    assert solver.last_result['method'] == 'fem_ier_cycle'
    assert [s['method'] for s in solver.last_result['stages']] == ['fem_ier', 'flow']


def test_random_control_is_reproducible_and_nonincreasing():
    labels, edges, weights, _ = cooperative_instance()
    a, ra = refine_fem_ier(labels, edges, 2, hyperedge_weights=weights,
        epsilon=0, rounds=2, backend='random', random_samples=32, seed=9)
    b, rb = refine_fem_ier(labels, edges, 2, hyperedge_weights=weights,
        epsilon=0, rounds=2, backend='random', random_samples=32, seed=9)
    assert np.array_equal(a, b)
    assert ra['final_native_cut'] == rb['final_native_cut'] <= ra['initial_native_cut']
    assert all(step['after_native_cut'] <= step['before_native_cut'] for step in ra['history'])
