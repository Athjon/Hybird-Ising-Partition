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
    assert step['backend_selector_count'] == 8
    assert step['backend_unique_selector_count'] == len(set(step['backend_selector_masks']))
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
    assert report['history'][0]['backend_selector_count'] == 8
    assert report['history'][0]['backend_unique_selector_count'] == 8


@pytest.mark.parametrize('backend', ['fem', 'random', 'exact', 'deterministic', 'pairs'])
def test_zero_moves_and_zero_rounds_are_safe_noops(monkeypatch, backend):
    labels = np.array([0, 1])
    monkeypatch.setattr('src.partition.hyper_ier.generate_balanced_moves', lambda *args, **kw: [])
    for options in ({'rounds': 0}, {'rounds': 2}):
        result, report = refine_fem_ier(labels, [], 2, epsilon=0, backend=backend, **options)
        assert np.array_equal(result, labels)
        assert report['final_native_cut'] == 0
        for step in report['history']:
            assert step['num_moves'] == 0
            assert step['selected_selector'] == []
            assert step['selected_source'] == 'all_off'
            assert step['backend_selector_count'] == step['backend_unique_selector_count'] == 0
            assert step['backend_best_native_cut'] is None
            assert step['backend_native_cuts'] == step['backend_selector_masks'] == []
            assert step['deterministic_native_cuts'] == [0]
            assert not step['accepted'] and not step['backend_beats_deterministic']


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
    for step in ra['history']:
        if step['num_moves']:
            assert step['backend_selector_count'] == 32
            assert step['backend_unique_selector_count'] == len(set(step['backend_selector_masks']))


@pytest.mark.parametrize('backend,expected_cut', [('deterministic', 4.), ('pairs', 0.)])
def test_pair_control_finds_cooperation_missing_from_shared_candidates(monkeypatch, backend, expected_cut):
    labels, edges, weights, moves = cooperative_instance()
    original = labels.copy()
    monkeypatch.setattr('src.partition.hyper_ier.generate_balanced_moves', lambda *args, **kw: moves)

    def no_random_draws(*args, **kwargs):
        raise AssertionError('deterministic/pairs selectors must not draw random samples')

    monkeypatch.setattr('src.partition.hyper_ier.np.random.default_rng', no_random_draws)
    result, report = refine_fem_ier(labels, edges, 2, hyperedge_weights=weights,
        epsilon=0, rounds=1, backend=backend)
    step = report['history'][0]
    assert np.array_equal(labels, original)
    assert connectivity_cost(result, edges, weights) == expected_cut
    assert np.array_equal(np.bincount(result, minlength=2), [5, 5])
    assert step['deterministic_best_native_cut'] == 4
    if backend == 'deterministic':
        assert not step['accepted']
        assert step['selected_source'] == 'all_off'
        assert step['backend_best_native_cut'] is None
        assert step['backend_native_cuts'] == step['backend_selector_masks'] == []
        assert step['backend_selector_count'] == step['backend_unique_selector_count'] == 0
        assert not step['backend_beats_deterministic']
    else:
        assert step['accepted'] and step['backend_beats_deterministic']
        assert step['selected_source'] == 'pairs_0'
        assert step['selected_selector'] == [1, 1, 0]
        assert step['backend_selector_masks'] == [3, 5, 6]
        assert step['backend_selector_count'] == step['backend_unique_selector_count'] == 3
        # Independently apply every reported selector to the original vertices.
        scores = []
        for mask in step['backend_selector_masks']:
            state = labels.copy()
            for j, move in enumerate(moves):
                if mask & (1 << j):
                    for vertex, block in move.items():
                        state[vertex] = block
            scores.append(connectivity_cost(state, edges, weights))
        assert step['backend_native_cuts'] == scores


@pytest.mark.parametrize('backend', ['deterministic', 'pairs'])
def test_shared_candidates_work_when_backend_has_no_selectors(monkeypatch, backend):
    labels = np.array([0, 0, 1, 1])
    edges, weights = [[0, 2], [1, 3]], np.array([2., 3.])
    monkeypatch.setattr('src.partition.hyper_ier.generate_balanced_moves',
                        lambda *args, **kw: [{0: 1, 3: 0}])
    result, report = refine_fem_ier(labels, edges, 2, hyperedge_weights=weights,
        epsilon=0, rounds=1, backend=backend)
    step = report['history'][0]
    assert connectivity_cost(result, edges, weights) == 0
    assert step['deterministic_native_cuts'] == [5., 0., 0.]
    assert step['selected_source'] == 'all_on'  # First of tied all-on/single.
    assert step['selected_selector'] == [1]
    assert step['accepted'] and not step['backend_beats_deterministic']
    assert step['backend_best_native_cut'] is None
    assert step['backend_native_cuts'] == step['backend_selector_masks'] == []
    assert step['backend_selector_count'] == step['backend_unique_selector_count'] == 0
    assert np.array_equal(np.bincount(result, minlength=2), [2, 2])


def test_pair_duplicate_of_all_on_keeps_shared_candidate_attribution(monkeypatch):
    labels, edges, weights, moves = cooperative_instance()
    monkeypatch.setattr('src.partition.hyper_ier.generate_balanced_moves', lambda *args, **kw: moves[:2])
    _, report = refine_fem_ier(labels, edges, 2, hyperedge_weights=weights,
                              epsilon=0, rounds=1, backend='pairs')
    step = report['history'][0]
    assert step['selected_source'] == 'all_on'
    assert step['backend_selector_masks'] == [3]
    assert step['backend_selector_count'] == step['backend_unique_selector_count'] == 1
    assert step['deterministic_best_native_cut'] == step['backend_best_native_cut'] == 0
    assert not step['backend_beats_deterministic']


@pytest.mark.parametrize('backend', ['deterministic', 'pairs'])
def test_new_controls_preserve_weighted_feasibility_and_native_monotonicity(backend):
    labels = np.array([0] * 6 + [1] * 6)
    nodes = np.array([1., 2.] * 6)
    edges = [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11],
             [0, 2, 4], [6, 8, 10], [1, 3, 5], [7, 9, 11], [2, 4, 8, 10]]
    weights = np.array([3., 4., 5., 1., 1.5, 2., 2.5, .5])
    result, report = refine_fem_ier(labels, edges, 2, node_weights=nodes,
        hyperedge_weights=weights, epsilon=0, rounds=3, backend=backend,
        max_moves=5, boundary_pool=12, seed=11)
    assert connectivity_cost(result, edges, weights) <= connectivity_cost(labels, edges, weights)
    assert np.max(np.bincount(result, weights=nodes, minlength=2)) <= 9. + 1e-9
    previous = report['initial_native_cut']
    for step in report['history']:
        assert step['before_native_cut'] == previous
        assert step['after_native_cut'] <= previous
        previous = step['after_native_cut']
        assert max(step['block_loads']) <= 9. + 1e-9
        m = step['num_moves']
        expected = m * (m - 1) // 2 if backend == 'pairs' else 0
        assert step['backend_selector_count'] == step['backend_unique_selector_count'] == expected
        assert all(mask.bit_count() == 2 for mask in step['backend_selector_masks'])
    assert report['final_native_cut'] == previous
