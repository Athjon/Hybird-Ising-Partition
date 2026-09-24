"""Independent delta and detailed-balance oracles for native Metropolis search."""

import itertools
import math

import numpy as np
import pytest

from benchmarks.hypergraph.native_mcmc import (
    NativeChain, metropolis_acceptance, run_annealing,
)


def native_cost(labels, edges, weights):
    return sum(weight * max(0, len({int(labels[v]) for v in edge}) - 1)
               for edge, weight in zip(edges, weights))


def test_all_joint_deltas_and_cache_updates_match_brute_force_weighted_q3():
    edges = [[0, 1, 1, 2], [1, 2, 3], [0, 3], [0], [2, 2], [], [0, 1, 2]]
    weights = [2.3, 1.5, .7, 99, 4, 6, .2]
    nodes = [2, 1, 1, 2]
    initial = np.array([0, 1, 1, 2])
    chain = NativeChain(edges, nodes, weights, 3, .5, initial, seed=4)
    before = native_cost(initial, edges, weights)
    for candidate in itertools.product(range(3), repeat=4):
        changes = dict(enumerate(candidate))
        proposal = chain.proposal(changes)
        after = native_cost(candidate, edges, weights)
        assert proposal.delta == pytest.approx(after - before, abs=1e-12)
        feasible = np.bincount(candidate, weights=nodes, minlength=3).max() <= 3
        assert proposal.feasible == feasible
        np.testing.assert_array_equal(chain.assignment, initial)
        if feasible:
            other = NativeChain(edges, nodes, weights, 3, .5, initial, seed=4)
            other._apply(other.proposal(changes))
            assert other.energy == pytest.approx(after, abs=1e-12)
            other.assert_consistent()


def test_shared_edge_swap_is_evaluated_jointly_not_as_two_single_moves():
    chain = NativeChain([[0, 1, 1]], [1, 1], [7], 2, 0, [0, 1], seed=1)
    assert chain.move_proposal(0, 1).delta == -7
    assert chain.move_proposal(1, 0).delta == -7
    joint = chain.swap_proposal(0, 1)
    assert joint.delta == 0
    assert joint.feasible
    chain._apply(joint)
    chain.assert_consistent()
    assert chain.energy == 7


def test_three_node_transposition_exchanges_two_unit_nodes_for_one_weight_two():
    chain = NativeChain([[0, 3], [1, 4], [2, 4]], [2, 1, 1, 2, 2], [1, 1, 1],
                        2, 0, [0, 1, 1, 0, 1], seed=5)
    assert not chain.swap_proposal(0, 1).feasible
    assert not chain.move_proposal(0, 1).feasible
    proposal = chain.block_proposal([0, 1, 2], 0, 1)
    assert proposal.feasible and proposal.delta == 3
    chain._apply(proposal)
    np.testing.assert_array_equal(chain.assignment, [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(chain.loads, [4, 4])
    reverse = chain.block_proposal([0, 1, 2], 0, 1)
    assert reverse.feasible and reverse.delta == -3
    chain._apply(reverse)
    np.testing.assert_array_equal(chain.assignment, [0, 1, 1, 0, 1])
    chain.assert_consistent()


@pytest.mark.parametrize('block_probability', [0.0, .35])
def test_complete_fixed_beta_transition_matrix_has_gibbs_detailed_balance(block_probability):
    # Repeated pins, singleton/empty edges, weighted q=3 and weighted capacities.
    edges = [[0, 1, 1], [1, 2], [0, 1, 2], [2], [0, 2], []]
    weights = [1.5, .7, .3, 7, 2, 99]
    node_weights = [2, 1, 1]
    n, q, beta, swap_probability = 3, 3, .7, .25
    states = [state for state in itertools.product(range(q), repeat=n)
              if np.bincount(state, weights=node_weights, minlength=q).max() <= 2]
    state_index = {state: index for index, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)))
    sizes = (2, 3)
    for row, state in enumerate(states):
        chain = NativeChain(edges, node_weights, weights, q, .5, state)

        def add(proposal, mass):
            if not proposal.feasible:
                transition[row, row] += mass
                return
            candidate = list(state)
            for vertex, label in proposal.updates:
                candidate[vertex] = label
            column = state_index[tuple(candidate)]
            accept = metropolis_acceptance(proposal.delta, beta)
            transition[row, column] += mass * accept
            transition[row, row] += mass * (1 - accept)

        move_mass = (1 - swap_probability - block_probability) / (n * q)
        for vertex in range(n):
            for label in range(q):
                add(chain.move_proposal(vertex, label), move_mass)
        swap_mass = swap_probability / (n * (n - 1))
        for first in range(n):
            for second in range(n):
                if first != second:
                    add(chain.swap_proposal(first, second), swap_mass)
        for size in sizes:
            block_mass = block_probability / (len(sizes) * math.comb(n, size) * q * (q - 1))
            for selected in itertools.combinations(range(n), size):
                for first, second in itertools.permutations(range(q), 2):
                    add(chain.block_proposal(selected, first, second), block_mass)
    energies = np.asarray([native_cost(state, edges, weights) for state in states])
    target = np.exp(-beta * (energies - energies.min()))
    target /= target.sum()
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-14)
    assert np.all(transition >= 0)
    assert np.all(np.diag(transition) > 0)
    flux = target[:, None] * transition
    np.testing.assert_allclose(flux, flux.T, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(target @ transition, target, atol=1e-14, rtol=1e-12)


def test_illegal_swaps_are_single_attempt_self_loops_without_resampling():
    chain = NativeChain([[0, 1, 2]], [2, 1, 1], [1], 2, 0, [0, 1, 1], seed=72)
    for _ in range(100):
        result = chain.step(1, swap_probability=1)
        assert not result['accepted']
    np.testing.assert_array_equal(chain.assignment, [0, 1, 1])
    assert chain.stats['attempted'] == 100
    assert chain.stats['capacity_rejected'] > 0
    assert chain.stats['no_op'] > 0
    assert chain.stats['self_loops'] == 100
    assert chain.stats['capacity_rejected'] + chain.stats['no_op'] == 100


def test_annealing_returns_reproducible_feasible_optimization_trajectory_and_stats():
    kwargs = dict(edges=[[0, 3], [1, 4], [2, 4]], node_weights=[2, 1, 1, 2, 2],
                  edge_weights=[1, 2, 3], q=2, epsilon=0,
                  assignment=[0, 1, 1, 0, 1], seed=91)
    first, second = NativeChain(**kwargs), NativeChain(**kwargs)
    options = dict(steps=250, beta_min=0, beta_max=0, swap_probability=.25,
                   block_probability=.5, block_sizes=(3,), history_stride=1)
    result = run_annealing(first, **options)
    again = run_annealing(second, **options)
    assert result['history'] == again['history']
    np.testing.assert_array_equal(result['final_assignment'], again['final_assignment'])
    assert len(result['history']) == 251
    assert result['stats']['uphill_accepted'] > 0
    assert result['stats']['accepted'] + result['stats']['self_loops'] == 250
    assert result['stats']['attempted'] == sum(counts['attempted']
                                             for counts in result['stats']['by_kind'].values())
    assert 'not certified Gibbs samples' in result['interpretation']
    best = [row['best_energy'] for row in result['history']]
    assert all(a >= b for a, b in zip(best, best[1:]))
    for row in result['history']:
        assert max(row['loads']) <= 4
        assert row['energy'] == native_cost(row['assignment'], kwargs['edges'], kwargs['edge_weights'])
    first.assert_consistent()


def test_zero_temperature_acceptance_and_schedule():
    assert metropolis_acceptance(2, math.inf) == 0
    assert metropolis_acceptance(0, math.inf) == 1
    assert metropolis_acceptance(-2, math.inf) == 1
    assert metropolis_acceptance(2, 0) == 1
    chain = NativeChain([[0, 3], [1, 4], [2, 4]], [2, 1, 1, 2, 2], [1, 1, 1],
                        2, 0, [0, 1, 1, 0, 1], seed=5)
    result = run_annealing(chain, 100, math.inf, math.inf, swap_probability=.25,
                           block_probability=.5, block_sizes=(3,))
    assert result['stats']['uphill_accepted'] == 0
    assert result['final_energy'] == 0


@pytest.mark.parametrize('scale', [1., 1e-6, 1e-12])
def test_capacity_checks_respect_node_weight_units(scale):
    chain = NativeChain([], np.ones(4) * scale, [], 2, 0, [0, 0, 1, 1])
    assert not chain.move_proposal(2, 0).feasible


@pytest.mark.parametrize('kwargs', [
    {'swap_probability': .7, 'block_probability': .4},
    {'swap_probability': -.1},
    {'block_probability': np.nan},
    {'block_sizes': (2, 4)},
    {'block_sizes': (2, 2)},
    {'block_sizes': ()},
])
def test_invalid_mixture_or_explicit_block_sizes_raise_before_attempt(kwargs):
    chain = NativeChain([[0, 1, 2]], [2, 1, 1], [1], 2, 0, [0, 1, 1])
    with pytest.raises(ValueError):
        chain.step(1, **kwargs)
    assert chain.stats['attempted'] == 0


def test_invalid_initial_assignment_is_not_repaired():
    with pytest.raises(ValueError, match='initial assignment exceeds'):
        NativeChain([[0, 1]], [2, 1, 1], [1], 2, 0, [0, 0, 1])
