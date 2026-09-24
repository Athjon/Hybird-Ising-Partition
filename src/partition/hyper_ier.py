"""FEM selection of disjoint balanced moves for native hypergraph refinement.

The outer loop regenerates moves at each accepted/current partition. Binary
FEM variables select entire move atoms, preserving correlations between their
pins. Every discrete combination preserves capacity and is scored with native
weighted km1. This is an opt-in research IER path, not the default refiner.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .hyper_ier_candidates import generate_balanced_moves
from .hyper_ier_objective import JointMoveObjective
from .hyper_refine_contract import capacity_state, normalized_hyperedges, validated_weights
from .hyper_quotient import connectivity_cost


def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, bool) or int(value) != value or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer')
    return int(value)


def _masks(selectors):
    return [sum(int(bit) << index for index, bit in enumerate(row)) for row in selectors]


def refine_fem_ier(assignment, hyperedges, q, *, node_weights=None,
                   hyperedge_weights=None, epsilon=.03, rounds=2, max_moves=24,
                   boundary_pool=96, num_trials=8, num_steps=100, seed=1,
                   backend='fem', random_samples=None, allow_triples=True,
                   pool_strategy='local',
                   learning_rate=.08, betamin=.5, betamax=50., dev='cpu',
                   exact_max_moves=16):
    """Return ``(partition, diagnostics)`` after cyclic joint-move selection.

    Every backend shares all-off, all-on and all single-atom candidates.
    ``fem`` adds learned selectors; ``random`` adds ``num_trials*num_steps``
    independent Bernoulli selectors unless random_samples is explicit.
    ``exact`` enumerates a bounded candidate subproblem, never the full graph.
    Finite budgets and the generated neighborhood can prevent improvement.
    """
    rounds = _positive_integer(rounds, 'rounds', allow_zero=True)
    max_moves = _positive_integer(max_moves, 'max_moves')
    boundary_pool = _positive_integer(boundary_pool, 'boundary_pool')
    num_trials = _positive_integer(num_trials, 'num_trials')
    num_steps = _positive_integer(num_steps, 'num_steps')
    seed = _positive_integer(seed, 'seed', allow_zero=True)
    if backend not in ('fem', 'random', 'exact'):
        raise ValueError('IER backend must be fem, random, or exact')
    if random_samples is None:
        random_samples = num_trials * num_steps
    random_samples = _positive_integer(random_samples, 'random_samples')
    exact_max_moves = _positive_integer(exact_max_moves, 'exact_max_moves', allow_zero=True)
    current, nodes, loads, capacity, tolerance = capacity_state(assignment, q, node_weights, epsilon)
    current = current.copy()
    if np.any(loads > capacity + tolerance):
        raise ValueError('FEM-IER requires a capacity-feasible initial partition')
    edges = normalized_hyperedges(hyperedges, len(current))
    weights = validated_weights(hyperedge_weights, len(edges), 'hyperedge_weights')
    initial_cost = float(connectivity_cost(current, edges, weights))
    report = {'method': 'fem_ier', 'backend': backend, 'initial_native_cut': initial_cost,
              'initial_loads': loads.tolist(), 'capacity': capacity, 'history': [],
              'config': {'rounds': rounds, 'max_moves': max_moves, 'boundary_pool': boundary_pool,
                         'num_trials': num_trials, 'num_steps': num_steps, 'random_samples': random_samples,
                         'allow_triples': bool(allow_triples), 'seed': seed,
                         'pool_strategy': pool_strategy,
                         'learning_rate': learning_rate, 'betamin': betamin, 'betamax': betamax,
                         'device': str(dev), 'dtype': 'float64'}}
    start_all = time.perf_counter()
    for index in range(rounds):
        round_start = time.perf_counter()
        round_seed = seed + 104729 * index
        before = float(connectivity_cost(current, edges, weights))
        moves = generate_balanced_moves(current, edges, q, nodes, weights, epsilon,
            max_moves=max_moves, boundary_pool=boundary_pool, seed=round_seed,
            allow_triples=allow_triples, pool_strategy=pool_strategy)
        record = {'round': index, 'seed': round_seed, 'before_native_cut': before,
                  'moves': [{str(v): int(label) for v, label in move.items()} for move in moves],
                  'num_moves': len(moves), 'candidate_generation_seconds': time.perf_counter()-round_start}
        if not moves:
            record.update(after_native_cut=before, accepted=False, reason='no_balanced_disjoint_moves',
                          total_seconds=time.perf_counter()-round_start)
            report['history'].append(record)
            continue
        objective = JointMoveObjective(current, edges, moves, q, node_weights=nodes,
                                        hyperedge_weights=weights, epsilon=epsilon)
        record['interaction_summary'] = objective.interaction_summary()
        m = len(moves)
        deterministic = np.concatenate((np.zeros((1, m), dtype=np.int64),
                                        np.ones((1, m), dtype=np.int64), np.eye(m, dtype=np.int64)))
        deterministic_names = ['all_off', 'all_on'] + [f'single_{j}' for j in range(m)]
        backend_start = time.perf_counter()
        probabilities = None
        if backend == 'fem':
            from fem import FEM
            case = FEM.from_couplings('customize', m, len(edges), torch.empty(0),
                customize_expected_func=objective.expectation,
                customize_infer_func=objective.inference, customize_energy_func=objective.energy)
            case.set_up_solver(num_trials, num_steps, q=2, dev=dev, dtype=torch.float64,
                seed=round_seed, manual_grad=False, optimizer='adam', learning_rate=learning_rate,
                anneal='exp', betamin=betamin, betamax=betamax, h_factor=.1,
                use_adaptive_annealing=False, use_compile=False)
            configs, _ = case.solve()
            backend_selectors = configs.detach().cpu().numpy().astype(np.int64)
            probabilities = case.solver.probabilities[..., 1].cpu().numpy().tolist()
        elif backend == 'random':
            backend_selectors = np.random.default_rng(round_seed).integers(0, 2, size=(random_samples, m))
        else:
            if m > exact_max_moves:
                raise ValueError('candidate pool exceeds exact_max_moves; no exact certificate was computed')
            backend_selectors = ((np.arange(1 << m, dtype=np.int64)[:, None]
                                  >> np.arange(m, dtype=np.int64)) & 1)
        record['backend_seconds'] = time.perf_counter()-backend_start
        # Evaluate the exact native expectation at deterministic binary states.
        # This is vectorized over affected hyperedges by JointMoveObjective.
        selectors = np.concatenate((deterministic, backend_selectors))
        with torch.no_grad():
            scores = np.concatenate([
                objective.energy(None, torch.as_tensor(selectors[offset:offset+256])).detach().cpu().numpy()
                for offset in range(0, len(selectors), 256)
            ])
        deterministic_best = float(scores[:len(deterministic)].min())
        backend_best = float(scores[len(deterministic):].min())
        score_tolerance = 1e-12 * max(abs(before), abs(deterministic_best),
                                    abs(backend_best), np.finfo(float).tiny)
        winner = int(np.argmin(scores))
        selected = selectors[winner]
        proposed = objective.apply(selected)
        actual = float(connectivity_cost(proposed, edges, weights))
        _, _, next_loads, _, _ = capacity_state(proposed, q, nodes, epsilon)
        if not np.isclose(actual, scores[winner], rtol=1e-12, atol=1e-12):
            raise AssertionError('joint-move score disagrees with independent native cut')
        if np.any(next_loads > capacity + tolerance):
            raise AssertionError('joint move exceeded original block capacities')
        # Independent native acceptance remains authoritative even when tiny
        # floating-point ordering differences affect the selector scores.
        accepted = actual < before
        if accepted:
            current = proposed
        source = deterministic_names[winner] if winner < len(deterministic) else f'{backend}_{winner-len(deterministic)}'
        record.update(after_native_cut=actual if accepted else before, accepted=accepted,
            proposed_native_cut=actual,
            selected_source=source, selected_selector=selected.tolist(),
            selected_move_count=int(selected.sum()), deterministic_best_native_cut=deterministic_best,
            backend_best_native_cut=backend_best,
            attribution_tolerance=score_tolerance,
            backend_beats_deterministic=backend_best < deterministic_best-score_tolerance,
            backend_selector_masks=_masks(backend_selectors),
            backend_native_cuts=scores[len(deterministic):].tolist(),
            deterministic_native_cuts=scores[:len(deterministic)].tolist(),
            block_loads=next_loads.tolist(), total_seconds=time.perf_counter()-round_start)
        if probabilities is not None:
            record['fem_move_probabilities'] = probabilities
        report['history'].append(record)
    _, _, final_loads, _, _ = capacity_state(current, q, nodes, epsilon)
    report.update(final_native_cut=float(connectivity_cost(current, edges, weights)),
                  final_loads=final_loads.tolist(), seconds=time.perf_counter()-start_all,
                  accepted_rounds=sum(h['accepted'] for h in report['history']))
    return current, report
