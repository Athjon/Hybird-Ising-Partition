"""Exact small-instance reachability and native Metropolis optimization probes.

Uses saved repaired FEM assignments and previously enumerated finite-gap cases.
Comparisons use a fixed proposal count, not equal wall time. Annealing states
and best-so-far states are optimization trajectories, not Gibbs samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hypergraph.native_mcmc import NativeChain
from src.partition.hyper_objective import capacity_limits
from src.partition.hyper_quotient import connectivity_cost


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_states(edges, node_weights, edge_weights, q, epsilon=0.):
    """Enumerate every labeled feasible assignment, including label symmetries."""
    n = len(node_weights)
    if q ** n > 2_000_000:
        raise ValueError('this oracle is intentionally limited to small inputs')
    codes = np.arange(q ** n, dtype=np.int64)
    powers = q ** np.arange(n, dtype=np.int64)
    states = ((codes[:, None] // powers) % q).astype(np.int8)
    capacity, tolerance = capacity_limits(np.asarray(node_weights), q, epsilon)
    feasible = np.ones(len(states), dtype=bool)
    for label in range(q):
        loads = (states == label) @ np.asarray(node_weights)
        feasible &= loads <= capacity + tolerance
    states, codes = states[feasible], codes[feasible]
    values = np.zeros(len(states))
    for edge, weight in zip(edges, edge_weights):
        pins = sorted(set(edge))
        if not pins:
            continue
        occupied = sum(np.any(states[:, pins] == label, axis=1) for label in range(q))
        values += weight * (occupied - 1)
    if not len(states):
        raise ValueError('enumerated input is infeasible')
    return states, codes, values


def profile_restricted_optimum(states, values, start, node_weights, q):
    """At exact capacity, local pair swaps preserve per-block weight counts.

    All positive weights and total load=q*capacity imply no single move or
    unequal-weight pair swap is feasible. Equal-weight transpositions connect
    exactly this profile class. This claim does not extend to epsilon>0.
    """
    node_weights = np.asarray(node_weights)
    allowed = np.ones(len(states), dtype=bool)
    profile = []
    for weight in np.unique(node_weights):
        selected = node_weights == weight
        counts = []
        for label in range(q):
            count = int(np.sum(np.asarray(start)[selected] == label))
            allowed &= np.sum(states[:, selected] == label, axis=1) == count
            counts.append(count)
        profile.append({'weight': float(weight), 'counts_by_block': counts})
    assert np.any(allowed)
    return {'profile': profile, 'reachable_labeled_states': int(allowed.sum()),
            'restricted_optimum': float(values[allowed].min())}


def canonical_binary(mask, n):
    return int(mask ^ ((1 << n) - 1) if mask & 1 else mask)


def exact_swap_barrier(codes, values, n, source, target):
    """Minimax energy along all balanced one-for-one swaps, modulo label flip."""
    energies = {canonical_binary(int(code), n): float(value) for code, value in zip(codes, values)}
    source, target = canonical_binary(source, n), canonical_binary(target, n)
    distance, parent = {source: energies[source]}, {}
    heap = [(energies[source], source)]
    while heap:
        height, mask = heapq.heappop(heap)
        if height != distance[mask]:
            continue
        if mask == target:
            path = [target]
            while path[-1] != source:
                path.append(parent[path[-1]])
            path.reverse()
            return {'minimum_path_max_energy': height,
                    'barrier_above_optimum': height - energies[source],
                    'canonical_path_masks': path,
                    'path_energies': [energies[m] for m in path]}
        ones = [i for i in range(n) if (mask >> i) & 1]
        zeros = [i for i in range(n) if not (mask >> i) & 1]
        for i, j in itertools.product(ones, zeros):
            neighbor = canonical_binary(mask ^ (1 << i) ^ (1 << j), n)
            candidate = max(height, energies[neighbor])
            if candidate < distance.get(neighbor, float('inf')):
                distance[neighbor], parent[neighbor] = candidate, mask
                heapq.heappush(heap, (candidate, neighbor))
    raise AssertionError('balanced bisection swap graph should be connected')


def inputs(repair, geometry):
    for n, q in itertools.product((8, 12), (2, 3)):
        name = f'weighted-n{n:02d}-q{q}'
        path = repair / f'{name}-input.json'
        data = json.loads(path.read_text())
        assignment_path = repair / f'{name}-assignments.npz'
        with np.load(assignment_path) as arrays:
            start = arrays['fem_final_assignment'].copy()
        yield {'name': name, 'family': 'weighted', 'q': q, 'edges': data['edges'],
               'nodes': np.array(data['node_weights']), 'weights': np.array(data['hyperedge_weights']),
               'start': start, 'input_hashes': {str(path): digest(path), str(assignment_path): digest(assignment_path)}}
    for name in ('random-n08-i003', 'random-n08-i009', 'random-n12-i008'):
        path = geometry / 'instances' / f'{name}.json'
        data = json.loads(path.read_text())
        optimal = sorted(mask for mask, value in zip(data['feasible_class_masks'], data['native_energy_by_feasible_mask'])
                         if value == data['exact_optimum'])
        assert len(optimal) == 2
        start = np.array([(optimal[0] >> i) & 1 for i in range(data['n'])], dtype=np.int64)
        yield {'name': name, 'family': 'finite_gap', 'q': 2, 'edges': data['hyperedges'],
               'nodes': np.array(data['node_weights']), 'weights': np.array(data['hyperedge_weights']),
               'start': start, 'optimal_classes': optimal,
               'input_hashes': {str(path): digest(path)}}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        raise ValueError('choose a new output directory')
    paths = [Path(__file__).resolve(), ROOT / 'benchmarks/hypergraph/native_mcmc.py',
             ROOT / 'src/partition/hyper_objective.py', ROOT / 'src/partition/hyper_refine_contract.py',
             ROOT / 'src/partition/hyper_quotient.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
              'config': {'seeds': args.seeds, 'steps': args.steps, 'beta_min': .05, 'beta_max': 8.,
                         'budget': 'equal proposals, not equal time', 'block_sizes': [2, 3, 4, 6],
                         'proposal_mixtures': {'local': {'swap': 1.}, 'block': {'swap': .75, 'block': .25}},
                         'weighted_initialization': 'saved repaired FEM final assignment',
                         'finite_gap_initialization': 'first canonical optimal class'},
              'source_sha256_before': hashes, 'instances': []}
    for item in inputs(args.repair, args.geometry):
        name, q, start = item['name'], item['q'], item['start']
        nodes, edges, weights = item['nodes'], item['edges'], item['weights']
        n = len(nodes)
        states, codes, values = exact_states(edges, nodes, weights, q)
        optimum = float(values.min())
        initial_cost = float(connectivity_cost(start, edges, weights))
        record = {'name': name, 'family': item['family'], 'n': n, 'q': q,
                  'input_hashes': item['input_hashes'], 'initial_assignment': start.tolist(),
                  'initial_cost': initial_cost, 'exact_optimum': optimum,
                  'feasible_labeled_states': len(states), 'runs': []}
        if item['family'] == 'weighted':
            record['local_reachability'] = profile_restricted_optimum(states, values, start, nodes, q)
        else:
            record['optimal_classes'] = item['optimal_classes']
            record['exact_swap_barrier'] = exact_swap_barrier(codes, values, n, *item['optimal_classes'])
        traces = {}
        for method in ('zero_local', 'anneal_local', 'anneal_block'):
            for seed in args.seeds:
                clock_start = time.perf_counter()
                chain = NativeChain(edges, nodes, weights, q, 0., start.copy(), seed=seed)
                trajectory_codes = np.empty(args.steps + 1, dtype=np.int64)
                trajectory_costs = np.empty(args.steps + 1)
                powers = q ** np.arange(n, dtype=np.int64)
                trajectory_codes[0] = int(start @ powers)
                trajectory_costs[0] = initial_cost
                best_cost, best_state = initial_cost, start.copy()
                for step in range(args.steps):
                    beta = float('inf') if method == 'zero_local' else .05 * (8. / .05) ** (step / max(1, args.steps - 1))
                    chain.step(beta, swap_probability=.75 if method == 'anneal_block' else 1.,
                               block_probability=.25 if method == 'anneal_block' else 0.,
                               block_sizes=(2, 3, 4, 6))
                    trajectory_codes[step+1] = int(chain.assignment @ powers)
                    trajectory_costs[step+1] = chain.energy
                    if chain.energy < best_cost:
                        best_cost, best_state = float(chain.energy), chain.assignment.copy()
                elapsed = time.perf_counter() - clock_start
                # Full oracle checking of every recorded state catches incremental drift
                # and infeasible intermediate states, not only final-state errors.
                locations = np.searchsorted(codes, trajectory_codes)
                assert np.all(locations < len(codes))
                assert np.array_equal(codes[locations], trajectory_codes)
                np.testing.assert_allclose(values[locations], trajectory_costs, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(connectivity_cost(best_state, edges, weights), best_cost)
                result = {'method': method, 'seed': seed, 'seconds': elapsed,
                          'best_cost': best_cost, 'best_gap': best_cost-optimum,
                          'final_cost': float(chain.energy), 'best_assignment': best_state.tolist(),
                          'final_assignment': chain.assignment.tolist(),
                          'proposal_statistics': chain.stats,
                          'distinct_labeled_states_visited': int(len(np.unique(trajectory_codes))),
                          'all_states_oracle_verified': True,
                          'accepted_uphill_steps': int(np.sum(np.diff(trajectory_costs) > 1e-10))}
                if item['family'] == 'finite_gap':
                    canonical = np.array([canonical_binary(int(c), n) for c in trajectory_codes])
                    other = canonical == item['optimal_classes'][1]
                    result['visited_other_optimal_class'] = bool(np.any(other))
                    result['first_other_optimal_class_step'] = int(np.flatnonzero(other)[0]) if np.any(other) else None
                    result['optimal_classes_visited'] = sorted(set(canonical[np.isclose(trajectory_costs, optimum)].tolist()))
                record['runs'].append(result)
                key = f'{method}_seed{seed}'
                traces[key + '_state_codes'] = trajectory_codes
                traces[key + '_energies'] = trajectory_costs
            selected = [r for r in record['runs'] if r['method'] == method]
            print(name, method, 'best gaps', dict(Counter(r['best_gap'] for r in selected)),
                  'other class', sum(r.get('visited_other_optimal_class', False) for r in selected), flush=True)
        record['aggregate'] = []
        for method in ('zero_local', 'anneal_local', 'anneal_block'):
            rows = [r for r in record['runs'] if r['method'] == method]
            record['aggregate'].append({'method': method, 'runs': len(rows),
                'optimal_hits': sum(abs(r['best_gap']) < 1e-10 for r in rows),
                'mean_best_gap': float(np.mean([r['best_gap'] for r in rows])),
                'median_seconds': float(np.median([r['seconds'] for r in rows])),
                'visited_other_optimal_class': sum(r.get('visited_other_optimal_class', False) for r in rows)})
        np.savez_compressed(args.output / (name + '-trajectories.npz'), **traces)
        save(args.output / (name + '.json'), record)
        report['instances'].append(record)
        save(args.output / 'progress.json', report)
    report['source_sha256_after'] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report['sources_unchanged_during_run'] = hashes == report['source_sha256_after']
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(args.output / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', type=Path, default=ROOT / 'benchmarks/hypergraph/results/fem-repair-20260924')
    parser.add_argument('--geometry', type=Path, default=ROOT / 'benchmarks/hypergraph/results/ogp-diagnostic-20260924/geometry')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    if args.steps < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('steps must be positive and seeds distinct')
    run(args)
