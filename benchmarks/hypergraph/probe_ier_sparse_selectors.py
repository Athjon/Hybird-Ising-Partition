"""Offline sparse-selector oracle on saved FEM-IER candidate pools.

Replays the recorded trajectory; sparse winners never become the next start.
Enumerates only shared deterministic selectors and all two/three-atom subsets.
Uses an independent NumPy native km1 scorer, without importing any solver code.
This is a diagnostic of saved pools, not a new timed or end-to-end solver arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_cut(assignment, edges, weights):
    return float(sum(float(w) * max(0, len({int(assignment[v]) for v in edge}) - 1)
                     for edge, w in zip(edges, weights)))


def read_hgr(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('%')]
    header = [int(v) for v in lines[0].split()]
    if len(header) > 2 and header[2] != 0:
        raise ValueError('this diagnostic accepts only the saved unweighted IBM hgr inputs')
    ne, n = header[:2]
    if len(lines) != ne + 1:
        raise ValueError('unexpected hgr edge count')
    edges = [sorted({int(v) - 1 for v in line.split()}) for line in lines[1:]]
    if any(v < 0 or v >= n for edge in edges for v in edge):
        raise ValueError('hgr vertex outside range')
    return edges, np.ones(n), np.ones(ne)


def selector_matrix(m):
    # Preserve the production deterministic tie order, then enumerate sparse sets.
    subsets = [(), tuple(range(m))] + [(i,) for i in range(m)]
    names = ['all_off', 'all_on'] + [f'single_{i}' for i in range(m)]
    for k in (2, 3):
        for subset in combinations(range(m), k):
            subsets.append(subset)
            names.append(f'size_{k}')
    selectors = np.zeros((len(subsets), m), dtype=np.int64)
    for i, subset in enumerate(subsets):
        selectors[i, list(subset)] = 1
    return selectors, names


class IndependentScorer:
    """Count each block's pins after disjoint atom updates on affected edges."""

    def __init__(self, start, edges, weights, nodes, q, epsilon, moves):
        self.start = np.asarray(start, dtype=np.int64)
        self.edges, self.weights, self.nodes = edges, weights, nodes
        self.q, self.moves = q, moves
        self.base_cut = native_cut(self.start, edges, weights)
        self.capacity = float(np.sum(nodes)) * (1 + epsilon) / q
        self.tolerance = max(float(np.sum(nodes)), 1.) * 1e-10
        self.base_loads = np.bincount(self.start, weights=nodes, minlength=q)
        self.load_delta = np.zeros((len(moves), q))
        owner, target = {}, {}
        for j, move in enumerate(moves):
            for v, label in move.items():
                if v in owner or not 0 <= v < len(start) or not 0 <= label < q:
                    raise AssertionError('candidate atoms must be disjoint valid moves')
                owner[v], target[v] = j, label
                self.load_delta[j, self.start[v]] -= nodes[v]
                self.load_delta[j, label] += nodes[v]
        bases, deltas, affected_weights = [], [], []
        for edge, weight in zip(edges, weights):
            touched = [v for v in edge if v in owner]
            if not touched:
                continue
            base = np.bincount(self.start[edge], minlength=q)
            delta = np.zeros((len(moves), q), dtype=np.int64)
            for v in touched:
                delta[owner[v], self.start[v]] -= 1
                delta[owner[v], target[v]] += 1
            bases.append(base)
            deltas.append(delta)
            affected_weights.append(weight)
        self.base_counts = np.asarray(bases, dtype=np.int64).reshape(-1, q)
        self.deltas = (np.stack(deltas, axis=1) if deltas
                       else np.zeros((len(moves), 0, q), dtype=np.int64))
        self.affected_weights = np.asarray(affected_weights)
        self.base_affected = float(np.dot((self.base_counts > 0).sum(axis=1) - 1,
                                          self.affected_weights))

    def score(self, selectors):
        loads = self.base_loads + selectors @ self.load_delta
        if np.any(loads > self.capacity + self.tolerance):
            raise AssertionError('sparse selector violates original weighted capacity')
        result = []
        for offset in range(0, len(selectors), 256):
            batch = selectors[offset:offset + 256]
            counts = self.base_counts[None, :, :] + np.einsum(
                'bm,meq->beq', batch, self.deltas, optimize=True)
            if np.any(counts < 0):
                raise AssertionError('negative pin occupancy')
            costs = ((counts > 0).sum(axis=2) - 1) @ self.affected_weights
            result.extend((self.base_cut - self.base_affected + costs).tolist())
        return np.asarray(result)

    def apply(self, selector):
        result = self.start.copy()
        for active, move in zip(selector, self.moves):
            if active:
                for v, label in move.items():
                    result[v] = label
        return result

    def direct_check(self, selector, expected):
        result = self.apply(selector)
        actual = native_cut(result, self.edges, self.weights)
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-10):
            raise AssertionError('vectorized scorer disagrees with full native km1')
        loads = np.bincount(result, weights=self.nodes, minlength=self.q)
        if np.any(loads > self.capacity + self.tolerance):
            raise AssertionError('independently reconstructed selector is infeasible')
        return result


def describe(selector, cost, before, single_costs, moves):
    ids = np.flatnonzero(selector).tolist()
    additive = before + sum(float(single_costs[i]) - before for i in ids)
    return {'native_cut': float(cost), 'selected_atoms': ids,
            'selector_mask': sum(1 << i for i in ids),
            'atom_count': len(ids), 'moved_vertex_count': sum(len(moves[i]) for i in ids),
            'single_atom_native_cuts': [float(single_costs[i]) for i in ids],
            'additive_native_cut': additive,
            'synergy_gain_over_additive': additive - float(cost),
            'gain_from_round_start': before - float(cost)}


def probe_round(start, h, edges, weights, nodes, q, epsilon):
    moves = [{int(v): int(label) for v, label in move.items()} for move in h['moves']]
    m = len(moves)
    scorer = IndependentScorer(start, edges, weights, nodes, q, epsilon, moves)
    before = scorer.base_cut
    if not np.isclose(before, h['before_native_cut'], rtol=1e-12, atol=1e-10):
        raise AssertionError('replayed round start differs from saved cut')
    selectors, names = selector_matrix(m)
    costs = scorer.score(selectors)
    n_det = m + 2
    np.testing.assert_allclose(costs[:n_det], h['deterministic_native_cuts'], rtol=1e-12, atol=1e-10)
    single_costs = costs[2:n_det]
    selected = np.asarray(h['selected_selector'], dtype=np.int64)
    selected_cost = float(scorer.score(selected[None, :])[0])
    np.testing.assert_allclose(selected_cost, h['proposed_native_cut'], rtol=1e-12, atol=1e-10)
    proposed = scorer.direct_check(selected, selected_cost)

    def best(indices):
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return None
        index = int(indices[np.argmin(costs[indices])])
        scorer.direct_check(selectors[index], costs[index])
        record = describe(selectors[index], costs[index], before, single_costs, moves)
        record.update(candidate_name=names[index], candidates_evaluated=len(indices),
                      tied_minimum_rows=int(np.count_nonzero(np.isclose(
                          costs[indices], costs[index], rtol=1e-12, atol=1e-10))))
        return record

    pair_ids = [i for i, name in enumerate(names) if name == 'size_2']
    triple_ids = [i for i, name in enumerate(names) if name == 'size_3']
    best_det = best(range(n_det))
    best_max2 = best([*range(n_det), *pair_ids])
    best_max3 = best(range(len(selectors)))
    winner = describe(selected, selected_cost, before, single_costs, moves)
    winner.update(source=h['selected_source'], accepted=h['accepted'],
                  recorded_backend_best_native_cut=h['backend_best_native_cut'])
    next_state = proposed if h['accepted'] else start.copy()
    np.testing.assert_allclose(native_cut(next_state, edges, weights), h['after_native_cut'],
                               rtol=1e-12, atol=1e-10)
    result = {'round': h['round'], 'seed': h['seed'], 'num_atoms': m,
              'round_start_sha256': hashlib.sha256(start.astype('<i8').tobytes()).hexdigest(),
              'before_native_cut': before, 'saved_after_native_cut': h['after_native_cut'],
              'candidate_rows': len(selectors),
              'distinct_selectors': len({tuple(row) for row in selectors}),
              'size_2_count': len(pair_ids), 'size_3_count': len(triple_ids),
              'all_evaluated_selectors_feasible': True,
              'deterministic_scores_match_saved': True,
              'direct_native_winners_verified': True,
              'saved_winner': winner, 'best_deterministic': best_det,
              'best_size_2': best(pair_ids), 'best_size_3': best(triple_ids),
              'best_deterministic_plus_pairs': best_max2,
              'best_deterministic_plus_pairs_triples': best_max3,
              'sparse_improvement_over_saved_winner': selected_cost - best_max3['native_cut'],
              'sparse_improvement_over_deterministic': best_det['native_cut'] - best_max3['native_cut'],
              'synergy_definition': 'before + sum(single_atom_cost - before) - joint_cost; '
                  'positive means joint improvement beyond the singleton-additive prediction; '
                  'for three atoms this includes pair interactions and is not a pure cubic coefficient.'}
    return result, next_state


def aggregate(rows):
    def wtl(values):
        return {'wins': sum(v > 1e-10 for v in values),
                'ties': sum(abs(v) <= 1e-10 for v in values),
                'losses': sum(v < -1e-10 for v in values)}
    return {'rounds': len(rows),
            'sparse_vs_saved_winner': wtl([r['sparse_improvement_over_saved_winner'] for r in rows]),
            'sparse_vs_deterministic': wtl([r['sparse_improvement_over_deterministic'] for r in rows]),
            'sum_per_round_improvement_over_saved_winner': sum(r['sparse_improvement_over_saved_winner'] for r in rows),
            'sum_per_round_improvement_over_deterministic': sum(r['sparse_improvement_over_deterministic'] for r in rows),
            'best_sparse_cardinalities': dict(Counter(
                r['best_deterministic_plus_pairs_triples']['atom_count'] for r in rows)),
            'warning': 'Per-round sums are descriptive over frozen saved starts, not attainable trajectory gains.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=ROOT / 'benchmarks/hypergraph/results/fem-ier-20260924')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.results.resolve()
    output = args.output or base / 'sparse_probe.json'
    if output.exists():
        parser.error('output already exists; choose a new path')
    started = time.perf_counter()
    summary_path = base / 'summary.json'
    summary = json.loads(summary_path.read_text())
    inputs = {str(summary_path): digest(summary_path)}

    def check_input(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise AssertionError(f'input hash changed: {path}')
        inputs[str(path)] = value
        return path

    # Preserve the historical benchmark's hashed files, without importing them.
    source_hashes = {str(ROOT / p): digest(ROOT / p) for p in summary['source_sha256_after']}
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': digest(__file__), 'script': str(Path(__file__).resolve()),
              'python': sys.version, 'numpy': np.__version__,
              'scope': 'Offline diagnostic on every saved weighted/fixed FEM and random round; '
                  'no new FEM run, no candidate generation, no sparse trajectory, no integration replay.',
              'timing': 'Not part of or comparable to the original timed solver arms.',
              'configuration': {'enumerated_atom_cardinalities': [2, 3],
                                'shared_selectors': ['all_off', 'all_on', 'each_single_atom'],
                                'scoring': 'Independent native km1 from per-block pin counts; no solver imports'},
              'cases': []}
    all_rows = []
    datasets = {}
    for section in ('weighted', 'fixed_starts'):
        for item in summary[section]:
            detail_path = check_input(base / item['detail_file'])
            case = json.loads(detail_path.read_text())
            name = case.get('name', f"{case.get('instance')}-seed{case.get('seed')}")
            if section == 'weighted':
                path = next(Path(p) for p in case['input_hashes'] if p.endswith('-input.json'))
                data = json.loads(check_input(path, case['input_hashes'][str(path)]).read_text())
                edges = [sorted(set(edge)) for edge in data['edges']]
                nodes, weights = np.array(data['node_weights']), np.array(data['hyperedge_weights'])
                q, epsilon = case['q'], case['epsilon']
            else:
                dataset_name = case['instance']
                if dataset_name not in datasets:
                    path = next(Path(p) for p in summary['input_sha256_after']
                                if Path(p).name == dataset_name + '.hgr')
                    check_input(path, summary['input_sha256_after'][str(path)])
                    datasets[dataset_name] = read_hgr(path)
                edges, nodes, weights = datasets[dataset_name]
                q, epsilon = summary['config']['q_ibm'], summary['config']['epsilon_ibm']
            archive_path = check_input(detail_path.parent / case['initial_assignment']['artifact'])
            result = {'name': name, 'family': section, 'detail_file': item['detail_file'], 'arms': {}}
            with np.load(archive_path) as archive:
                for arm, run in case['arms'].items():
                    current = archive[run['initial_assignment']['key']].copy()
                    histories = [h for stage in run['last_result']['stages'] for h in stage['history']]
                    rows = []
                    for history in histories:
                        row, current = probe_round(current, history, edges, weights, nodes, q, epsilon)
                        row.update(case=name, family=section, arm=arm)
                        rows.append(row)
                    np.testing.assert_array_equal(current, archive[run['final_assignment']['key']])
                    result['arms'][arm] = {'rounds': rows, 'replayed_final_assignment_matches_saved': True,
                                          'aggregate': aggregate(rows)}
                    all_rows.extend(rows)
            report['cases'].append(result)
            print(f'{name}: replayed {sum(len(a["rounds"]) for a in result["arms"].values())} saved rounds', flush=True)
    report['aggregate'] = aggregate(all_rows)
    report['aggregate_by_family_arm'] = {
        f'{family}/{arm}': aggregate([r for r in all_rows if r['family'] == family and r['arm'] == arm])
        for family in ('weighted', 'fixed_starts') for arm in ('fem_ier', 'random')}
    report['input_sha256_before'] = inputs
    report['input_sha256_after'] = {p: digest(p) for p in inputs}
    report['inputs_unchanged'] = report['input_sha256_after'] == inputs
    report['historical_source_sha256_before'] = source_hashes
    report['historical_sources_unchanged'] = source_hashes == {p: digest(p) for p in source_hashes}
    report['script_unchanged'] = report['script_sha256'] == digest(__file__)
    if not all(report[k] for k in ('inputs_unchanged', 'historical_sources_unchanged', 'script_unchanged')):
        raise AssertionError('source or input changed while probing')
    report['diagnostic_wall_seconds_not_solver_timing'] = time.perf_counter() - started
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['aggregate_by_family_arm'], indent=2))
    print(f'Wrote {output}; script SHA256 {report["script_sha256"]}', flush=True)


if __name__ == '__main__':
    main()
