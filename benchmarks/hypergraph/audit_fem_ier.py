"""Independently replay saved FEM-IER evidence, without importing solver code.

Native costs are evaluated by physically applying saved disjoint moves and
counting occupied labels on affected edges. Tiny weighted instances and their
selector spaces are exhaustively enumerated; IBM selector spaces are not.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, *, atol=1e-9):
    require(np.asarray(actual).shape == np.asarray(expected).shape and
            np.allclose(actual, expected, rtol=1e-12, atol=atol),
            f'{label}: actual={actual!r}, recorded={expected!r}')


def read_case(path):
    data = json.loads(path.read_text())
    with np.load(path.parent / 'assignments.npz', allow_pickle=False) as archive:
        references = []

        def resolve(value):
            if isinstance(value, dict):
                if set(value) == {'artifact', 'key'}:
                    require(value['artifact'] == 'assignments.npz', 'unknown array artifact')
                    references.append(value['key'])
                    return archive[value['key']].copy()
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return value

        data = resolve(data)
        require(set(references) == set(archive.files), 'unreferenced/missing saved arrays')
    return data, references


class NativeGraph:
    def __init__(self, edges, nodes, weights, q, epsilon):
        self.edges = [np.asarray(sorted(set(edge)), dtype=np.int64) for edge in edges]
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.n, self.q = len(nodes), q
        require(len(self.edges) == len(self.weights), 'edge weight length')
        require(np.all(np.isfinite(self.nodes)) and np.all(self.nodes > 0), 'node weights')
        require(np.all(np.isfinite(self.weights)) and np.all(self.weights >= 0), 'edge weights')
        require(all(np.all((e >= 0) & (e < self.n)) for e in self.edges), 'pin bounds')
        self.total = float(self.nodes.sum())
        self.cap = (1 + epsilon) * self.total / q
        self.tol = 1e-10 * max(self.total, self.cap)

    def validate_labels(self, assignment):
        x = np.asarray(assignment)
        require(x.shape == (self.n,), 'assignment shape')
        require(np.issubdtype(x.dtype, np.integer), 'assignment integer dtype')
        require(np.all((x >= 0) & (x < self.q)), 'assignment label bounds')

    def edge_cost(self, x, edge):
        return max(0, len(set(x[edge].tolist())) - 1)

    def costs(self, assignments):
        """Direct occupied-label counting, independent of production formula."""
        result = np.zeros(len(assignments), dtype=np.float64)
        for edge, weight in zip(self.edges, self.weights):
            if len(edge) > 1 and weight:
                labels = assignments[:, edge]
                occupied = sum(np.any(labels == b, axis=1) for b in range(self.q))
                result += weight * (occupied - 1)
        return result

    def measure(self, x):
        self.validate_labels(x)
        loads = np.bincount(x, weights=self.nodes, minlength=self.q)
        cut = sum(weight * self.edge_cost(x, edge)
                  for edge, weight in zip(self.edges, self.weights))
        return dict(native_cut=float(cut), block_loads=loads.tolist(),
                    capacity=self.cap, capacity_tolerance=self.tol,
                    feasible=bool(np.all(loads <= self.cap + self.tol)))

    def check_measure(self, x, recorded):
        actual = self.measure(x)
        for key in ('native_cut', 'block_loads', 'capacity'):
            close(actual[key], recorded[key], key)
        close(actual['capacity_tolerance'], recorded['capacity_tolerance'],
              'capacity_tolerance', atol=1e-18)
        require(actual['feasible'] == recorded['feasible'], 'feasibility mismatch')
        require(actual['feasible'], 'saved assignment is infeasible')
        return actual

    def exact(self):
        optimum, feasible_count = np.inf, 0
        count = self.q ** self.n
        powers = self.q ** np.arange(self.n, dtype=np.int64)
        for start in range(0, count, 32768):
            ids = np.arange(start, min(start + 32768, count), dtype=np.int64)
            states = (ids[:, None] // powers) % self.q
            loads = np.stack([(states == b) @ self.nodes for b in range(self.q)], axis=1)
            valid = states[np.all(loads <= self.cap + self.tol, axis=1)]
            feasible_count += len(valid)
            if len(valid):
                optimum = min(optimum, float(self.costs(valid).min()))
        return dict(optimum=optimum, feasible_labeled_states=feasible_count,
                    all_labeled_states=count)


def read_hgr(path, q, epsilon):
    rows = [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('%')]
    header = list(map(int, rows[0].split()))
    require(len(header) == 2 or header[2] == 0, 'IBM input must be unweighted')
    m, n = header[:2]
    require(len(rows) == m + 1, 'hgr edge count')
    edges = [[int(pin) - 1 for pin in row.split()] for row in rows[1:]]
    return NativeGraph(edges, np.ones(n), np.ones(m), q, epsilon)


def move_setup(graph, x, saved_moves):
    moves = [{int(v): int(b) for v, b in move.items()} for move in saved_moves]
    touched = {}
    deltas = np.zeros((len(moves), graph.q))
    for j, move in enumerate(moves):
        require(bool(move), 'empty move')
        for v, b in move.items():
            require(0 <= v < graph.n and 0 <= b < graph.q, 'move bounds')
            require(v not in touched, 'overlapping move vertices')
            require(x[v] != b, 'no-op move entry')
            touched[v] = j
            deltas[j, x[v]] -= graph.nodes[v]
            deltas[j, b] += graph.nodes[v]
    require(np.all(np.abs(deltas) <= graph.tol), 'unbalanced move')
    base_loads = np.bincount(x, weights=graph.nodes, minlength=graph.q)
    require(np.all(base_loads + np.maximum(deltas, 0).sum(axis=0) <= graph.cap + graph.tol),
            'some selector subset can violate capacity')
    affected = [i for i, edge in enumerate(graph.edges) if any(v in touched for v in edge)]
    return moves, touched, deltas, affected


def apply_moves(x, moves, mask):
    result = x.copy()
    for j, move in enumerate(moves):
        if (mask >> j) & 1:
            for v, b in move.items():
                result[v] = b
    return result


def score_masks(graph, x, moves, affected, masks):
    """Physically apply selectors, then recompute affected native edge costs."""
    m = len(moves)
    masks = [int(mask) for mask in masks]
    require(all(0 <= mask < (1 << m) for mask in masks), 'selector mask out of bounds')
    unique = sorted(set(masks))
    base = graph.measure(x)['native_cut']
    affected_base = sum(graph.weights[i] * graph.edge_cost(x, graph.edges[i]) for i in affected)
    scores = {}
    for start in range(0, len(unique), 128):
        batch = unique[start:start + 128]
        states = np.repeat(x[None, :], len(batch), axis=0)
        for j, move in enumerate(moves):
            rows = np.array([(mask >> j) & 1 for mask in batch], dtype=bool)
            for v, b in move.items():
                states[rows, v] = b
        values = np.full(len(batch), base - affected_base, dtype=np.float64)
        for i in affected:
            edge, weight = graph.edges[i], graph.weights[i]
            if len(edge) > 1 and weight:
                labels = states[:, edge]
                occupied = sum(np.any(labels == b, axis=1) for b in range(graph.q))
                values += weight * (occupied - 1)
        scores.update(zip(batch, values.tolist()))
    return np.asarray([scores[mask] for mask in masks])


def interaction_summary(graph, touched, num_moves):
    histogram, groups, positive = Counter({'0': 0}), Counter({'0': 0, '1': 0, '2': 0, '3+': 0}), Counter({'0': 0, '1': 0, '2': 0, '3+': 0})
    for edge, weight in zip(graph.edges, graph.weights):
        if len(edge) < 2:
            continue
        count = len({touched[v] for v in edge if v in touched})
        histogram[str(count)] += 1
        key = str(count) if count < 3 else '3+'
        groups[key] += 1
        positive[key] += int(weight > 0)
    nontrivial = sum(histogram.values())
    return dict(num_moves=num_moves, total_hyperedges=len(graph.edges),
                nontrivial_hyperedges=nontrivial,
                excluded_empty_or_singleton_hyperedges=len(graph.edges) - nontrivial,
                edges_touching_atoms=dict(groups), positive_weight_edges_touching_atoms=dict(positive),
                exact_touched_atom_histogram=dict(histogram),
                max_atoms_touching_one_hyperedge=max(map(int, histogram), default=0))


def audit_arm(graph, arm, counts, exact_optimum=None):
    require(arm['status'] == 'success', 'arm was not successful')
    x = arm['initial_assignment'].copy()
    graph.check_measure(x, arm['initial'])
    graph.check_measure(arm['final_assignment'], arm['final'])
    counts['saved_original_assignments_measured'] += 2
    close(arm['finished_at'] - arm['started_at'], arm['pipeline_seconds'], 'pipeline time')
    diag = arm['last_result']['stages'][0]
    close(diag['initial_native_cut'], arm['initial']['native_cut'], 'diagnostic initial cut')
    close(diag['initial_loads'], arm['initial']['block_loads'], 'diagnostic initial loads')
    close(diag['capacity'], graph.cap, 'diagnostic capacity')
    require(diag['backend'] == ('fem' if arm['arm'] == 'fem_ier' else 'random'), 'backend identity')
    round_reports, accepted_count = [], 0
    for index, record in enumerate(diag['history']):
        counts['rounds_replayed'] += 1
        require(record['round'] == index, 'round index')
        before = graph.measure(x)
        close(before['native_cut'], record['before_native_cut'], 'round initial cut')
        moves, touched, deltas, affected = move_setup(graph, x, record['moves'])
        m = len(moves)
        require(m == record['num_moves'], 'move count')
        if not m:
            require(not record['accepted'], 'accepted empty move pool')
            close(record['after_native_cut'], before['native_cut'], 'empty pool cut')
            round_reports.append(dict(round=index, num_moves=0, pool_optimum=before['native_cut']))
            continue
        for key, value in interaction_summary(graph, touched, m).items():
            require(value == record['interaction_summary'][key], f'interaction summary {key}')
        counts['interaction_summaries_verified'] += 1
        det_masks = [0, (1 << m) - 1] + [1 << j for j in range(m)]
        backend_masks = record['backend_selector_masks']
        expected_count = diag['config']['num_trials'] if diag['backend'] == 'fem' else diag['config']['random_samples']
        require(len(backend_masks) == expected_count, 'terminal backend selector count')
        masks = det_masks + backend_masks
        scores = score_masks(graph, x, moves, affected, masks)
        det_scores, backend_scores = scores[:m + 2], scores[m + 2:]
        close(det_scores, record['deterministic_native_cuts'], 'deterministic scores')
        close(backend_scores, record['backend_native_cuts'], 'backend scores')
        counts['deterministic_selector_scores_verified'] += len(det_masks)
        counts['backend_selector_scores_verified'] += len(backend_masks)
        counts['unique_selector_scores_recomputed_per_round_sum'] += len(set(masks))
        flags = np.asarray([[(mask >> j) & 1 for j in range(m)] for mask in masks])
        loads = np.asarray(before['block_loads']) + flags @ deltas
        require(np.all(loads <= graph.cap + graph.tol), 'evaluated selector infeasible')
        counts['selector_feasibility_checks'] += len(masks)
        winner = int(np.argmin(scores))
        labels = ['all_off', 'all_on'] + [f'single_{j}' for j in range(m)]
        labels += [f"{diag['backend']}_{j}" for j in range(len(backend_masks))]
        selector = [(masks[winner] >> j) & 1 for j in range(m)]
        require(record['selected_source'] == labels[winner], 'first-argmin selected source')
        require(record['selected_selector'] == selector, 'first-argmin selected selector')
        require(record['selected_move_count'] == sum(selector), 'selected atom count')
        close(record['proposed_native_cut'], scores[winner], 'proposal cut')
        det_best, backend_best = float(det_scores.min()), float(backend_scores.min())
        close(record['deterministic_best_native_cut'], det_best, 'deterministic best')
        close(record['backend_best_native_cut'], backend_best, 'backend best')
        tol = 1e-12 * max(abs(before['native_cut']), abs(det_best), abs(backend_best), np.finfo(float).tiny)
        close(record['attribution_tolerance'], tol, 'attribution tolerance', atol=1e-18)
        require(record['backend_beats_deterministic'] == (backend_best < det_best - tol), 'backend attribution')
        counts['backend_beats_deterministic_rounds'] += int(record['backend_beats_deterministic'])
        proposed = apply_moves(x, moves, masks[winner])
        full_proposal = graph.measure(proposed)
        close(full_proposal['native_cut'], scores[winner], 'full native winner cross-check')
        require(full_proposal['feasible'], 'full winner feasibility')
        accepted = full_proposal['native_cut'] < before['native_cut']
        require(record['accepted'] == accepted, 'strict native acceptance')
        report = dict(round=index, num_moves=m, before_native_cut=before['native_cut'],
                      selected_native_cut=float(scores[winner]), selected_source=labels[winner],
                      deterministic_best=det_best, backend_best=backend_best,
                      backend_unique_selectors=len(set(backend_masks)), accepted=accepted)
        if exact_optimum is not None:
            require(m <= 12, 'weighted selector oracle intentionally limited to <=12 atoms')
            exhaustive = score_masks(graph, x, moves, affected, range(1 << m))
            pool_best = float(exhaustive.min())
            report.update(selector_space_size=1 << m, pool_optimum=pool_best,
                          pool_optimal_masks=np.flatnonzero(exhaustive == pool_best).tolist(),
                          selected_minus_pool_optimum=float(scores[winner] - pool_best),
                          pool_minus_global_optimum=pool_best - exact_optimum)
            counts['weighted_selector_spaces_exhausted'] += 1
            counts['weighted_exhaustive_selector_states'] += 1 << m
        if diag['backend'] == 'fem':
            p = np.asarray(record['fem_move_probabilities'])
            require(p.shape == (expected_count, m) and np.all(np.isfinite(p)) and
                    np.all((p >= 0) & (p <= 1)), 'FEM probability validity')
            decoded = [sum(int(value > .5) << j for j, value in enumerate(row)) for row in p]
            require(decoded == backend_masks, 'FEM terminal probability hard decisions')
            counts['fem_probability_rows_verified'] += len(p)
        if accepted:
            x = proposed
            accepted_count += 1
        after = graph.measure(x)
        close(record['after_native_cut'], after['native_cut'], 'round after cut')
        close(record['block_loads'], after['block_loads'], 'round after loads')
        round_reports.append(report)
    require(np.array_equal(x, arm['final_assignment']), 'replayed final assignment differs')
    close(diag['final_native_cut'], arm['final']['native_cut'], 'diagnostic final cut')
    close(diag['final_loads'], arm['final']['block_loads'], 'diagnostic final loads')
    close(arm['last_result']['final_native_cut'], arm['final']['native_cut'], 'outer final cut')
    require(diag['accepted_rounds'] == accepted_count, 'accepted round count')
    require(len(round_reports) == diag['config']['rounds'], 'round count')
    counts['accepted_rounds'] += accepted_count
    counts['arms_replayed'] += 1
    if exact_optimum is not None:
        close(arm['exact_gap'], arm['final']['native_cut'] - exact_optimum, 'exact gap')
    return dict(initial_native_cut=arm['initial']['native_cut'], final_native_cut=arm['final']['native_cut'], rounds=round_reports)


def audit_flow(graph, data, counts):
    flow = data['additional_flow']
    close(flow['budget_seconds'], data['arms']['fem_ier']['pipeline_seconds'], 'flow time budget')
    current, eligible, starts = data['initial_assignment'], [], []
    for index, record in enumerate(flow['records']):
        require(record['attempt'] == index, 'flow attempt index')
        require(record['run_id'] == f'flow_{index:04d}', 'flow run id')
        require(record['solver_seed'] == data['seed'] + index, 'flow seed')
        require(np.array_equal(record['initial_assignment'], current), 'flow continuation chain')
        graph.check_measure(record['initial_assignment'], record['initial'])
        counts['saved_original_assignments_measured'] += 1
        require(record['status'] == 'success', 'flow call failed')
        graph.check_measure(record['final_assignment'], record['final'])
        counts['saved_original_assignments_measured'] += 1
        require(record['final']['native_cut'] <= record['initial']['native_cut'], 'flow native cut increased')
        current = record['final_assignment']
        close(record['finished_at'] - record['started_at'], record['pipeline_seconds'], 'flow pipeline duration')
        starts.append(record['finished_at'] - record['completion_seconds'])
        inside = record['completion_seconds'] <= flow['budget_seconds']
        require(record['completed_within_budget'] == inside, 'flow deadline flag')
        if inside:
            eligible.append(record)
        else:
            require(index == len(flow['records']) - 1, 'flow continued after overshoot')
        counts['additional_flow_calls_verified'] += 1
    if starts:
        close(starts, np.repeat(starts[0], len(starts)), 'flow loop start timestamps')
        require(all(r['started_at'] - starts[0] < flow['budget_seconds'] for r in flow['records']),
                'flow call started after budget')
        require(flow['wall_seconds_including_overshoot'] >= flow['records'][-1]['completion_seconds'],
                'flow wall time below last completion')
    require(flow['attempts'] == len(flow['records']), 'flow attempt count')
    require(flow['eligible_completed_runs'] == len(eligible), 'eligible flow count')
    close(flow['overshoot_seconds'], max([r['completion_seconds'] - flow['budget_seconds'] for r in flow['records']] + [0]), 'flow overshoot')
    if not eligible:
        require(flow['status'] == 'no_result' and flow['winner_run_id'] is None and flow['winner'] is None,
                'flow no-result record')
    else:
        winner = min(eligible, key=lambda r: (r['final']['native_cut'], r['completion_seconds']))
        require(flow['status'] == 'success' and flow['winner_run_id'] == winner['run_id'], 'flow winner')
        for key in ('native_cut', 'block_loads'):
            close(flow['winner'][key], winner['final'][key], f'flow winner {key}')
        for key in ('solver_seed', 'completion_seconds', 'pipeline_seconds'):
            close(flow['winner'][key], winner[key], f'flow winner {key}')
    counts['additional_flow_eligible_calls'] += len(eligible)
    return dict(calls=len(flow['records']), eligible_calls=len(eligible), winner_run_id=flow['winner_run_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    summary = json.loads((args.results / 'summary.json').read_text())
    counts, errors, reports, checked_hashes = Counter(), [], {}, {}
    result = dict(audit_utc=datetime.now(timezone.utc).isoformat(),
                  audit_version=1, python_version=sys.version, numpy_version=np.__version__,
                  script_sha256=digest(__file__), results=str(args.results.resolve()),
                  implementation='NumPy/standard-library only; no production or benchmark imports.',
                  coverage=counts, cases=reports, errors=errors,
                  limitations=[
                      'Integration intermediate full assignments and per-level graphs were not saved: only original initial/final arrays are independently scored; intermediate records are checked for internal monotonicity and capacity consistency, not selector replay.',
                      'IBM 24-atom spaces are not exhaustively enumerated: all saved selectors are audited, without certifying selector-global optimality.',
                      'FEM/random are fixed-count controls (8 terminal FEM selectors versus 800 random selectors), not equal-time or equal-work arms.',
                      'No asymptotic OGP, MCMC mixing, or general algorithm-performance claim follows from this audit.'
                  ])

    def hashes(mapping, kind):
        for name, expected in mapping.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            path = path.resolve()
            key = str(path)
            require(digest(path) == expected, f'{kind} hash mismatch: {name}')
            if key in checked_hashes:
                require(checked_hashes[key] == expected, f'inconsistent claimed hash: {name}')
            else:
                checked_hashes[key] = expected
                counts[kind + '_unique_hashes_verified'] += 1

    try:
        for kind, flag in [('source', 'sources'), ('input', 'inputs')]:
            before, after = summary[kind + '_sha256_before'], summary[kind + '_sha256_after']
            require(before == after, f'{kind} before/after hashes differ')
            require(summary[flag + '_unchanged_during_run'], f'{kind} unchanged flag false')
            hashes(before, kind)
        manifest = json.loads((args.results / 'manifest.json').read_text())
        for key in ('config', 'source_sha256_before'):
            require(summary[key] == manifest[key], f'manifest/summary {key} mismatch')
    except Exception as exc:
        errors.append(dict(scope='provenance', error=str(exc), traceback=traceback.format_exc()))

    ibm_graphs = {}
    for category in ('weighted', 'fixed_starts', 'integration'):
        for row in summary[category]:
            key = row['detail_file']
            print(f'Auditing {key}', flush=True)
            try:
                data, refs = read_case(args.results / key)
                counts['artifact_arrays_resolved'] += len(refs)
                hashes(data['input_hashes'], 'input')
                if category == 'weighted':
                    spec_path = next(Path(p) for p in data['input_hashes'] if p.endswith('-input.json'))
                    spec = json.loads(spec_path.read_text())
                    graph = NativeGraph(spec['edges'], spec['node_weights'], spec['hyperedge_weights'], data['q'], data['epsilon'])
                    prior_path = next(Path(p) for p in data['input_hashes'] if p.endswith('-assignments.npz'))
                    with np.load(prior_path, allow_pickle=False) as old:
                        require(np.array_equal(old['fem_final_assignment'], data['initial_assignment']), 'weighted saved start mismatch')
                    oracle = graph.exact()
                    close(oracle['optimum'], data['exact_optimum'], 'weighted global optimum')
                    require(oracle['feasible_labeled_states'] == data['feasible_labeled_states'], 'weighted feasible count')
                    close(data['initial_exact_gap'], data['initial']['native_cut'] - oracle['optimum'], 'initial exact gap')
                    counts['weighted_original_spaces_exhausted'] += 1
                    counts['weighted_original_states_enumerated'] += oracle['all_labeled_states']
                    counts['weighted_original_feasible_states_scored'] += oracle['feasible_labeled_states']
                else:
                    name = data['instance']
                    if name not in ibm_graphs:
                        input_path = next(Path(p) for p in summary['input_sha256_after'] if p.endswith('/' + name + '.hgr'))
                        ibm_graphs[name] = read_hgr(input_path, summary['config']['q_ibm'], summary['config']['epsilon_ibm'])
                    graph = ibm_graphs[name]
                    old_path = next(Path(p) for p in data['input_hashes'] if p.endswith('/runs.json'))
                    old_data = json.loads(old_path.read_text())
                    old_refs = old_data['single_runs']['fem']['assignment_artifacts']
                    with np.load(old_path.parent / old_refs['final_assignment']['file'], allow_pickle=False) as old:
                        if category == 'fixed_starts':
                            require(np.array_equal(old[old_refs['final_assignment']['key']], data['initial_assignment']), 'IBM saved start mismatch')
                        else:
                            coarse = old[old_refs['coarse_assignment']['key']]
                            require(np.array_equal(coarse, data['coarse_assignment']), 'integration coarse saved start')
                            require(np.array_equal(coarse[old['original_to_coarse']], data['initial_assignment']), 'integration coarse projection')
                            counts['saved_coarse_assignments_verified_against_input'] += 1
                graph.check_measure(data['initial_assignment'], data['initial'])
                counts['saved_original_assignments_measured'] += 1
                for field in ('initial', 'exact_optimum'):
                    if field in row:
                        require(row[field] == data[field], f'summary {field} mismatch')
                case_report = reports[key] = dict(arms={})
                if category == 'weighted':
                    case_report['oracle'] = oracle
                if category != 'integration':
                    for name, arm in data['arms'].items():
                        require(np.array_equal(arm['initial_assignment'], data['initial_assignment']), 'arms did not share initial state')
                        case_report['arms'][name] = audit_arm(graph, arm, counts, data.get('exact_optimum'))
                        require(row['arms'][name]['final'] == arm['final'], 'summary arm final')
                    a, b = [data['arms'][name]['last_result']['stages'][0]['history'][0] for name in ('fem_ier', 'random')]
                    require(a['seed'] == b['seed'] and a['moves'] == b['moves'], 'first-round A/B pool or seed differs')
                    counts['first_round_ab_pools_identical'] += 1
                    if 'additional_flow' in data:
                        case_report['additional_flow'] = audit_flow(graph, data, counts)
                        for field, value in row['additional_flow'].items():
                            require(data['additional_flow'][field] == value, f'flow summary {field}')
                else:
                    for name, arm in data['arms'].items():
                        require(arm['status'] == 'success', 'integration arm failed')
                        graph.check_measure(arm['final_assignment'], arm['final'])
                        counts['saved_original_assignments_measured'] += 1
                        require(row['arms'][name]['final'] == arm['final'], 'integration summary final')
                        previous = data['initial']['native_cut']
                        for stage in arm['stages']:
                            close(stage['before']['native_cut'], previous, 'integration stage continuity')
                            require(stage['after']['native_cut'] <= stage['before']['native_cut'], 'integration reported monotonicity')
                            for measured in (stage['before'], stage['after']):
                                close(sum(measured['block_loads']), graph.total, 'integration reported total load')
                                close(measured['capacity'], graph.cap, 'integration reported capacity')
                                require(measured['feasible'] and max(measured['block_loads']) <= graph.cap + graph.tol,
                                        'integration reported feasibility')
                            previous = stage['after']['native_cut']
                            counts['integration_stage_records_consistency_checked_only'] += 1
                            if stage['last_result']:
                                for diagnostic in stage['last_result'].get('stages', []):
                                    if diagnostic.get('method') == 'fem_ier':
                                        counts['integration_ier_rounds_not_independently_replayed'] += len(diagnostic['history'])
                        close(previous, arm['final']['native_cut'], 'integration final stage')
                        case_report['arms'][name] = dict(final_native_cut=arm['final']['native_cut'],
                            stages_consistency_checked=len(arm['stages']))
                        counts['integration_final_arms_verified'] += 1
                counts[category + '_cases_passed'] += 1
                case_report['passed'] = True
            except Exception as exc:
                errors.append(dict(scope=key, error=str(exc), traceback=traceback.format_exc()))
                reports.setdefault(key, {})['passed'] = False
    result['passed'] = not errors
    result['elapsed_seconds'] = time.perf_counter() - started
    result['verified_file_hashes'] = checked_hashes
    output = args.results / 'audit.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(passed=result['passed'], coverage=counts, errors=errors, output=str(output)), indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
