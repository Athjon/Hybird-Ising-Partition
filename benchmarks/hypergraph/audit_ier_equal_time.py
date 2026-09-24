"""Independent audit of complete V-cycle and deadline-controlled IER runs.

This module imports only the earlier independent NumPy audit evaluator, never
production solvers or objectives. Run after all measured benchmark work ends.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hypergraph.audit_fem_ier import (
    NativeGraph, apply_moves, close, digest, interaction_summary, move_setup,
    read_hgr, require, score_masks,
)


def audit_rounds(graph, initial, diagnostic, counts, *, expected_seed=None):
    """Replay every saved selector by modifying actual vertex assignments."""
    require(diagnostic['method'] == 'fem_ier', 'unexpected IER method')
    backend = diagnostic['backend']
    require(backend in ('fem', 'random', 'deterministic', 'pairs', 'exact'), 'unknown backend')
    config = diagnostic['config']
    if expected_seed is not None:
        require(config['seed'] == expected_seed, 'IER stage seed')
    x = np.asarray(initial).copy()
    initial_measure = graph.measure(x)
    require(initial_measure['feasible'], 'IER stage starts infeasible')
    close(diagnostic['initial_native_cut'], initial_measure['native_cut'], 'IER initial cut')
    close(diagnostic['initial_loads'], initial_measure['block_loads'], 'IER initial loads')
    close(diagnostic['capacity'], graph.cap, 'IER capacity')
    accepted_count, round_reports = 0, []
    for index, record in enumerate(diagnostic['history']):
        before = graph.measure(x)
        require(record['round'] == index, 'IER round index')
        require(record['seed'] == config['seed'] + 104729 * index, 'IER round seed')
        close(record['before_native_cut'], before['native_cut'], 'IER round initial cut')
        require(record['candidate_generation_seconds'] >= 0 and record['backend_seconds'] >= 0,
                'negative round component duration')
        require(record['total_seconds'] >= record['candidate_generation_seconds'] + record['backend_seconds'],
                'round duration below component durations')
        moves, touched, deltas, affected = move_setup(graph, x, record['moves'])
        m = len(moves)
        require(record['num_moves'] == m, 'saved move count')
        require(m <= config['max_moves'], 'pool exceeds configured maximum')
        if not m:
            require(record['reason'] == 'no_balanced_disjoint_moves', 'empty pool reason')
            require(record['selected_source'] == 'all_off' and record['selected_selector'] == [], 'empty-pool winner')
            require(record['selected_move_count'] == 0 and not record['accepted'], 'empty-pool acceptance')
            require(record['backend_selector_masks'] == record['backend_native_cuts'] == [], 'empty backend arrays')
            require(record['backend_selector_count'] == record['backend_unique_selector_count'] == 0,
                    'empty backend selector counts')
            require(record['backend_best_native_cut'] is None and not record['backend_beats_deterministic'],
                    'empty backend attribution')
            for key in ('after_native_cut', 'proposed_native_cut', 'deterministic_best_native_cut'):
                close(record[key], before['native_cut'], key)
            close(record['deterministic_native_cuts'], [before['native_cut']], 'implicit all-off score')
            close(record['block_loads'], before['block_loads'], 'empty-pool loads')
            counts['empty_rounds_verified'] += 1
            counts['deterministic_selector_scores_verified'] += 1
            counts['selector_feasibility_checks'] += 1
        else:
            for key, value in interaction_summary(graph, touched, m).items():
                require(record['interaction_summary'][key] == value, f'interaction summary {key}')
            counts['interaction_summaries_verified'] += 1
            deterministic_masks = [0, (1 << m) - 1] + [1 << j for j in range(m)]
            backend_masks = record['backend_selector_masks']
            if backend == 'deterministic':
                expected_masks = []
            elif backend == 'pairs':
                expected_masks = [(1 << j) | (1 << k) for j in range(m) for k in range(j + 1, m)]
            elif backend == 'random':
                bits = np.random.default_rng(record['seed']).integers(0, 2, size=(config['random_samples'], m))
                expected_masks = [sum(int(bit) << j for j, bit in enumerate(row)) for row in bits]
            elif backend == 'exact':
                expected_masks = list(range(1 << m))
            else:
                p = np.asarray(record['fem_move_probabilities'])
                require(p.shape == (config['num_trials'], m), 'FEM probability shape')
                require(np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)), 'FEM probabilities')
                expected_masks = [sum(int(value > .5) << j for j, value in enumerate(row)) for row in p]
                counts['fem_terminal_probability_rows_verified'] += len(p)
            require(backend_masks == expected_masks, 'backend selector generation/terminal inference')
            require(record['backend_selector_count'] == len(backend_masks), 'backend selector count')
            require(record['backend_unique_selector_count'] == len(set(backend_masks)), 'backend unique count')
            masks = deterministic_masks + backend_masks
            scores = score_masks(graph, x, moves, affected, masks)
            cut = len(deterministic_masks)
            deterministic_scores, backend_scores = scores[:cut], scores[cut:]
            close(record['deterministic_native_cuts'], deterministic_scores, 'shared candidate scores')
            close(record['backend_native_cuts'], backend_scores, 'backend native scores')
            counts['deterministic_selector_scores_verified'] += len(deterministic_masks)
            counts['backend_selector_scores_verified'] += len(backend_masks)
            counts['unique_selector_scores_recomputed_per_round_sum'] += len(set(masks))
            bits = np.asarray([[(mask >> j) & 1 for j in range(m)] for mask in masks])
            candidate_loads = np.asarray(before['block_loads']) + bits @ deltas
            require(np.all(candidate_loads <= graph.cap + graph.tol), 'selector capacity')
            counts['selector_feasibility_checks'] += len(masks)
            winner = int(np.argmin(scores))
            names = ['all_off', 'all_on'] + [f'single_{j}' for j in range(m)]
            names += [f'{backend}_{j}' for j in range(len(backend_masks))]
            selector = [(masks[winner] >> j) & 1 for j in range(m)]
            require(record['selected_source'] == names[winner], 'first-argmin winner attribution')
            require(record['selected_selector'] == selector, 'first-argmin winner bits')
            require(record['selected_move_count'] == sum(selector), 'winner atom count')
            det_best = float(deterministic_scores.min())
            backend_best = float(backend_scores.min()) if len(backend_scores) else None
            close(record['deterministic_best_native_cut'], det_best, 'shared best score')
            if backend_best is None:
                require(record['backend_best_native_cut'] is None, 'empty backend best must be null')
            else:
                close(record['backend_best_native_cut'], backend_best, 'backend best score')
            tolerance = 1e-12 * max(abs(before['native_cut']), abs(det_best),
                                   abs(backend_best) if backend_best is not None else 0., np.finfo(float).tiny)
            close(record['attribution_tolerance'], tolerance, 'attribution tolerance', atol=1e-18)
            backend_improves = backend_best is not None and backend_best < det_best - tolerance
            require(record['backend_beats_deterministic'] == backend_improves, 'backend attribution flag')
            counts['backend_beats_deterministic_rounds'] += int(backend_improves)
            proposed = apply_moves(x, moves, masks[winner])
            proposed_measure = graph.measure(proposed)
            require(proposed_measure['feasible'], 'full winning state infeasible')
            close(proposed_measure['native_cut'], scores[winner], 'full winning native score')
            close(record['proposed_native_cut'], proposed_measure['native_cut'], 'saved proposed cut')
            accepted = proposed_measure['native_cut'] < before['native_cut']
            require(record['accepted'] == accepted, 'strict native acceptance')
            if accepted:
                x = proposed
                accepted_count += 1
            after = graph.measure(x)
            close(record['after_native_cut'], after['native_cut'], 'round accepted state cut')
            close(record['block_loads'], after['block_loads'], 'round accepted state loads')
        round_reports.append(dict(round=index, num_moves=m, before_native_cut=before['native_cut'],
                                  after_native_cut=record['after_native_cut'],
                                  backend_selector_count=record['backend_selector_count'],
                                  backend_unique_selector_count=record['backend_unique_selector_count'],
                                  backend_beats_deterministic=record['backend_beats_deterministic']))
        counts['ier_rounds_replayed'] += 1
    require(len(diagnostic['history']) == config['rounds'], 'IER configured round count')
    require(diagnostic['accepted_rounds'] == accepted_count, 'IER accepted count')
    final_measure = graph.measure(x)
    close(diagnostic['final_native_cut'], final_measure['native_cut'], 'IER final cut')
    close(diagnostic['final_loads'], final_measure['block_loads'], 'IER final loads')
    require(diagnostic['seconds'] >= sum(r['total_seconds'] for r in diagnostic['history']), 'IER total duration')
    counts['ier_stages_replayed'] += 1
    counts['accepted_rounds'] += accepted_count
    return x, round_reports


def assert_same_graph(a, b, label):
    require(a.n == b.n and a.q == b.q, f'{label}: dimensions')
    close(a.nodes, b.nodes, f'{label}: node weights')
    close(a.weights, b.weights, f'{label}: edge weights')
    require(len(a.edges) == len(b.edges) and all(np.array_equal(x, y) for x, y in zip(a.edges, b.edges)),
            f'{label}: hyperedges/order')


def assert_contraction(fine, coarse, remap):
    """Check the quotient independently, preserving parallel edge order."""
    remap = np.asarray(remap)
    require(remap.shape == (fine.n,) and np.issubdtype(remap.dtype, np.integer), 'hierarchy remap shape/type')
    require(np.all((remap >= 0) & (remap < coarse.n)), 'hierarchy remap bounds')
    require(len(np.unique(remap)) == coarse.n, 'hierarchy map has empty coarse vertices')
    close(np.bincount(remap, weights=fine.nodes, minlength=coarse.n), coarse.nodes, 'quotient node weights')
    mapped = [(np.asarray(sorted(set(remap[edge].tolist()))), weight)
              for edge, weight in zip(fine.edges, fine.weights)]
    mapped = [(edge, weight) for edge, weight in mapped if len(edge) > 1]
    require(len(mapped) == len(coarse.edges), 'quotient edge count')
    require(all(np.array_equal(edge, saved) for (edge, _), saved in zip(mapped, coarse.edges)), 'quotient edge order/pins')
    close([weight for _, weight in mapped], coarse.weights, 'quotient edge weights')


def array_digest(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({'dtype': array.dtype.str, 'shape': array.shape}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + array.tobytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def load_bundle(directory, artifact, counts, file_hashes, assignment_hashes):
    for kind in ('json', 'npz'):
        path = directory / artifact[kind]
        actual = digest(path)
        require(actual == artifact[kind + '_sha256'], f'artifact hash mismatch: {path}')
        file_hashes[str(path.resolve())] = actual
        counts['artifact_file_hashes_verified'] += 1
    raw = json.loads((directory / artifact['json']).read_text())
    with np.load(directory / artifact['npz'], allow_pickle=False) as archive:
        refs = []

        def resolve(value):
            if isinstance(value, dict):
                if set(value) == {'artifact', 'key'}:
                    require(value['artifact'] == artifact['npz'], 'array reference points outside its bundle')
                    key = value['key']
                    array = archive[key].copy()
                    refs.append(key)
                    if key.split('__')[-1] in ('assignment', 'initial_assignment', 'final_assignment', 'coarse_assignment'):
                        assignment_hashes[str((directory / artifact['npz']).resolve()) + '::' + key] = array_digest(array)
                        counts['assignment_arrays_hashed'] += 1
                    return array
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return value

        data = resolve(raw)
        require(len(refs) == len(set(refs)) and set(refs) == set(archive.files),
                'unreferenced/duplicate/missing array artifacts')
        counts['artifact_arrays_resolved'] += len(refs)
    return data


def unpack_ragged(packed):
    offsets, values = np.asarray(packed['offsets']), np.asarray(packed['values'])
    require(offsets.ndim == values.ndim == 1 and len(offsets) >= 1, 'ragged dimensions')
    require(np.issubdtype(offsets.dtype, np.integer) and np.issubdtype(values.dtype, np.integer), 'ragged integer type')
    require(offsets[0] == 0 and offsets[-1] == len(values) and np.all(np.diff(offsets) >= 0), 'ragged offsets')
    return [values[start:end].tolist() for start, end in zip(offsets[:-1], offsets[1:])]


def audit_hierarchy(bundle, original, q, epsilon, counts):
    graphs = {key: NativeGraph(unpack_ragged(value['hyperedges']), value['node_weights'],
                              value['hyperedge_weights'], q, epsilon)
              for key, value in bundle['graphs'].items()}
    assert_same_graph(graphs['original'], original, 'original graph versus input hgr')
    coarse = NativeGraph(unpack_ragged(bundle['coarse_hyperedges']), bundle['coarse_node_weights'],
                         bundle['coarse_hyperedge_weights'], q, epsilon)
    levels = bundle['hierarchy_stack']
    require(set(graphs) == {'original'} | {f'level_{i}' for i in range(len(levels))}, 'hierarchy graph inventory')
    composed = np.arange(original.n, dtype=np.int64)
    for index, level in enumerate(levels):
        require(level['graph_id'] == f'level_{index}', 'hierarchy level ordering')
        fine = graphs[level['graph_id']]
        require(level['num_nodes'] == fine.n, 'hierarchy node count')
        if index == 0:
            assert_same_graph(fine, original, 'first hierarchy level')
        groups = unpack_ragged(level['groups'])
        require(len(groups) == fine.n and all(groups), 'hierarchy groups dimensions')
        flattened = [v for group in groups for v in group]
        require(sorted(flattened) == list(range(original.n)), 'groups do not partition original vertices')
        from_groups = np.empty(original.n, dtype=np.int64)
        for group_index, group in enumerate(groups):
            from_groups[group] = group_index
        require(np.array_equal(composed, from_groups), 'groups disagree with composed remaps')
        close([original.nodes[group].sum() for group in groups], fine.nodes, 'group node weights')
        next_graph = graphs[f'level_{index + 1}'] if index + 1 < len(levels) else coarse
        assert_contraction(fine, next_graph, level['remap'])
        composed = level['remap'][composed]
        counts['hierarchy_contractions_verified'] += 1
    if not levels:
        assert_same_graph(coarse, original, 'empty hierarchy coarse graph')
    require(np.array_equal(composed, bundle['original_to_coarse']), 'composed original-to-coarse mapping')
    coarse_metrics = coarse.measure(bundle['coarse_assignment'])
    require(coarse_metrics['feasible'], 'given coarse state infeasible')
    lifted = bundle['coarse_assignment'][composed]
    lifted_metrics = original.measure(lifted)
    close(lifted_metrics['native_cut'], coarse_metrics['native_cut'], 'coarse lift native objective')
    close(lifted_metrics['block_loads'], coarse_metrics['block_loads'], 'coarse lift block loads')
    counts['given_coarse_states_measured'] += 1
    counts['hierarchies_verified'] += 1
    return graphs, coarse, lifted


def check_metrics(actual, recorded, label):
    for key in ('native_cut', 'block_loads', 'capacity'):
        close(actual[key], recorded[key], label + ': ' + key)
    require(actual['feasible'] == recorded['feasible'], label + ': feasibility')
    if 'capacity_tolerance' in recorded:
        close(actual['capacity_tolerance'], recorded['capacity_tolerance'], label + ': capacity tolerance', atol=1e-18)


def measure_saved(graph, state, counts, *recorded):
    actual = graph.measure(state)
    require(actual['feasible'], 'saved state exceeds weighted capacities')
    for index, metrics in enumerate(recorded):
        check_metrics(actual, metrics, f'saved metrics {index}')
    counts['saved_assignments_independently_measured'] += 1
    return actual


def audit_attempt(record, arm, hierarchy, graphs, lifted, counts, frontend, previous):
    require(record['status'] == 'success', 'attempt failed or failed benchmark validation')
    initial = measure_saved(graphs['original'], record['initial_assignment'], counts, record['initial'])
    final = measure_saved(graphs['original'], record['final_assignment'], counts, record['final'], record['native_metrics'])
    require(final['native_cut'] <= initial['native_cut'], 'complete attempt increases native cut')
    if record['kind'] == 'vcycle':
        require(record['predecessor_run_id'] is None, 'V-cycle restart has a predecessor')
        require(np.array_equal(record['coarse_assignment'], hierarchy['coarse_assignment']), 'V-cycle did not restart from given coarse state')
        require(np.array_equal(record['initial_assignment'], lifted), 'V-cycle initial is not independent coarse lift')
        current = hierarchy['coarse_assignment'].copy()
        levels = list(reversed(hierarchy['hierarchy_stack']))
        expected_graphs = [level['graph_id'] for level in levels] + ['original']
        counts['independent_vcycle_restarts_verified'] += 1
        counts['saved_coarse_assignments_compared_to_given'] += 1
    else:
        require(record['kind'] == 'flow_continuation' and arm == 'flow', 'invalid continuation arm/kind')
        require(previous is not None and previous['status'] == 'success', 'flow continuation lacks predecessor')
        require(record['predecessor_run_id'] == previous['run_id'], 'flow predecessor run id')
        require(np.array_equal(record['initial_assignment'], previous['final_assignment']), 'flow did not continue own prior state')
        current, levels, expected_graphs = record['initial_assignment'].copy(), [], ['original']
        counts['own_flow_continuations_verified'] += 1
    require([stage['graph_id'] for stage in record['stages']] == expected_graphs, 'incomplete/misordered V-cycle stages')
    stage_reports = []
    for index, stage in enumerate(record['stages']):
        require(stage['status'] == 'success', 'incomplete stage')
        projected = current[levels[index]['remap']] if index < len(levels) else current
        require(np.array_equal(stage['initial_assignment'], projected), 'stage initial differs from preceding layer projection')
        counts['stage_projections_verified'] += 1
        graph = graphs[stage['graph_id']]
        before = measure_saved(graph, stage['initial_assignment'], counts, stage['initial'])
        after = measure_saved(graph, stage['final_assignment'], counts, stage['final'])
        require(after['native_cut'] <= before['native_cut'], 'stage native objective increased')
        details = dict(graph_id=stage['graph_id'], before_native_cut=before['native_cut'], after_native_cut=after['native_cut'])
        if arm == 'flow':
            require(stage['last_result'] is None, 'unexpected flow diagnostic')
        else:
            last = stage['last_result']
            require(last['method'] == 'fem_ier_cycle' and last['mode_cycle'] == frontend['mode_cycle'], 'mode cycle diagnostic')
            require([part['method'] for part in last['stages']] == ['fem_ier', 'flow'], 'mixed stage order')
            ier = last['stages'][0]
            require(ier['backend'] == arm, 'stage backend does not match arm')
            for option, field in [('ier_rounds', 'rounds'), ('ier_max_moves', 'max_moves'),
                                  ('ier_boundary_pool', 'boundary_pool'), ('ier_pool_strategy', 'pool_strategy'),
                                  ('ier_num_trials', 'num_trials'), ('ier_num_steps', 'num_steps'),
                                  ('ier_random_samples', 'random_samples'), ('ier_allow_triples', 'allow_triples')]:
                require(ier['config'][field] == frontend[option], f'frontend option {option}')
            after_ier, rounds = audit_rounds(graph, stage['initial_assignment'], ier, counts,
                                           expected_seed=record['solver_seed'])
            require(after['native_cut'] <= graph.measure(after_ier)['native_cut'], 'flow following IER increased cut')
            close(last['stages'][1]['final_native_cut'], after['native_cut'], 'flow component final cut')
            close(last['final_native_cut'], after['native_cut'], 'mode cycle final cut')
            details['rounds'] = rounds
        current = stage['final_assignment']
        stage_reports.append(details)
        counts['refinement_stages_verified'] += 1
    require(np.array_equal(current, record['final_assignment']), 'final attempt state is not final stage state')
    expected_fixed = arm == 'flow' and record['kind'] == 'flow_continuation' and np.array_equal(record['initial_assignment'], record['final_assignment'])
    require(record['exact_assignment_fixed_point'] == expected_fixed, 'exact flow fixed-point claim')
    require(record['independent_stage_and_projection_verification'], 'benchmark stage verification flag')
    counts['complete_attempts_verified'] += 1
    return final, stage_reports


def audit_arm(result, arm, base_seed, budgets, hierarchy, graphs, lifted, counts, frontend, stride):
    require(result['arm'] == arm and result['base_seed'] == base_seed, 'arm identity/base seed')
    require(result['budgets_seconds'] == budgets, 'arm deadlines')
    require(result['status'] == 'execution_complete', 'arm execution incomplete')
    start, finish = result['started_at'], result['finished_at']
    require(np.isfinite(start) and np.isfinite(finish) and finish >= start, 'arm clock interval')
    close(result['wall_seconds_including_overshoot'], finish - start, 'arm wall duration')
    close(result['overshoot_seconds'], max(0., finish - start - budgets[-1]), 'arm overshoot duration')
    fallback = result['fallback']
    require(fallback['status'] == 'success' and fallback['source'] == 'initial_only', 'fallback identity/status')
    require(np.array_equal(fallback['assignment'], lifted), 'fallback is not independent given coarse lift')
    fallback_metrics = measure_saved(graphs['original'], fallback['assignment'], counts,
                                     fallback['native_metrics'], fallback['verified'])
    require(0 <= fallback['available_seconds'] <= finish - start, 'fallback availability')
    counts['independent_timed_fallbacks_verified'] += 1
    previous, final_metrics, attempt_reports = None, {}, []
    best_complete, best_available = float('inf'), fallback_metrics['native_cut']
    previous_finish = start + fallback['available_seconds']
    for index, record in enumerate(result['records']):
        require(record['arm'] == arm and record['attempt'] == index, 'attempt arm/index')
        require(record['run_id'] == f'{arm}_{index:04d}', 'attempt run id')
        require(record['solver_seed'] == base_seed + stride * index, 'restart seed')
        require(record['kind'] == ('flow_continuation' if arm == 'flow' and index else 'vcycle'), 'restart kind')
        require(record['started_at'] >= previous_finish, 'attempt overlaps predecessor or fallback')
        require(record['started_at'] - start < budgets[-1], 'attempt starts after maximum budget')
        require(record['finished_at'] >= record['started_at'] and record['finished_at'] <= finish, 'attempt clock interval')
        close(record['completion_seconds'], record['finished_at'] - start, 'completion relative to arm timer')
        close(record['pipeline_seconds'], record['finished_at'] - record['started_at'], 'whole attempt duration')
        within = record['completion_seconds'] <= budgets[-1]
        require(record['completed_within_max_budget'] == within, 'maximum-deadline eligibility flag')
        if not within:
            require(index == len(result['records']) - 1, 'attempt executed after overshooting return')
            counts['overshooting_attempts_excluded_from_max_deadline'] += 1
        metrics, stage_reports = audit_attempt(record, arm, hierarchy, graphs, lifted, counts, frontend, previous)
        require(record['improves_completed_incumbent'] == (metrics['native_cut'] < best_complete), 'completed incumbent update')
        require(record['improves_available_incumbent'] == (metrics['native_cut'] < best_available), 'available incumbent update')
        best_complete = min(best_complete, metrics['native_cut'])
        best_available = min(best_available, metrics['native_cut'])
        final_metrics[record['run_id']] = metrics
        attempt_reports.append(dict(run_id=record['run_id'], kind=record['kind'],
                                    completion_seconds=record['completion_seconds'],
                                    native_cut=metrics['native_cut'], stages=stage_reports))
        previous, previous_finish = record, record['finished_at']
    require(result['attempts'] == len(result['records']), 'attempt count')
    if previous is not None and previous['completion_seconds'] > budgets[-1]:
        require(result['stopped_reason'] == 'overshoot', 'overshoot stop reason')
    elif previous is not None and previous['exact_assignment_fixed_point']:
        require(result['stopped_reason'] == 'exact_assignment_fixed_point', 'flow fixed-point stopping')
        counts['flow_exact_fixed_point_stops_verified'] += 1
    else:
        require(result['stopped_reason'] in ('deadline', 'max_attempts'), 'unexpected stop reason')
    require(result['max_attempts_reached'] == (result['stopped_reason'] == 'max_attempts'), 'max-attempt flag')
    expected_checkpoints = []
    for budget in budgets:
        eligible = [record for record in result['records'] if record['completion_seconds'] <= budget]
        winner = min(eligible, key=lambda record: (final_metrics[record['run_id']]['native_cut'], record['completion_seconds'])) if eligible else None
        completed = None if winner is None else dict(source='completed_solver', run_id=winner['run_id'],
            native_cut=final_metrics[winner['run_id']]['native_cut'], block_loads=final_metrics[winner['run_id']]['block_loads'],
            available_seconds=winner['completion_seconds'])
        initial = None if fallback['available_seconds'] > budget else dict(source='initial_only', run_id=None,
            native_cut=fallback_metrics['native_cut'], block_loads=fallback_metrics['block_loads'],
            available_seconds=fallback['available_seconds'])
        choices = [value for value in (initial, completed) if value is not None]
        available = min(choices, key=lambda value: (value['native_cut'], value['available_seconds'])) if choices else None
        expected_checkpoints.append(dict(budget_seconds=budget, optimized_status='success' if winner is not None else 'no_result',
                                         eligible_completed_runs=len(eligible),
                                         eligible_completed_vcycles=sum(r['kind'] == 'vcycle' for r in eligible),
                                         eligible_flow_continuations=sum(r['kind'] == 'flow_continuation' for r in eligible),
                                         best_completed=completed, best_available=available))
        counts['deadline_checkpoints_verified'] += 1
        counts['late_returns_excluded_across_deadlines'] += len(result['records']) - len(eligible)
    require(jsonable(result['checkpoints']) == expected_checkpoints, 'checkpoint eligibility, tie-break, or best-available rule')
    require(result['validation_summary'] == dict(successful_runs=len(result['records']), execution_failed_runs=0,
                                                 validation_failed_runs=0, fallback_valid=True), 'validation summary')
    require(not result['has_failures'], 'arm reports failed runs')
    counts['arms_verified'] += 1
    return dict(fallback_native_cut=fallback_metrics['native_cut'], attempts=attempt_reports,
                stopped_reason=result['stopped_reason'], checkpoints=expected_checkpoints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, help='Default: results/audit.json')
    args = parser.parse_args()
    summary_path = args.results / 'summary.json'
    if not summary_path.exists():
        parser.error('A completed summary.json is required; do not audit during formal timing.')
    summary = json.loads(summary_path.read_text())
    if 'finished_utc' not in summary or 'source_sha256_after' not in summary:
        parser.error('Benchmark has not finished; defer CPU audit until all timing ends.')
    start = time.perf_counter()
    counts, errors, cases, file_hashes, assignment_hashes = Counter(), [], {}, {}, {}
    result = dict(audit_version=1, audit_utc=datetime.now(timezone.utc).isoformat(),
                  results=str(args.results.resolve()), script_sha256=digest(__file__),
                  evaluator_sha256=digest(ROOT / 'benchmarks/hypergraph/audit_fem_ier.py'),
                  python_version=sys.version, numpy_version=np.__version__,
                  summary_sha256=digest(summary_path),
                  implementation='Independent NumPy native evaluator: physical move application and occupied-label counting; no production imports.',
                  coverage=counts, cases=cases, errors=errors, verified_file_hashes=file_hashes,
                  assignment_array_sha256=assignment_hashes,
                  assignment_hash_definition='SHA256(sorted-key JSON of dtype.str and shape, NUL separator, C-contiguous array bytes).',
                  limitations=[
                      'Every saved selector and stage endpoint is audited; the internal flow trajectory is not saved or replayed.',
                      'IBM selector spaces are not exhaustively enumerated, and FEM gradient trajectories are not reoptimized.',
                      'Deadline eligibility is checked against recorded monotonic-clock events and frozen source; saved data cannot independently prove machine scheduling or absence of unrelated system load.',
                      'Failed or validation-failed attempts make this strict audit fail; partial failed-run coverage is not claimed.',
                      'This finite experiment does not establish an asymptotic OGP or a general algorithmic guarantee.'
                  ])

    def hashes(mapping, kind):
        for name, expected in mapping.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            actual = digest(path)
            require(actual == expected, f'{kind} hash mismatch: {path}')
            file_hashes[str(path.resolve())] = actual
            counts[kind + '_hashes_verified'] += 1

    def failure(scope, exc):
        errors.append(dict(scope=scope, error=str(exc), traceback=traceback.format_exc()))

    config = summary['config']
    try:
        for kind, plural in [('source', 'sources'), ('input', 'inputs')]:
            require(summary[kind + '_sha256_before'] == summary[kind + '_sha256_after'], f'{kind} run-time mutation')
            require(summary[plural + '_unchanged_during_run'], f'{kind} unchanged flag')
            hashes(summary[kind + '_sha256_before'], kind)
        manifest_path = args.results / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        file_hashes[str(manifest_path.resolve())] = digest(manifest_path)
        for field in ('config', 'source_sha256_before', 'input_sha256_before', 'started_utc'):
            require(manifest[field] == summary[field], f'manifest/summary {field}')
        require(config['arms'] == ['fem', 'pairs', 'deterministic', 'flow'], 'formal arm inventory')
        require(config['budgets_seconds'] == sorted(set(config['budgets_seconds'])), 'deadline order/uniqueness')
        require(config['primary_budget_seconds'] == max(config['budgets_seconds']), 'primary deadline')
        require([(case['instance'], case['seed']) for case in summary['cases']] ==
                [(name, seed) for name in config['instances'] for seed in config['seeds']], 'case inventory/order')
    except Exception as exc:
        failure('provenance', exc)
    originals, previous_arm_finish = {}, None
    for index, case in enumerate(summary['cases']):
        case_id = f"{case['instance']}-seed{case['seed']}"
        print(f'Auditing {case_id}', flush=True)
        case_report = cases[case_id] = dict(arms={})
        directory = args.results / case_id
        try:
            require(case['status'] == 'completed' and case['coarse_mapping_and_weights_match_saved'], 'case setup status')
            arms = config['arms']
            require(case['arm_order'] == arms[index % len(arms):] + arms[:index % len(arms)], 'cyclic arm order')
            require(set(case['arms']) == set(arms), 'missing formal arm')
            name = case['instance']
            if name not in originals:
                path = next(Path(p) for p in summary['input_sha256_before'] if p.endswith('/' + name + '.hgr'))
                originals[name] = read_hgr(path, config['q'], config['epsilon'])
            hierarchy = load_bundle(directory, case['hierarchy_artifact'], counts, file_hashes, assignment_hashes)
            graphs, coarse_graph, lifted = audit_hierarchy(hierarchy, originals[name], config['q'], config['epsilon'], counts)
            baseline_path = next(Path(p) for p in summary['input_sha256_before'] if p.endswith('/' + case_id + '/runs.json'))
            baseline = json.loads(baseline_path.read_text())
            reference = baseline['single_runs']['fem']['assignment_artifacts']['coarse_assignment']
            archive_path = baseline_path.parent / reference['file']
            require(str(archive_path.resolve()) in summary['input_sha256_before'], 'saved-start archive not hashed as input')
            with np.load(archive_path, allow_pickle=False) as archive:
                require(np.array_equal(hierarchy['coarse_assignment'], archive[reference['key']]), 'given coarse assignment differs from saved FEM input')
                require(np.array_equal(hierarchy['original_to_coarse'], archive['original_to_coarse']), 'saved hierarchy mapping mismatch')
                require(np.array_equal(hierarchy['coarse_node_weights'], archive['coarse_node_weights']), 'saved coarse node weights mismatch')
            counts['baseline_coarse_starts_and_mappings_verified'] += 1
            case_report['coarse_native_cut'] = coarse_graph.measure(hierarchy['coarse_assignment'])['native_cut']
        except Exception as exc:
            failure(case_id + '/hierarchy', exc)
            case_report['passed'] = False
            continue
        for arm in case['arm_order']:
            print(f'  {arm}: replaying saved stages and selector masks', flush=True)
            try:
                data = load_bundle(directory, case['arms'][arm]['artifact'], counts, file_hashes, assignment_hashes)
                if previous_arm_finish is not None:
                    require(data['started_at'] >= previous_arm_finish, 'measured arms overlap or disagree with arm order')
                previous_arm_finish = data['finished_at']
                for field, value in case['arms'][arm].items():
                    if field != 'artifact':
                        require(jsonable(data[field]) == value, f'arm summary {field}')
                case_report['arms'][arm] = audit_arm(data, arm, case['seed'], config['budgets_seconds'], hierarchy,
                    graphs, lifted, counts, config['frontend'][arm], config['seed_stride'])
                case_report['arms'][arm]['passed'] = True
            except Exception as exc:
                failure(case_id + '/' + arm, exc)
                case_report['arms'][arm] = dict(passed=False)
        case_report['passed'] = all(arm.get('passed') for arm in case_report['arms'].values())
        counts['cases_passed'] += int(case_report['passed'])
    result['passed'] = not errors
    result['elapsed_seconds'] = time.perf_counter() - start
    output = args.output or args.results / 'audit.json'
    output.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=result['passed'], coverage=counts, errors=errors, output=str(output)), indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
