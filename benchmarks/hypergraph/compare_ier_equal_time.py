"""Equal-deadline complete V-cycle comparison from saved coarse FEM starts.

The common hierarchy is reconstructed and checked before timing. Each arm pays
for lifting/scoring its fallback, configuration, copies, full solver work,
diagnostic/state capture, native candidate scoring and incumbent selection.
Only independent verification and artifact serialization occur after arm timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

# Set native-library thread limits before importing NumPy/Torch when invoked
# as a program. Importing this module for tests does not mutate their settings.
if __name__ == '__main__':
    for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_thread_variable] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hyper_solver import HyperRefineSolver, KahyparLikeSolver, vcycle_uncoarsen
from src.partition.hyper_quotient import connectivity_cost
from src.partition.hyper_refine_contract import capacity_state
from benchmarks.hypergraph.compare_fem_multiseed import digest, measure, save
from benchmarks.hypergraph.validate_fem_ier import saved_ibm_case, check_hierarchy
from benchmarks.hypergraph.time_budget_hgr import read_unweighted_hgr

ARMS = ('fem', 'pairs', 'deterministic', 'flow')
SEED_STRIDE = 1_000_003


def frontend_options(arm, seed, epsilon):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    return dict(mode_cycle=('flow',) if arm == 'flow' else ('fem_ier', 'flow'),
                ier_backend='fem' if arm == 'flow' else arm,
                ier_rounds=2, ier_max_moves=24, ier_boundary_pool=96,
                ier_pool_strategy='local', ier_num_trials=8, ier_num_steps=100,
                ier_random_samples=800, ier_allow_triples=True,
                flow_passes=2, seed=seed, max_imbalance=epsilon, repair_balance=True)


def arm_order(case_index):
    offset = case_index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def production_metrics(assignment, edges, nodes, weights, q, epsilon):
    _, _, loads, capacity, tolerance = capacity_state(assignment, q, nodes, epsilon)
    return {'native_cut': float(connectivity_cost(assignment, edges, weights)),
            'block_loads': loads.copy(), 'capacity': float(capacity),
            'feasible': bool(np.all(loads <= capacity + tolerance))}


class StateRecorder(HyperRefineSolver):
    """Capture every refine input/output inside the timed production call."""
    def __init__(self, graph_ids):
        super().__init__()
        self.graph_ids, self.stages = graph_ids, []

    def refine(self, assignment, edges, q, node_weights=None, hyperedge_weights=None, **kwargs):
        stage = {'graph_id': self.graph_ids[len(self.stages)],
                 'initial_assignment': np.asarray(assignment, dtype=np.int64).copy()}
        self.stages.append(stage)
        try:
            final = super().refine(assignment, edges, q, node_weights=node_weights,
                                   hyperedge_weights=hyperedge_weights, **kwargs)
            stage.update(final_assignment=np.asarray(final, dtype=np.int64).copy(),
                         last_result=getattr(self, 'last_result', None), status='success')
            # Production replaces last_result on each call; keeping this object
            # retains complete diagnostics without an arm-specific deep copy.
            return final
        except Exception as exc:
            stage.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                         traceback=traceback.format_exc())
            raise


def run_attempt(arm, kind, seed, continuation, coarse_start, coarse, edges, nodes, weights,
                q, epsilon):
    """No timer boundary here: caller times all setup, scoring and capture."""
    record = {'arm': arm, 'kind': kind, 'solver_seed': seed}
    try:
        if kind == 'vcycle':
            record['coarse_assignment'] = np.asarray(coarse_start, dtype=np.int64).copy()
            record['initial_assignment'] = record['coarse_assignment'][coarse['original_to_coarse']].copy()
            graph_ids = [f'level_{i}' for i in reversed(range(len(coarse['hierarchy_stack'])))] + ['original']
        elif kind == 'flow_continuation' and arm == 'flow' and continuation is not None:
            record['initial_assignment'] = np.asarray(continuation, dtype=np.int64).copy()
            graph_ids = ['original']
        else:
            raise ValueError('invalid continuation kind/state')
        refiner = StateRecorder(graph_ids)
        record['stages'] = refiner.stages
        refiner.update_params(**frontend_options(arm, seed, epsilon))
        if kind == 'vcycle':
            final = vcycle_uncoarsen(record['coarse_assignment'].copy(), coarse['hierarchy_stack'],
                edges, q, refiner, verbose=False, node_weights=nodes, hyperedge_weights=weights)
        else:
            final = refiner.refine(record['initial_assignment'].copy(), edges, q,
                                   node_weights=nodes, hyperedge_weights=weights)
        record['final_assignment'] = np.asarray(final, dtype=np.int64).copy()
        record['native_metrics'] = production_metrics(final, edges, nodes, weights, q, epsilon)
        if not record['native_metrics']['feasible']:
            raise AssertionError('returned candidate violates original capacity')
        record['status'] = 'success'
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    return record


def collect_arm(arm, base_seed, budgets, prepare_fallback, run_once, *,
                clock=time.perf_counter, max_attempts=10000):
    """One uninterrupted timed arm; callbacks perform all algorithmic work.

    run_once receives (kind, seed, continuation). Full V-cycles always receive
    None; only flow's continuation receives this arm's own previous final.
    Checkpoint selection/independent verification happens after this function.
    """
    budgets = sorted(set(float(b) for b in budgets))
    if arm not in ARMS or not budgets or any(not math.isfinite(b) or b < 0 for b in budgets):
        raise ValueError('invalid arm or budgets')
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    start = clock()
    result = {'arm': arm, 'base_seed': base_seed, 'budgets_seconds': budgets,
              'started_at': start, 'records': []}
    try:
        fallback = prepare_fallback()
        fallback.update(source='initial_only', status='success')
        if not fallback['native_metrics']['feasible']:
            raise AssertionError('given lifted coarse fallback is infeasible')
        result['fallback'] = fallback
        incumbent_available = fallback['native_metrics']['native_cut']
        fallback['available_seconds'] = clock() - start
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), stopped_reason='fallback_failed')
        result['finished_at'] = clock()
        result['wall_seconds_including_overshoot'] = result['finished_at'] - start
        return result
    current, incumbent_complete = None, float('inf')
    reason = 'deadline'
    while len(result['records']) < max_attempts:
        attempt_start = clock()
        if attempt_start - start >= budgets[-1]:
            break
        attempt = len(result['records'])
        kind = 'flow_continuation' if arm == 'flow' and attempt else 'vcycle'
        seed = base_seed + SEED_STRIDE * attempt
        try:
            record = run_once(kind, seed, current if kind == 'flow_continuation' else None)
        except Exception as exc:
            record = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                      'traceback': traceback.format_exc()}
        record.update(run_id=f'{arm}_{attempt:04d}', attempt=attempt, kind=kind,
                      solver_seed=seed, started_at=attempt_start,
                      predecessor_run_id=(result['records'][-1]['run_id'] if kind == 'flow_continuation' else None))
        converged = False
        if record['status'] == 'success':
            metrics = record['native_metrics']
            if not metrics['feasible']:
                record.update(status='failed', error='candidate infeasible before availability')
            else:
                cut = metrics['native_cut']
                record['improves_completed_incumbent'] = cut < incumbent_complete
                record['improves_available_incumbent'] = cut < incumbent_available
                if cut < incumbent_complete:
                    incumbent_complete = cut
                if cut < incumbent_available:
                    incumbent_available = cut
                if arm == 'flow':
                    converged = kind == 'flow_continuation' and np.array_equal(
                        record['initial_assignment'], record['final_assignment'])
                    current = record['final_assignment'].copy()
        record['exact_assignment_fixed_point'] = converged
        result['records'].append(record)
        # Native scoring, feasibility, incumbent comparisons, continuation copy,
        # and diagnostic capture all precede this availability event.
        complete = clock()
        record.update(finished_at=complete, completion_seconds=complete-start,
                      pipeline_seconds=complete-attempt_start,
                      completed_within_max_budget=complete-start <= budgets[-1])
        if complete - start > budgets[-1]:
            reason = 'overshoot'
            break
        if arm == 'flow' and record['status'] != 'success':
            reason = 'flow_failed'
            break
        if converged:
            reason = 'exact_assignment_fixed_point'
            break
    if len(result['records']) >= max_attempts and reason == 'deadline':
        reason = 'max_attempts'
    result.update(status='execution_complete', stopped_reason=reason, finished_at=clock())
    result['wall_seconds_including_overshoot'] = result['finished_at'] - start
    result['attempts'] = len(result['records'])
    result['max_attempts_reached'] = reason == 'max_attempts'
    result['overshoot_seconds'] = max(0., result['wall_seconds_including_overshoot']-budgets[-1])
    return result


def choose_checkpoints(result):
    """Use only independently verified complete returns before each deadline."""
    checkpoints = []
    for budget in result['budgets_seconds']:
        eligible = [r for r in result['records'] if r['completion_seconds'] <= budget
                    and r['status'] == 'success' and r.get('final', {}).get('feasible', False)]
        winner = min(eligible, key=lambda r: (r['final']['native_cut'], r['completion_seconds'])) if eligible else None
        completed = None if winner is None else {
            'source': 'completed_solver', 'run_id': winner['run_id'], 'native_cut': winner['final']['native_cut'],
            'block_loads': winner['final']['block_loads'], 'available_seconds': winner['completion_seconds']}
        fallback = result.get('fallback', {})
        initial = ({'source': 'initial_only', 'run_id': None, 'native_cut': fallback['verified']['native_cut'],
                    'block_loads': fallback['verified']['block_loads'], 'available_seconds': fallback['available_seconds']}
                   if fallback.get('status') == 'success' and fallback['available_seconds'] <= budget
                   and fallback.get('verified', {}).get('feasible', False) else None)
        choices = [x for x in (initial, completed) if x is not None]
        available = min(choices, key=lambda x: (x['native_cut'], x['available_seconds'])) if choices else None
        checkpoints.append({'budget_seconds': budget, 'optimized_status': 'success' if completed else 'no_result',
                            'eligible_completed_runs': len(eligible),
                            'eligible_completed_vcycles': sum(r['kind'] == 'vcycle' for r in eligible),
                            'eligible_flow_continuations': sum(r['kind'] == 'flow_continuation' for r in eligible),
                            'best_completed': completed,
                            'best_available': available})
    result['checkpoints'] = checkpoints


def graph_data(coarse, edges, nodes, weights):
    graphs = {'original': {'edges': edges, 'nodes': np.asarray(nodes), 'weights': np.asarray(weights)}}
    for i, level in enumerate(coarse['hierarchy_stack']):
        graphs[f'level_{i}'] = {'edges': level['hyperedges'], 'nodes': np.asarray(level['node_weights']),
                              'weights': np.asarray(level['hyperedge_weights'])}
    return graphs


def verify_arm(result, coarse_start, coarse, graphs, q, epsilon):
    """External double-implementation audit after the entire arm timer stops."""
    original = graphs['original']

    def checked(state, graph):
        metrics = measure(state, graph['edges'], graph['nodes'], graph['weights'], q, epsilon)
        if not metrics['feasible']:
            raise AssertionError('infeasible saved state')
        return metrics

    fallback = result.get('fallback')
    if fallback:
        try:
            fallback['verified'] = checked(fallback['assignment'], original)
            np.testing.assert_array_equal(fallback['assignment'], np.asarray(coarse_start)[coarse['original_to_coarse']])
            np.testing.assert_allclose(fallback['verified']['native_cut'], fallback['native_metrics']['native_cut'])
        except Exception as exc:
            fallback.update(status='validation_failed', validation_error=str(exc), validation_traceback=traceback.format_exc())
    previous = None
    for record in result['records']:
        if record['status'] != 'success':
            previous = record
            continue
        try:
            record['initial'] = checked(record['initial_assignment'], original)
            record['final'] = checked(record['final_assignment'], original)
            np.testing.assert_allclose(record['native_metrics']['native_cut'], record['final']['native_cut'])
            if record['final']['native_cut'] > record['initial']['native_cut'] + 1e-10:
                raise AssertionError('complete candidate increased native cut')
            if record['kind'] == 'vcycle':
                np.testing.assert_array_equal(record['coarse_assignment'], coarse_start)
                np.testing.assert_array_equal(record['initial_assignment'], np.asarray(coarse_start)[coarse['original_to_coarse']])
                current = np.asarray(coarse_start)
                levels = list(reversed(coarse['hierarchy_stack']))
                expected_graphs = [f'level_{i}' for i in reversed(range(len(levels)))] + ['original']
            else:
                if previous is None or previous['status'] != 'success':
                    raise AssertionError('continuation lacks its own verified predecessor')
                np.testing.assert_array_equal(record['initial_assignment'], previous['final_assignment'])
                current, levels, expected_graphs = record['initial_assignment'], [], ['original']
            if [st['graph_id'] for st in record['stages']] != expected_graphs:
                raise AssertionError('incomplete or misordered V-cycle stages')
            for index, stage in enumerate(record['stages']):
                if index < len(levels):
                    remap = levels[index]['remap']
                    expected = np.array([current[remap[v]] for v in range(levels[index]['num_nodes'])])
                else:
                    expected = current
                np.testing.assert_array_equal(stage['initial_assignment'], expected)
                graph = graphs[stage['graph_id']]
                stage['initial'] = checked(stage['initial_assignment'], graph)
                stage['final'] = checked(stage['final_assignment'], graph)
                if stage['final']['native_cut'] > stage['initial']['native_cut'] + 1e-10:
                    raise AssertionError('stage increased native cut')
                current = stage['final_assignment']
            np.testing.assert_array_equal(current, record['final_assignment'])
            if record['exact_assignment_fixed_point'] and not np.array_equal(
                    record['initial_assignment'], record['final_assignment']):
                raise AssertionError('invalid flow fixed-point claim')
            record['independent_stage_and_projection_verification'] = True
        except Exception as exc:
            record.update(status='validation_failed', validation_error=str(exc), validation_traceback=traceback.format_exc())
        previous = record
    result['validation_summary'] = {
        'successful_runs': sum(r['status'] == 'success' for r in result['records']),
        'execution_failed_runs': sum(r['status'] == 'failed' for r in result['records']),
        'validation_failed_runs': sum(r['status'] == 'validation_failed' for r in result['records']),
        'fallback_valid': bool(fallback and fallback.get('status') == 'success'
                               and fallback.get('verified', {}).get('feasible', False)),
    }
    result['has_failures'] = (not result['validation_summary']['fallback_valid'] or any(
        r['status'] != 'success' for r in result['records']))
    choose_checkpoints(result)


def persist_bundle(directory, stem, value):
    """Persist every ndarray/tensor recursively to compressed NPZ after timing."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path, array_path = directory / (stem + '.json'), directory / (stem + '.npz')
    if json_path.exists() or array_path.exists():
        raise FileExistsError('artifact already exists')
    arrays = {}

    def extract(item, path=()):
        if isinstance(item, (np.ndarray, torch.Tensor)):
            key = '__'.join(path)
            arrays[key] = np.asarray(item.detach().cpu() if isinstance(item, torch.Tensor) else item)
            return {'artifact': array_path.name, 'key': key}
        if isinstance(item, dict):
            return {str(k): extract(v, path+(str(k),)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [extract(v, path+(str(i),)) for i, v in enumerate(item)]
        return item.item() if isinstance(item, np.generic) else item

    serialized = extract(value)
    np.savez_compressed(array_path, **arrays)
    save(json_path, serialized)
    return {'json': json_path.name, 'npz': array_path.name,
            'json_sha256': digest(json_path), 'npz_sha256': digest(array_path)}


def pack_ragged(rows):
    return {'offsets': np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64),
            'values': np.array([v for row in rows for v in row], dtype=np.int64)}


def hierarchy_artifact(coarse_start, coarse, graphs):
    result = {'coarse_assignment': np.asarray(coarse_start).copy(),
              'original_to_coarse': np.asarray(coarse['original_to_coarse']),
              'coarse_node_weights': np.asarray(coarse['coarse_node_weights']),
              'coarse_hyperedges': pack_ragged(coarse['coarse_hyperedges']),
              'coarse_hyperedge_weights': np.asarray(coarse['coarse_hyperedge_weights']),
              'graphs': {key: {'hyperedges': pack_ragged(g['edges']), 'node_weights': g['nodes'],
                               'hyperedge_weights': g['weights']} for key, g in graphs.items()},
              'hierarchy_stack': []}
    for i, level in enumerate(coarse['hierarchy_stack']):
        result['hierarchy_stack'].append({'graph_id': f'level_{i}', 'num_nodes': level['num_nodes'],
            'remap': np.array([level['remap'][v] for v in range(level['num_nodes'])], dtype=np.int64),
            'groups': pack_ragged(level['groups'])})
    return result


def run_timed_arm(arm, seed, budgets, coarse_start, coarse, edges, nodes, weights, q, epsilon, *, clock=time.perf_counter):
    def fallback():
        state = np.asarray(coarse_start, dtype=np.int64)[coarse['original_to_coarse']].copy()
        return {'assignment': state, 'native_metrics': production_metrics(state, edges, nodes, weights, q, epsilon)}

    def once(kind, attempt_seed, continuation):
        return run_attempt(arm, kind, attempt_seed, continuation, coarse_start, coarse,
                           edges, nodes, weights, q, epsilon)

    return collect_arm(arm, seed, budgets, fallback, once, clock=clock)


def warmup():
    edges = [[0, 2, 4], [1, 3, 5], [2, 6], [3, 7], [0, 1], [4, 5]]
    nodes, weights, start = np.ones(8), np.ones(len(edges)), np.array([0, 0, 1, 1, 0, 0, 1, 1])
    coarse = {'original_to_coarse': np.arange(8), 'hierarchy_stack': []}
    report = {'excluded_from_all_main_timing': True, 'arms': {}}
    for arm in ARMS:
        # A single complete tiny path initializes the FEM and new sparse backends.
        record = run_attempt(arm, 'vcycle', 20260924, None, start, coarse, edges, nodes, weights, 2, 0.)
        if record['status'] != 'success':
            report['arms'][arm] = record
            report['status'] = 'failed'
            return report
        verified = measure(record['final_assignment'], edges, nodes, weights, 2, 0.)
        if not verified['feasible']:
            raise AssertionError('warmup output infeasible')
        report['arms'][arm] = {'status': 'success', 'final': verified,
            'ier_rounds_with_nonempty_pool': sum(
                h.get('num_moves', 0) > 0 for stage in record['stages']
                for diag in (stage.get('last_result') or {}).get('stages', [])
                for h in diag.get('history', []))}
    report['status'] = 'success'
    return report


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / 'src/hyper_solver.py',
             ROOT / 'benchmarks/hypergraph/IER_EQUAL_TIME_PROTOCOL.md',
             ROOT / 'tests/test_ier_equal_time_benchmark.py',
             ROOT / 'benchmarks/hypergraph/validate_fem_ier.py',
             ROOT / 'benchmarks/hypergraph/compare_fem_multiseed.py',
             ROOT / 'benchmarks/hypergraph/time_budget_hgr.py',
             ROOT / 'benchmarks/hypergraph/native_search_probe.py',
             ROOT / 'benchmarks/hypergraph/native_mcmc.py']
    paths += sorted((ROOT / 'src/partition').rglob('*.py'))
    paths += sorted((ROOT / 'lib/qubo-solver/src/fem').rglob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'benchmarks/hypergraph/results/fem-multiseed-20260924-v2')
    parser.add_argument('--ibm-directory', type=Path, default=Path('/private/tmp/ising-hgr.K5jm2Y'))
    parser.add_argument('--instances', nargs='+', choices=['ibm01', 'ibm02'], default=['ibm01', 'ibm02'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(30, 35)))
    parser.add_argument('--budgets', type=float, nargs='+', default=[5., 10., 20.])
    args = parser.parse_args()
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
            or len(set(args.instances)) != len(args.instances)
            or not args.budgets or any(not math.isfinite(b) or b <= 0 for b in args.budgets)):
        parser.error('seeds must be unique/nonnegative and budgets finite/positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest_path = args.baseline / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    input_hashes = {str(manifest_path.resolve()): digest(manifest_path)}
    # Load and hash every original saved start before the first measured arm.
    cases = []
    for name in args.instances:
        path = args.ibm_directory / (name + '.hgr')
        if digest(path) != next(d['sha256'] for d in manifest['inputs'] if d['name'] == name):
            raise ValueError('input hgr differs from baseline manifest')
        input_hashes[str(path.resolve())] = digest(path)
        n, edges = read_unweighted_hgr(path)
        for seed in args.seeds:
            saved = saved_ibm_case(args.baseline, name, seed)
            input_hashes.update(saved['input_hashes'])
            cases.append((name, seed, edges, np.ones(n), np.ones(len(edges)), saved))
    before = source_hashes()
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'invocation_argv': sys.argv, 'working_directory': str(Path.cwd()),
              'config': {'seeds': args.seeds, 'instances': args.instances, 'budgets_seconds': sorted(set(args.budgets)),
                         'primary_budget_seconds': max(args.budgets), 'q': 4, 'epsilon': .03,
                         'seed_stride': SEED_STRIDE, 'arms': ARMS,
                         'frontend': {arm: frontend_options(arm, 30, .03) for arm in ARMS},
                         'order': 'Cyclic rotation across cases; position counts differ by at most one.',
                         'fallback': 'Each arm pays for coarse lifting and production native scoring after its timer starts.',
                         'flow': 'One V-cycle, then own original-graph flow(2 passes) until identical assignment or deadline.',
                         'other_arms': 'Independent complete V-cycle restarts from the same saved coarse assignment.',
                         'timing': 'All algorithm work, recording arrays/diagnostics, native scoring and incumbent comparisons included; external audit and I/O after whole arm.'},
              'source_sha256_before': before, 'input_sha256_before': input_hashes.copy(),
              'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
                  'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'platform': platform.platform(),
                  'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}},
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'submodule_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'lib/qubo-solver', text=True).strip(),
              'cases': []}
    report['warmup'] = warmup()
    save(args.output / 'manifest.json', report)
    if report['warmup']['status'] != 'success':
        raise SystemExit('warmup failed; no formal arm started')
    for index, (name, seed, edges, nodes, weights, saved) in enumerate(cases):
        case = {'instance': name, 'seed': seed, 'arm_order': arm_order(index), 'arms': {}}
        report['cases'].append(case)
        directory = args.output / f'{name}-seed{seed}'
        try:
            coarse = KahyparLikeSolver().coarsen(edges, len(nodes), 4, coarsen_to=200, seed=seed,
                score_mode='hem', enforce_balance_cap=True, epsilon=.03, node_weights=nodes, hyperedge_weights=weights)
            check_hierarchy(coarse, saved)
            case['coarse_mapping_and_weights_match_saved'] = True
            graphs = graph_data(coarse, edges, nodes, weights)
            case['hierarchy_artifact'] = persist_bundle(directory, 'hierarchy', hierarchy_artifact(saved['coarse_assignment'], coarse, graphs))
            for arm in case['arm_order']:
                result = run_timed_arm(arm, seed, args.budgets, saved['coarse_assignment'], coarse,
                                       edges, nodes, weights, 4, .03)
                verify_arm(result, saved['coarse_assignment'], coarse, graphs, 4, .03)
                artifact = persist_bundle(directory, arm, result)
                case['arms'][arm] = {k: result[k] for k in ('status', 'stopped_reason', 'attempts',
                    'wall_seconds_including_overshoot', 'overshoot_seconds', 'checkpoints',
                    'validation_summary', 'has_failures') if k in result}
                case['arms'][arm]['artifact'] = artifact
                save(args.output / 'progress.json', report)
                print(f'{name} seed={seed} arm={arm} attempts={len(result["records"])} '
                      f'stop={result["stopped_reason"]} checkpoints='
                      f'{[(p["budget_seconds"],p["optimized_status"],None if p["best_completed"] is None else p["best_completed"]["native_cut"]) for p in result["checkpoints"]]}', flush=True)
            case['status'] = 'completed'
        except Exception as exc:
            case.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            print(f'{name} seed={seed} setup/artifact error: {exc}', flush=True)
            save(args.output / 'progress.json', report)
    report['source_sha256_after'] = source_hashes()
    report['input_sha256_after'] = {path: digest(path) for path in input_hashes}
    report['sources_unchanged_during_run'] = report['source_sha256_after'] == before
    report['inputs_unchanged_during_run'] = report['input_sha256_after'] == input_hashes
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(args.output / 'summary.json', report)
    if not report['sources_unchanged_during_run'] or not report['inputs_unchanged_during_run']:
        raise SystemExit('source/input changed; benchmark invalid')


if __name__ == '__main__':
    main()
