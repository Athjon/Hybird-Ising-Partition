"""Paired production FEM/greedy comparison with strict greedy-restart budgets.

For each input and seed, coarsen once and alternate the order of single greedy
and native FEM. Time initialization plus V-cycle identically for both. External
quality checks and artifact I/O are outside all timed intervals. Greedy restarts
run consecutively without intervening quality checks or writes; their completion
timestamps are measured against the FEM downstream duration. An over-budget
completion is retained for auditing but can never win the budgeted comparison.

Example (use an unused output directory):
    python benchmarks/hypergraph/compare_fem_multiseed.py --seeds 30 31 32 \
        --ibm-directory /path/to/inputs --output /path/to/new/results
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hyper_solver import FemCoarsenSolver, HyperRefineSolver, KahyparLikeSolver, vcycle_uncoarsen
from src.partition.hyper_objective import capacity_limits
from src.partition.hyper_quotient import connectivity_cost
from benchmarks.hypergraph.time_budget_hgr import read_unweighted_hgr


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    selected = [Path(__file__).resolve(), ROOT / 'src/hyper_solver.py',
        ROOT / 'src/partition/hyper_objective.py', ROOT / 'src/partition/hyper_utils.py',
        ROOT / 'src/partition/hyper_refine_contract.py', ROOT / 'src/partition/hyper_quotient.py',
        ROOT / 'benchmarks/hypergraph/time_budget_hgr.py']
    selected.extend(sorted((ROOT / 'lib/qubo-solver/src/fem').rglob('*.py')))
    return {str(p.relative_to(ROOT)): digest(p) for p in selected}


def measure(assignment, edges, nodes, edge_weights, q, epsilon):
    assignment = np.asarray(assignment)
    nodes = np.asarray(nodes, dtype=np.float64)
    edge_weights = np.asarray(edge_weights, dtype=np.float64)
    if assignment.shape != (len(nodes),) or not np.issubdtype(assignment.dtype, np.integer):
        raise AssertionError('invalid assignment shape or dtype')
    if np.any(assignment < 0) or np.any(assignment >= q):
        raise AssertionError('assignment labels outside [0,q)')
    cut = float(connectivity_cost(assignment, edges, edge_weights))
    independent_cut = sum(float(w) * max(0, len({int(assignment[v]) for v in edge}) - 1)
                          for edge, w in zip(edges, edge_weights))
    if not np.isclose(cut, independent_cut, atol=1e-10, rtol=1e-12):
        raise AssertionError('native cut disagrees with independent direct computation')
    loads = np.bincount(assignment, weights=nodes, minlength=q)
    independent_loads = [sum(float(w) for w, label in zip(nodes, assignment) if label == b) for b in range(q)]
    np.testing.assert_allclose(loads, independent_loads, atol=1e-10, rtol=1e-12)
    capacity, tolerance = capacity_limits(nodes, q, epsilon)
    return {'native_cut': cut, 'block_loads': loads, 'capacity': capacity,
            'capacity_tolerance': tolerance, 'feasible': bool(np.all(loads <= capacity + tolerance)),
            'independent_cut_and_loads_verified': True}


def run_pipeline(method, solver_seed, coarse, edges, nodes, weights, *, q=4, epsilon=.03):
    """Timed production execution only; callers perform external verification."""
    record = {'method': method, 'solver_seed': solver_seed}
    start = time.perf_counter()
    stage = 'initial_partition'
    try:
        if method == 'fem':
            initializer = FemCoarsenSolver()
            assignment = initializer.initial_partition(coarse['coarse_hyperedges'],
                coarse['coarse_node_weights'], q, hyperedge_weights=coarse['coarse_hyperedge_weights'],
                method='fem', map_type='native', num_trials=8, num_steps=150,
                seed=solver_seed, epsilon=epsilon, dev='cpu', dtype=torch.float64,
                anneal='exp', betamin=.5, betamax=50., learning_rate=.08,
                optimizer='adam', h_factor=.1, imbalance_weight=5.,
                exact_balance_max_nodes=20, balance_search_budget=200000)
        elif method == 'greedy':
            initializer = KahyparLikeSolver()
            assignment = initializer.initial_partition_greedy(coarse['coarse_hyperedges'],
                coarse['coarse_node_weights'], q, hyperedge_weights=coarse['coarse_hyperedge_weights'],
                seed=solver_seed, epsilon=epsilon)
        else:
            raise ValueError('method must be fem or greedy')
        init_end = time.perf_counter()
        record['coarse_assignment'] = assignment
        record['initial_partition_seconds'] = init_end - start
        stage = 'vcycle'
        refiner = HyperRefineSolver()
        refiner.update_params(mode_cycle=('flow',), flow_passes=2,
                              max_imbalance=epsilon, repair_balance=True)
        final = vcycle_uncoarsen(assignment, coarse['hierarchy_stack'], edges, q, refiner,
                                verbose=False, node_weights=nodes, hyperedge_weights=weights)
        finish = time.perf_counter()
        record.update(status='success', final_assignment=final,
                      vcycle_seconds=finish-init_end)
        if method == 'fem':
            record['solver_details'] = initializer.last_result
    except Exception as exc:
        finish = time.perf_counter()
        record.update(status='failed', failed_stage=stage, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    record.update(started_at=start, finished_at=finish, pipeline_seconds=finish-start)
    return record


def verify_run(record, coarse, edges, nodes, weights, q=4, epsilon=.03):
    """Validate after timing, including failed runs that produced an initial state."""
    try:
        if 'coarse_assignment' in record:
            initial = record['coarse_assignment']
            record['coarse_initial'] = measure(initial, coarse['coarse_hyperedges'],
                coarse['coarse_node_weights'], coarse['coarse_hyperedge_weights'], q, epsilon)
            lifted = initial[coarse['original_to_coarse']]
            record['lifted_initial'] = measure(lifted, edges, nodes, weights, q, epsilon)
            if record['coarse_initial']['native_cut'] != record['lifted_initial']['native_cut']:
                raise AssertionError('quotient and lifted native cuts disagree')
            np.testing.assert_allclose(record['coarse_initial']['block_loads'], record['lifted_initial']['block_loads'])
        if record['status'] == 'success':
            record['final'] = measure(record['final_assignment'], edges, nodes, weights, q, epsilon)
            if not record['final']['feasible']:
                raise AssertionError('returned final partition exceeds capacity')
            if record['coarse_initial']['feasible'] and record['final']['native_cut'] > record['coarse_initial']['native_cut'] + 1e-10:
                raise AssertionError('refinement increased native cut from a feasible initial state')
    except Exception as exc:
        record['execution_status_before_validation'] = record['status']
        record.update(status='validation_failed', validation_error_type=type(exc).__name__,
                      validation_error=str(exc), validation_traceback=traceback.format_exc())


def deterministic_warmup():
    """Initialize both production paths before any measured IBM pipeline.

    Strong disjoint pairs and four-pin ring edges give a fixed 16-node input.
    Its results are recorded solely as warmup checks; they are never passed to
    the dataset loop, restart collector, or benchmark winner selection.
    """
    seed, n, q = 20260924, 16, 4
    pairs = [[2*i, 2*i+1] for i in range(n//2)]
    edges = [edge.copy() for edge in pairs for _ in range(3)]
    edges += [pairs[i] + pairs[(i+1) % len(pairs)] for i in range(len(pairs))]
    nodes, weights = np.ones(n), np.ones(len(edges))
    result = {'seed': seed, 'nodes': n, 'edges': len(edges), 'q': q, 'coarsen_to': 8,
              'pipeline_order': ['fem', 'greedy'], 'included_in_benchmark_results': False,
              'used_for_restart_budget_or_candidates': False, 'single_runs': {}}
    start = time.perf_counter()
    try:
        coarse = KahyparLikeSolver().coarsen(edges, n, q, coarsen_to=8, seed=seed,
            score_mode='hem', enforce_balance_cap=True, epsilon=.03,
            node_weights=nodes, hyperedge_weights=weights)
        result['coarsening_seconds'] = time.perf_counter()-start
        result['actual_coarse_nodes'] = len(coarse['coarse_node_weights'])
        for method in result['pipeline_order']:
            run = run_pipeline(method, seed, coarse, edges, nodes, weights, q=q, epsilon=.03)
            verify_run(run, coarse, edges, nodes, weights, q=q, epsilon=.03)
            result['single_runs'][method] = {
                key: run[key] for key in ('status', 'pipeline_seconds', 'initial_partition_seconds',
                    'vcycle_seconds', 'coarse_initial', 'final', 'error', 'validation_error') if key in run}
        result['status'] = 'success' if all(run['status'] == 'success'
            for run in result['single_runs'].values()) else 'failed'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result['total_wall_seconds_including_verification'] = time.perf_counter()-start
    return result


def collect_restarts(run_once, budget_seconds, *, clock=time.perf_counter, max_attempts=10000):
    """Run consecutive full pipelines; preserve one possible overshoot.

    run_once(attempt) returns a record whose finished_at uses the same clock.
    This function does not verify or select a winner, so checks and disk writes
    cannot consume only the restart arm's budget. A failed attempt still costs
    time. No previously computed candidate is passed into this function.
    """
    if not math.isfinite(budget_seconds) or budget_seconds < 0 or max_attempts < 1:
        raise ValueError('budget must be finite/nonnegative and max_attempts positive')
    start = clock()
    records = []
    while len(records) < max_attempts and clock() - start < budget_seconds:
        attempt = len(records)
        record = run_once(attempt)
        completion = record['finished_at'] - start
        record.update(attempt=attempt, completion_seconds=completion,
                      completed_within_budget=completion <= budget_seconds)
        records.append(record)
        if completion > budget_seconds:
            break
    elapsed = clock() - start
    return {'budget_seconds': budget_seconds, 'records': records, 'wall_seconds_including_overshoot': elapsed,
            'attempts': len(records), 'max_attempts_reached': len(records) >= max_attempts,
            'overshoot_seconds': max([r['completion_seconds'] - budget_seconds for r in records] + [0.]),
            'timing_definition': 'Wall completion time from restart-loop start; verification and artifact writes occur after this loop.'}


def choose_restart_winner(restarts):
    """Select only independently validated results completed by the deadline."""
    eligible = [r for r in restarts['records'] if r['completed_within_budget']
                and r['status'] == 'success' and r.get('final', {}).get('feasible', False)]
    restarts['eligible_completed_runs'] = len(eligible)
    if not eligible:
        restarts.update(status='no_result', winner_run_id=None, winner=None)
        return
    winner = min(eligible, key=lambda r: (r['final']['native_cut'], r['completion_seconds']))
    restarts.update(status='success', winner_run_id=winner['run_id'], winner={
        'native_cut': winner['final']['native_cut'], 'block_loads': winner['final']['block_loads'],
        'solver_seed': winner['solver_seed'], 'completion_seconds': winner['completion_seconds'],
        'pipeline_seconds': winner['pipeline_seconds']})


def persist_unit(directory, unit, coarse):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {'original_to_coarse': coarse['original_to_coarse'],
              'coarse_node_weights': np.asarray(coarse['coarse_node_weights'])}
    records = list(unit.get('single_runs', {}).values()) + unit.get('greedy_restarts', {}).get('records', [])
    for record in records:
        refs = {}
        for field in ('coarse_assignment', 'final_assignment'):
            if field in record:
                key = record['run_id'] + '_' + field
                arrays[key] = record[field]
                refs[field] = {'file': 'assignments.npz', 'key': key}
        record['assignment_artifacts'] = refs
    np.savez_compressed(directory / 'assignments.npz', **arrays)

    def without_arrays(value):
        if isinstance(value, dict):
            return {k: without_arrays(v) for k, v in value.items()
                    if not (k in ('coarse_assignment', 'final_assignment') and isinstance(v, (np.ndarray, torch.Tensor)))}
        if isinstance(value, list):
            return [without_arrays(v) for v in value]
        return value
    save(directory / 'runs.json', without_arrays(unit))


def compact_unit(unit):
    result = {k: v for k, v in unit.items() if k not in ('single_runs', 'greedy_restarts')}
    result['single_runs'] = {method: {k: run[k] for k in ('status', 'pipeline_seconds', 'initial_partition_seconds', 'vcycle_seconds', 'final', 'error', 'validation_error') if k in run}
                             for method, run in unit.get('single_runs', {}).items()}
    if 'greedy_restarts' in unit:
        result['greedy_restarts'] = {k: v for k, v in unit['greedy_restarts'].items() if k != 'records'}
    return result


def aggregates(units):
    result = []
    for name in sorted({u['instance'] for u in units}):
        rows = [u for u in units if u['instance'] == name]
        report = {'instance': name, 'seed_count': len(rows), 'comparisons': {}}
        for control in ('single_greedy', 'greedy_restarts'):
            pairs = []
            for row in rows:
                fem = row.get('single_runs', {}).get('fem', {})
                baseline = row.get('single_runs', {}).get('greedy', {}) if control == 'single_greedy' else row.get('greedy_restarts', {})
                if fem.get('status') == baseline.get('status') == 'success':
                    baseline_cut = baseline['final']['native_cut'] if control == 'single_greedy' else baseline['winner']['native_cut']
                    pairs.append((fem['final']['native_cut'], baseline_cut))
            report['comparisons'][control] = {
                'paired_successes': len(pairs), 'fem_wins': sum(a < b for a, b in pairs),
                'ties': sum(a == b for a, b in pairs), 'fem_losses': sum(a > b for a, b in pairs),
                'mean_fem_cut_paired_only': float(np.mean([a for a, b in pairs])) if pairs else None,
                'mean_control_cut_paired_only': float(np.mean([b for a, b in pairs])) if pairs else None}
        report['restart_no_result'] = sum(r.get('greedy_restarts', {}).get('status') == 'no_result' for r in rows)
        report['coarsening_failures'] = sum(r.get('status') == 'coarsening_failed' for r in rows)
        result.append(report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(30, 40)))
    parser.add_argument('--ibm-directory', type=Path, default=Path('/private/tmp/ising-hgr.K5jm2Y'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('seeds must be unique nonnegative integers')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    before = source_hashes()
    datasets = []
    for name in ('ibm01', 'ibm02'):
        path = args.ibm_directory / (name + '.hgr')
        sha = digest(path)
        n, edges = read_unweighted_hgr(path)
        if digest(path) != sha:
            raise RuntimeError('input file changed during loading')
        datasets.append({'name': name, 'path': path.resolve(), 'sha256': sha, 'n': n, 'edges': edges})
    git = lambda cwd, *parts: subprocess.check_output(['git', *parts], cwd=cwd, text=True).strip()
    manifest = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'invocation': shlex.join([sys.executable, *sys.argv]),
        'working_directory': str(Path.cwd()), 'seeds': args.seeds, 'source_sha256_before': before,
        'inputs': [{k: str(v) if isinstance(v, Path) else v for k, v in data.items() if k != 'edges'}
                   | {'edge_count': len(data['edges'])} for data in datasets],
        'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
            'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'platform': platform.platform(),
            'thread_environment': {key: os.environ.get(key)
                for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'git_head': git(ROOT, 'rev-parse', 'HEAD'), 'git_status': git(ROOT, 'status', '--short'),
        'submodule_head': git(ROOT / 'lib/qubo-solver', 'rev-parse', 'HEAD'),
        'submodule_status': git(ROOT / 'lib/qubo-solver', 'status', '--short'),
        'config': {'q': 4, 'epsilon_upper_capacity': .03, 'coarsen_to': 200, 'coarsener': 'hem',
            'enforce_balance_cap': True, 'flow_passes_per_refinement_call': 2,
            'fem_method': 'fem', 'fem_map_type': 'native', 'fem_trials': 8, 'fem_steps': 150,
            'fem_optimizer': 'adam', 'fem_anneal': 'exp', 'fem_betamin': .5, 'fem_betamax': 50.,
            'fem_learning_rate': .08, 'fem_h_factor': .1, 'fem_imbalance_weight': 5.,
            'fem_dtype': 'float64', 'fem_device': 'cpu', 'balance_search_budget': 200000,
            'order': 'greedy,fem when (instance_index+seed_index) is even; fem,greedy otherwise',
            'restart_seed': '1000000 + 10000 * hierarchy_seed + attempt_index',
            'max_restart_attempts': 10000},
        'timing': 'One separate deterministic FEM+greedy warmup runs before the measured dataset loop and is excluded from all benchmark budgets and candidates. Downstream timer includes initializer construction, init, refiner construction, and V-cycle; shared coarsening is separate. External validation and serialization are excluded uniformly. Restart completion uses wall time from restart-loop start, with no external checks or writes between runs.',
        'limits': ['No KaHyPar comparison or universal performance claim.',
                   'A restart that first finishes after its deadline cannot win; no_result is retained.',
                   'Restart overshoots are measured and retained but not charged as usable in-budget solutions.',
                   'CPU timings can depend on thermal state and external workloads; run without other timed workloads.',
                   'Paired means exclude failures and no_result cases; counts and raw failures are retained.'],
    }
    manifest['warmup'] = deterministic_warmup()
    save(args.output / 'manifest.json', manifest)
    if manifest['warmup']['status'] != 'success':
        raise SystemExit('Deterministic warmup failed; see manifest.json. No IBM pipeline was run.')
    print(f"Warmup excluded: {manifest['warmup']['total_wall_seconds_including_verification']:.3f}s; FEM and greedy checks passed.", flush=True)
    units = []
    for instance_index, data in enumerate(datasets):
        edges, n = data['edges'], data['n']
        nodes, weights = np.ones(n), np.ones(len(edges))
        for seed_index, seed in enumerate(args.seeds):
            name = f"{data['name']}-seed{seed}"
            directory = args.output / name
            directory.mkdir()
            unit = {'instance': data['name'], 'seed': seed, 'detail_file': name + '/runs.json', 'single_runs': {}}
            start = time.perf_counter()
            try:
                coarse = KahyparLikeSolver().coarsen(edges, n, 4, coarsen_to=200, seed=seed,
                    score_mode='hem', enforce_balance_cap=True, epsilon=.03,
                    node_weights=nodes, hyperedge_weights=weights)
                unit['coarsening'] = {'seconds': time.perf_counter()-start,
                    'nodes': len(coarse['coarse_node_weights']), 'levels': len(coarse['hierarchy_stack'])}
            except Exception as exc:
                unit.update(status='coarsening_failed', coarsening_seconds=time.perf_counter()-start,
                            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                save(directory / 'runs.json', unit)
                units.append(unit)
                save(args.output / 'progress.json', {'units': units, 'aggregates': aggregates(units)})
                print(f'{name}: COARSENING FAILED: {exc}', flush=True)
                continue
            order = ('greedy', 'fem') if (instance_index+seed_index) % 2 == 0 else ('fem', 'greedy')
            unit['single_run_order'] = order
            for method in order:
                run = run_pipeline(method, seed, coarse, edges, nodes, weights)
                run['run_id'] = 'single_' + method
                verify_run(run, coarse, edges, nodes, weights)
                unit['single_runs'][method] = run
            # Save both singles before the restart loop; this I/O is outside
            # every timed interval and cannot consume its budget.
            persist_unit(directory, unit, coarse)
            fem = unit['single_runs']['fem']
            if fem['status'] == 'success':
                def restart(attempt):
                    run = run_pipeline('greedy', 1000000 + 10000*seed + attempt,
                                       coarse, edges, nodes, weights)
                    run['run_id'] = f'restart_{attempt:04d}'
                    return run
                restarts = collect_restarts(restart, fem['pipeline_seconds'])
                for run in restarts['records']:
                    verify_run(run, coarse, edges, nodes, weights)
                choose_restart_winner(restarts)
                unit['greedy_restarts'] = restarts
            else:
                unit['greedy_restarts'] = {'status': 'not_run', 'reason': 'FEM pipeline or independent validation failed; no valid full-pipeline reference budget'}
            unit['status'] = 'completed'
            persist_unit(directory, unit, coarse)
            compact = compact_unit(unit)
            units.append(compact)
            save(args.output / 'progress.json', {'units': units, 'aggregates': aggregates(units)})
            singles = {m: r.get('final', {}).get('native_cut', r['status']) for m, r in unit['single_runs'].items()}
            r = unit['greedy_restarts']
            print(f"{name} order={','.join(order)} single={singles} restarts={r.get('winner', {}).get('native_cut') if r.get('winner') else r['status']} completed={r.get('eligible_completed_runs', 0)} overshoot={r.get('overshoot_seconds', 0):.3f}s", flush=True)
    after = source_hashes()
    input_hashes_after = {data['name']: digest(data['path']) for data in datasets}
    summary = {'manifest': 'manifest.json', 'units': units, 'aggregates': aggregates(units),
        'source_sha256_after': after, 'sources_unchanged_during_run': before == after,
        'input_sha256_after': input_hashes_after,
        'inputs_unchanged_during_run': all(input_hashes_after[data['name']] == data['sha256'] for data in datasets),
        'finished_utc': datetime.now(timezone.utc).isoformat()}
    save(args.output / 'summary.json', summary)
    print(f"Saved {args.output / 'summary.json'}", flush=True)
    if not summary['sources_unchanged_during_run'] or not summary['inputs_unchanged_during_run']:
        raise SystemExit('Sources or inputs changed during the experiment; inspect hashes before interpreting results.')


if __name__ == '__main__':
    main()
