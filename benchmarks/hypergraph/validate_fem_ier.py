"""Validate native FEM-IER from saved starts, with selector and time controls.

No benchmark should run until the production FEM-IER interface is ready.
IBM arms start independently from saved flow-refined FEM final assignments.
FEM-IER and random selection share the first-round start/seed/pool; their later
states may diverge. Random uses a fixed selector count, not an equal-time claim.
Only the additional-flow arm has a strict measured wall-time budget.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
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

from src.hyper_solver import HyperRefineSolver, KahyparLikeSolver, vcycle_uncoarsen
from benchmarks.hypergraph.compare_fem_multiseed import (
    choose_restart_winner, collect_restarts, deterministic_warmup, digest, measure, plain, save,
)
from benchmarks.hypergraph.native_search_probe import exact_states
from benchmarks.hypergraph.time_budget_hgr import read_unweighted_hgr


def ier_options(rounds, seed, backend, epsilon):
    return dict(mode_cycle=('fem_ier',), ier_backend=backend,
        ier_rounds=rounds, ier_max_moves=24, ier_boundary_pool=96,
        ier_pool_strategy='local',
        ier_num_trials=8, ier_num_steps=100, ier_random_samples=800,
        seed=seed, max_imbalance=epsilon, repair_balance=True)


def sources():
    paths = [Path(__file__).resolve(), ROOT / 'src/hyper_solver.py',
             ROOT / 'benchmarks/hypergraph/compare_fem_multiseed.py',
             ROOT / 'benchmarks/hypergraph/native_search_probe.py',
             ROOT / 'benchmarks/hypergraph/native_mcmc.py',
             ROOT / 'benchmarks/hypergraph/time_budget_hgr.py']
    paths += sorted((ROOT / 'src/partition').rglob('*.py'))
    paths += sorted((ROOT / 'lib/qubo-solver/src/fem').rglob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def run_refinement(arm, initial, edges, nodes, weights, q, epsilon, seed, rounds,
                   *, clock=time.perf_counter):
    """Include candidate building, optimization, and native acceptance in timing."""
    initial = np.asarray(initial, dtype=np.int64).copy()
    record = {'arm': arm, 'solver_seed': seed, 'initial_assignment': initial.copy()}
    start = clock()
    try:
        refiner = HyperRefineSolver()
        if arm == 'flow':
            refiner.update_params(mode_cycle=('flow',), flow_passes=2,
                max_imbalance=epsilon, repair_balance=True, seed=seed)
        elif arm in ('fem_ier', 'random'):
            refiner.update_params(**ier_options(rounds, seed, 'random' if arm == 'random' else 'fem', epsilon))
        else:
            raise ValueError('unknown refinement arm')
        final = refiner.refine(initial, edges, q, node_weights=nodes, hyperedge_weights=weights)
        finish = clock()
        record.update(status='success', final_assignment=np.asarray(final, dtype=np.int64).copy())
        # Copying diagnostics and external verification are excluded equally
        # from all arms. Production candidate construction/acceptance is not.
        record['last_result'] = copy.deepcopy(getattr(refiner, 'last_result', None))
    except Exception as exc:
        finish = clock()
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    record.update(started_at=start, finished_at=finish, pipeline_seconds=finish-start)
    return record


def verify_refinement(record, edges, nodes, weights, q, epsilon):
    try:
        record['initial'] = measure(record['initial_assignment'], edges, nodes, weights, q, epsilon)
        if not record['initial']['feasible']:
            raise AssertionError('saved/continuation start is infeasible')
        if record['status'] == 'success':
            record['final'] = measure(record['final_assignment'], edges, nodes, weights, q, epsilon)
            if not record['final']['feasible']:
                raise AssertionError('returned refinement is infeasible')
            if record['final']['native_cut'] > record['initial']['native_cut'] + 1e-10:
                raise AssertionError('refinement increased the native objective')
            if record['arm'] in ('fem_ier', 'random') and not isinstance(record.get('last_result'), dict):
                raise AssertionError('production FEM-IER did not expose last_result diagnostics')
    except Exception as exc:
        record.update(execution_status_before_validation=record['status'], status='validation_failed',
                      validation_error_type=type(exc).__name__, validation_error=str(exc),
                      validation_traceback=traceback.format_exc())


def additional_flow(initial, edges, nodes, weights, q, epsilon, seed, rounds, budget,
                    *, clock=time.perf_counter):
    """Sequential extra flow, from this arm's own saved start, never A/B output."""
    current = np.asarray(initial, dtype=np.int64).copy()

    def once(attempt):
        nonlocal current
        record = run_refinement('flow', current, edges, nodes, weights, q, epsilon,
                                seed+attempt, rounds, clock=clock)
        record['run_id'] = f'flow_{attempt:04d}'
        if record['status'] == 'success':
            current = record['final_assignment'].copy()
        return record

    result = collect_restarts(once, budget, clock=clock)
    # No independent checks or I/O between timed attempts. Validate all states
    # afterward, then admit only completed and verified candidates.
    for record in result['records']:
        verify_refinement(record, edges, nodes, weights, q, epsilon)
    choose_restart_winner(result)
    result['continuation_policy'] = 'Start from the saved initial partition; each attempt continues only its own previous flow state.'
    return result


def persist(directory, case):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def extract(value, path=()):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                location = path + (str(key),)
                if key in ('initial_assignment', 'final_assignment', 'coarse_assignment') and isinstance(item, np.ndarray):
                    array_key = '__'.join(location)
                    arrays[array_key] = item
                    result[key] = {'artifact': 'assignments.npz', 'key': array_key}
                else:
                    result[key] = extract(item, location)
            return result
        if isinstance(value, list):
            return [extract(item, path+(str(index),)) for index, item in enumerate(value)]
        return value

    serialized = extract(case)
    np.savez_compressed(directory / 'assignments.npz', **arrays)
    save(directory / 'result.json', serialized)


def compact_arm(run):
    return {key: run[key] for key in ('status', 'pipeline_seconds', 'initial', 'final', 'error', 'validation_error') if key in run}


def saved_ibm_case(baseline, name, seed):
    path = baseline / f'{name}-seed{seed}' / 'runs.json'
    data = json.loads(path.read_text())
    if data['single_runs']['fem']['status'] != 'success':
        raise ValueError('saved FEM baseline was not successful')
    reference = data['single_runs']['fem']['assignment_artifacts']
    archive = path.parent / reference['final_assignment']['file']
    with np.load(archive) as arrays:
        final = arrays[reference['final_assignment']['key']].copy()
        coarse = arrays[reference['coarse_assignment']['key']].copy()
        mapping = arrays['original_to_coarse'].copy()
        coarse_nodes = arrays['coarse_node_weights'].copy()
    return {'initial_assignment': final, 'coarse_assignment': coarse, 'mapping': mapping,
            'coarse_nodes': coarse_nodes, 'recorded_final': data['single_runs']['fem']['final'],
            'input_hashes': {str(path.resolve()): digest(path), str(archive.resolve()): digest(archive)}}


def check_hierarchy(coarse, saved):
    np.testing.assert_array_equal(coarse['original_to_coarse'], saved['mapping'],
                                  err_msg='reconstructed hierarchy differs from the saved assignment map')
    np.testing.assert_array_equal(np.asarray(coarse['coarse_node_weights']), saved['coarse_nodes'])
    if len(saved['coarse_assignment']) != len(coarse['coarse_node_weights']):
        raise AssertionError('saved coarse assignment size does not match reconstructed hierarchy')


class IntegrationRecorder(HyperRefineSolver):
    """Instrument each V-cycle call; integration timings are not comparisons."""
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon, self.stages = epsilon, []

    def refine(self, assignment, edges, q, node_weights=None, hyperedge_weights=None, **kwargs):
        before = measure(assignment, edges, node_weights, hyperedge_weights, q, self.epsilon)
        final = super().refine(assignment, edges, q, node_weights=node_weights,
                               hyperedge_weights=hyperedge_weights, **kwargs)
        after = measure(final, edges, node_weights, hyperedge_weights, q, self.epsilon)
        if not after['feasible'] or after['native_cut'] > before['native_cut'] + 1e-10:
            raise AssertionError('V-cycle stage violated feasibility or native monotonicity')
        self.stages.append({'nodes': len(assignment), 'before': before, 'after': after,
                            'last_result': copy.deepcopy(getattr(self, 'last_result', None))})
        return final


def integration_case(name, edges, nodes, weights, saved, rounds):
    seed, q, epsilon = 30, 4, .03
    result = {'instance': name, 'seed': seed, 'comparison': 'Integration sanity only; not equal time.',
              'input_hashes': saved['input_hashes'], 'coarse_assignment': saved['coarse_assignment'], 'arms': {}}
    coarse = KahyparLikeSolver().coarsen(edges, len(nodes), q, coarsen_to=200, seed=seed,
        score_mode='hem', enforce_balance_cap=True, epsilon=epsilon, node_weights=nodes, hyperedge_weights=weights)
    check_hierarchy(coarse, saved)
    result['reconstructed_hierarchy_matches_saved'] = True
    result['initial_assignment'] = saved['coarse_assignment'][coarse['original_to_coarse']]
    result['initial'] = measure(result['initial_assignment'], edges, nodes, weights, q, epsilon)
    for name_arm, modes in (('flow', ('flow',)), ('fem_ier_flow', ('fem_ier', 'flow'))):
        refiner = IntegrationRecorder(epsilon)
        options = ier_options(rounds, seed, 'fem', epsilon)
        options.update(mode_cycle=modes, flow_passes=2)
        refiner.update_params(**options)
        start = time.perf_counter()
        record = {'mode_cycle': modes, 'stages': refiner.stages}
        result['arms'][name_arm] = record
        try:
            final = vcycle_uncoarsen(saved['coarse_assignment'].copy(), coarse['hierarchy_stack'], edges, q,
                refiner, verbose=False, node_weights=nodes, hyperedge_weights=weights)
            record.update(status='success', final_assignment=final,
                          final=measure(final, edges, nodes, weights, q, epsilon))
            if not record['final']['feasible']:
                raise AssertionError('integration output infeasible')
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        record['wall_seconds_with_instrumentation_not_a_timing_comparison'] = time.perf_counter()-start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(30, 35)))
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'benchmarks/hypergraph/results/fem-multiseed-20260924-v2')
    parser.add_argument('--repair', type=Path, default=ROOT / 'benchmarks/hypergraph/results/fem-repair-20260924')
    parser.add_argument('--ibm-directory', type=Path, default=Path('/private/tmp/ising-hgr.K5jm2Y'))
    args = parser.parse_args()
    if args.rounds < 1 or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('rounds must be positive and seeds unique/nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    before = sources()
    manifest_path = args.baseline / 'manifest.json'
    baseline_manifest = json.loads(manifest_path.read_text())
    input_hashes = {str(manifest_path.resolve()): digest(manifest_path)}
    datasets = []
    for name in ('ibm01', 'ibm02'):
        path = args.ibm_directory / (name + '.hgr')
        sha = digest(path)
        expected = next(d['sha256'] for d in baseline_manifest['inputs'] if d['name'] == name)
        if sha != expected:
            raise ValueError(f'{name} input differs from the saved baseline')
        n, edges = read_unweighted_hgr(path)
        input_hashes[str(path.resolve())] = sha
        datasets.append((name, edges, np.ones(n), np.ones(len(edges))))
    config = {'seeds': args.seeds, 'rounds': args.rounds, 'q_ibm': 4, 'epsilon_ibm': .03,
        'ier_max_moves': 24, 'ier_boundary_pool': 96, 'ier_pool_strategy': 'local',
        'ier_num_trials': 8, 'ier_num_steps': 100,
        'ier_random_samples': 800, 'random_selector_budget': '800 independent Bernoulli selectors per round, plus deterministic all-off/all-on/single-atom candidates in both backends; fixed count, not equal wall time.',
        'a_b_order': 'Alternate by instance_index+seed_index; only first-round initial state/seed/pool are shared, later states may diverge.',
        'flow_control': 'Sequential complete flow(2 passes) calls from an independent copy of saved initial; only full returns by the measured A deadline can win.',
        'integration': 'Seed30 same reconstructed hierarchy and saved coarse FEM assignment; flow versus fem_ier+flow, not equal time.',
        'weighted': 'Four saved weighted repair-final starts; epsilon0, original node/edge weights, exhaustive native feasible-state oracle.'}
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
        'invocation': shlex.join([sys.executable, *sys.argv]), 'working_directory': str(Path.cwd()),
        'config': config, 'source_sha256_before': before, 'input_sha256_before': input_hashes,
        'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
            'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'platform': platform.platform(),
            'thread_environment': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'submodule_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'lib/qubo-solver', text=True).strip(),
        'weighted': [], 'fixed_starts': [], 'integration': [],
        'limits': ['Saved states were flow-refined; further-flow controls test residual improvement rather than assuming proven FM local optimality.',
            'Random selector control is not time matched; additional flow is the only strict time control.',
            'No_result means no verified full flow call completed by the deadline, not that the initial state was infeasible.',
            'No KaHyPar comparison, asymptotic OGP conclusion, or general performance guarantee.']}
    # Exclude cold FEM/Adam startup from every later IBM A-arm time budget.
    report['core_warmup_excluded'] = deterministic_warmup()
    save(args.output / 'manifest.json', report)
    if report['core_warmup_excluded']['status'] != 'success':
        raise SystemExit('Core warmup failed; no IER benchmark was started.')

    # This non-time-matched small phase also loads/initializes the IER path
    # before the fixed-start IBM comparisons.
    for n in (8, 12):
        for q in (2, 3):
            name = f'weighted-n{n:02d}-q{q}'
            path, archive = args.repair/(name+'-input.json'), args.repair/(name+'-assignments.npz')
            data = json.loads(path.read_text())
            with np.load(archive) as arrays:
                initial = arrays['fem_final_assignment'].copy()
            edges, nodes, weights = data['edges'], np.asarray(data['node_weights']), np.asarray(data['hyperedge_weights'])
            seed = data['seed']
            hashes = {str(path.resolve()): digest(path), str(archive.resolve()): digest(archive)}
            input_hashes.update(hashes)
            states, codes, energies = exact_states(edges, nodes, weights, q, 0.)
            optimum, powers = float(energies.min()), q ** np.arange(n, dtype=np.int64)
            case = {'name': name, 'q': q, 'epsilon': 0., 'seed': seed, 'initial_assignment': initial,
                'initial': measure(initial, edges, nodes, weights, q, 0.), 'input_hashes': hashes,
                'exact_optimum': optimum, 'feasible_labeled_states': len(states), 'arms': {}}
            case['initial_exact_gap'] = case['initial']['native_cut'] - optimum
            for arm in ('fem_ier', 'random'):
                record = run_refinement(arm, initial, edges, nodes, weights, q, 0., seed, args.rounds)
                verify_refinement(record, edges, nodes, weights, q, 0.)
                if record['status'] == 'success':
                    code = int(record['final_assignment'] @ powers)
                    position = int(np.searchsorted(codes, code))
                    if position >= len(codes) or codes[position] != code or energies[position] != record['final']['native_cut']:
                        raise AssertionError('IER output disagrees with the exhaustive weighted oracle')
                    record.update(exact_gap=record['final']['native_cut']-optimum, exact_oracle_verified=True)
                case['arms'][arm] = record
            persist(args.output/'weighted'/name, case)
            report['weighted'].append({'name': name, 'initial': case['initial'], 'exact_optimum': optimum,
                'arms': {arm: compact_arm(run) | {k: run[k] for k in ('exact_gap', 'exact_oracle_verified') if k in run}
                         for arm, run in case['arms'].items()}, 'detail_file': f'weighted/{name}/result.json'})
            print(f'{name}: '+str({arm: run.get('final', {}).get('native_cut', run['status']) for arm,run in case['arms'].items()}), flush=True)

    for instance_index, (name, edges, nodes, weights) in enumerate(datasets):
        for seed_index, seed in enumerate(args.seeds):
            saved = saved_ibm_case(args.baseline, name, seed)
            initial = saved['initial_assignment']
            input_hashes.update(saved['input_hashes'])
            case = {'instance': name, 'seed': seed, 'initial_assignment': initial,
                'initial': measure(initial, edges, nodes, weights, 4, .03), 'input_hashes': saved['input_hashes'], 'arms': {}}
            if case['initial']['native_cut'] != saved['recorded_final']['native_cut']:
                raise AssertionError('saved baseline cost does not match its actual assignment')
            order = ('fem_ier', 'random') if (instance_index+seed_index) % 2 == 0 else ('random', 'fem_ier')
            case['arm_order'] = order
            for arm in order:
                record = run_refinement(arm, initial, edges, nodes, weights, 4, .03, seed, args.rounds)
                verify_refinement(record, edges, nodes, weights, 4, .03)
                case['arms'][arm] = record
            a = case['arms']['fem_ier']
            if a['status'] == 'success':
                case['additional_flow'] = additional_flow(initial, edges, nodes, weights, 4, .03,
                    seed, args.rounds, a['pipeline_seconds'])
            else:
                case['additional_flow'] = {'status': 'not_run', 'reason': 'FEM-IER failed; no valid reference budget'}
            directory = f'fixed/{name}-seed{seed}'
            persist(args.output/directory, case)
            report['fixed_starts'].append({'instance': name, 'seed': seed, 'initial': case['initial'],
                'arm_order': order, 'arms': {arm: compact_arm(run) for arm,run in case['arms'].items()},
                'additional_flow': {k:v for k,v in case['additional_flow'].items() if k != 'records'},
                'detail_file': directory+'/result.json'})
            save(args.output/'progress.json', report)
            print(f'{name} seed{seed}: '+str({arm: run.get('final', {}).get('native_cut', run['status']) for arm,run in case['arms'].items()})+f" flow={case['additional_flow']['status']}", flush=True)

        # Always seed30, regardless of the fixed-start seed subset requested.
        saved = saved_ibm_case(args.baseline, name, 30)
        input_hashes.update(saved['input_hashes'])
        try:
            case = integration_case(name, edges, nodes, weights, saved, args.rounds)
        except Exception as exc:
            case = {'instance': name, 'seed': 30, 'status': 'failed', 'error_type': type(exc).__name__,
                    'error': str(exc), 'traceback': traceback.format_exc()}
        directory = f'integration/{name}-seed30'
        persist(args.output/directory, case)
        report['integration'].append({'instance': name, 'seed': 30, 'detail_file': directory+'/result.json',
            'reconstructed_hierarchy_matches_saved': case.get('reconstructed_hierarchy_matches_saved', False),
            'arms': {arm: {k:v for k,v in run.items() if k not in ('final_assignment','stages')}
                     for arm, run in case.get('arms',{}).items()},
            **{k:case[k] for k in ('status','error') if k in case}})
        save(args.output/'progress.json', report)

    after = sources()
    report.update(source_sha256_after=after, sources_unchanged_during_run=before==after,
                  input_sha256_before=input_hashes,
                  input_sha256_after={path:digest(path) for path in input_hashes},
                  finished_utc=datetime.now(timezone.utc).isoformat())
    report['inputs_unchanged_during_run'] = report['input_sha256_before']==report['input_sha256_after']
    save(args.output/'summary.json', report)
    print(f"Saved {args.output/'summary.json'}", flush=True)
    if not report['sources_unchanged_during_run'] or not report['inputs_unchanged_during_run']:
        raise SystemExit('Source or input hashes changed; inspect before interpreting results.')


if __name__ == '__main__':
    main()
