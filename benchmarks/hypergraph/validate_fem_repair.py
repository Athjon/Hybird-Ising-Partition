"""Exercise repaired production FEM and weighted flow on saved and real inputs.

Uses FemCoarsenSolver, never the independent diagnostic optimizer. Records
failures as failures and does not change inputs, capacities, or hyperparameters
to rescue a run. Results are not a KaHyPar comparison or a scaling claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
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

from src.hyper_solver import (
    FemCoarsenSolver, HyperRefineSolver, KahyparLikeSolver, vcycle_uncoarsen,
)
from src.partition.hyper_objective import capacity_limits
from src.partition.hyper_quotient import connectivity_cost
from benchmarks.hypergraph.time_budget_hgr import read_unweighted_hgr


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def sources():
    selected = [Path(__file__).resolve(), ROOT / 'src/hyper_solver.py',
        ROOT / 'src/partition/hyper_objective.py', ROOT / 'src/partition/hyper_utils.py',
        ROOT / 'src/partition/hyper_refine_contract.py', ROOT / 'src/partition/hyper_quotient.py',
        ROOT / 'benchmarks/hypergraph/time_budget_hgr.py']
    selected += sorted((ROOT / 'lib/qubo-solver/src/fem').glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in selected}


def measure(assignment, edges, nodes, weights, q, epsilon):
    assignment = np.asarray(assignment, dtype=np.int64)
    nodes, weights = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    assert assignment.shape == nodes.shape
    assert np.all((assignment >= 0) & (assignment < q))
    native = float(connectivity_cost(assignment, edges, weights))
    independent = sum(float(w) * max(0, len({int(assignment[v]) for v in edge}) - 1)
                      for edge, w in zip(edges, weights))
    assert np.isclose(native, independent, rtol=1e-12, atol=1e-12)
    capacity, tolerance = capacity_limits(nodes, q, epsilon)
    loads = np.bincount(assignment, weights=nodes, minlength=q)
    return {'native_cut': native, 'block_loads': loads, 'capacity': capacity,
            'capacity_tolerance': tolerance, 'feasible': bool(np.all(loads <= capacity + tolerance)),
            'independent_native_cut_verified': True}


class RecordingRefiner(HyperRefineSolver):
    def __init__(self, q, epsilon, passes=2):
        super().__init__()
        self.q, self.epsilon = q, epsilon
        self.records = []
        self.update_params(mode_cycle=('flow',), flow_passes=passes,
                           max_imbalance=epsilon, repair_balance=True)

    def refine(self, assignment, edges, q, node_weights=None, hyperedge_weights=None, **options):
        before = measure(assignment, edges, node_weights, hyperedge_weights, q, self.epsilon)
        record = {'call': len(self.records), 'nodes': len(assignment), 'edges': len(edges), 'before': before}
        self.records.append(record)
        start = time.perf_counter()
        try:
            result = super().refine(assignment, edges, q, node_weights=node_weights,
                                    hyperedge_weights=hyperedge_weights, **options)
            record['after'] = measure(result, edges, node_weights, hyperedge_weights, q, self.epsilon)
            assert record['after']['feasible']
            if before['feasible']:
                assert record['after']['native_cut'] <= before['native_cut'] + 1e-8
            return result
        except Exception as exc:
            record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            record['seconds'] = time.perf_counter() - start


def enumerate_bisection(edges, n):
    # Fix vertex zero's label to remove the global label-flip duplicate.
    choices = list(itertools.combinations(range(1, n), n // 2))
    states = np.zeros((len(choices), n), dtype=np.int8)
    for i, indices in enumerate(choices):
        states[i, list(indices)] = 1
    values = np.zeros(len(states))
    for edge in edges:
        counts = states[:, edge].sum(-1)
        values += (counts > 0) & (counts < len(edge))
    return float(values.min())


def production_small(path, output):
    instances = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(instances) == 40 and {x['n'] for x in instances} == {12, 16}
    rows, aliases = [], []
    seen_groups = set()
    for item in instances:
        edges, n, seed = item['edges'], item['n'], item['seed']
        optimum = enumerate_bisection(edges, n)
        assert optimum == item['optimum']
        solver = FemCoarsenSolver()
        start = time.perf_counter()
        record = {'n': n, 'family': item['family'], 'seed': seed, 'exact_optimum': optimum}
        try:
            assignment = solver.initial_partition(edges, np.ones(n), 2,
                num_trials=32, num_steps=300, seed=seed, epsilon=0, dev='cpu',
                hyperedge_weights=np.ones(len(edges)))
            record.update(measure(assignment, edges, np.ones(n), np.ones(len(edges)), 2, 0))
            record.update(returned_native_gap=record['native_cut'] - optimum,
                          selected_trial=solver.last_result['selected_trial'],
                          assignment=assignment, solver_details=solver.last_result,
                          status='success')
            assert record['feasible'] and record['returned_native_gap'] >= 0
            group = (n, item['family'])
            if group not in seen_groups:
                alias = FemCoarsenSolver().initial_partition(edges, np.ones(n), 2,
                    num_trials=32, num_steps=300, seed=seed, epsilon=0, dev='cpu',
                    hyperedge_weights=np.ones(len(edges)), method='pubo')
                identical = bool(np.array_equal(alias, assignment))
                aliases.append({'n': n, 'family': item['family'], 'seed': seed, 'identical_assignment': identical})
                assert identical
                seen_groups.add(group)
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        record['seconds_including_optional_alias_check'] = time.perf_counter() - start
        rows.append(record)
        save(output / 'saved_instances.json', {'input': str(path), 'sha256': digest(path), 'rows': rows, 'alias_checks': aliases})
        print(f"saved n={n} {item['family']} seed={seed} status={record['status']} gap={record.get('returned_native_gap')}", flush=True)
    aggregate = []
    for n, family in sorted({(x['n'], x['family']) for x in instances}):
        selected = [r for r in rows if r['n'] == n and r['family'] == family]
        succeeded = [r for r in selected if r['status'] == 'success']
        aggregate.append({'n': n, 'family': family, 'instances': len(selected), 'successes': len(succeeded),
            'feasible': sum(r['feasible'] for r in succeeded),
            'exact_hits': sum(r['returned_native_gap'] == 0 for r in succeeded),
            'mean_gap_successes_only': float(np.mean([r['returned_native_gap'] for r in succeeded])) if succeeded else None})
    return {'aggregate': aggregate, 'alias_checks': aliases, 'detail_file': 'saved_instances.json'}


def pipeline(name, edges, nodes, weights, q, seed, epsilon, target, trials, steps, output, global_guard=False):
    coarsener = KahyparLikeSolver()
    start = time.perf_counter()
    result = coarsener.coarsen(edges, len(nodes), q, coarsen_to=target, seed=seed,
        score_mode='hem', enforce_balance_cap=True, epsilon=epsilon,
        enforce_global_feasibility=global_guard, node_weights=nodes, hyperedge_weights=weights)
    coarse_seconds = time.perf_counter() - start
    coarse_nodes = result['coarse_node_weights'].numpy()
    coarse_edges = result['coarse_hyperedges']
    coarse_weights = np.asarray(result['coarse_hyperedge_weights'])
    record = {'name': name, 'n': len(nodes), 'edges': len(edges), 'q': q, 'seed': seed,
        'epsilon': epsilon, 'coarsen_to': target, 'num_trials': trials, 'num_steps': steps,
        'coarsening': {'seconds': coarse_seconds, 'nodes': len(coarse_nodes), 'edges': len(coarse_edges),
                       'levels': len(result['hierarchy_stack']), 'exact_global_guard': global_guard},
        'methods': {}}
    arrays = {'original_to_coarse': result['original_to_coarse'], 'coarse_node_weights': coarse_nodes,
              'original_node_weights': nodes, 'original_hyperedge_weights': weights}
    print(f"{name}: coarsened {len(nodes)}->{len(coarse_nodes)} in {coarse_seconds:.3f}s", flush=True)
    for method in ('greedy', 'fem'):
        stage = 'initial_partition'
        method_record = {}
        record['methods'][method] = method_record
        start = time.perf_counter()
        try:
            if method == 'greedy':
                assignment = coarsener.initial_partition_greedy(coarse_edges, result['coarse_node_weights'], q,
                    seed=seed, epsilon=epsilon, hyperedge_weights=coarse_weights)
            else:
                solver = FemCoarsenSolver()
                assignment = solver.initial_partition(coarse_edges, coarse_nodes, q,
                    seed=seed, epsilon=epsilon, hyperedge_weights=coarse_weights,
                    num_trials=trials, num_steps=steps, dev='cpu')
                method_record['solver_details'] = solver.last_result
            method_record['initial_partition_seconds'] = time.perf_counter() - start
            method_record['coarse_initial'] = measure(assignment, coarse_edges, coarse_nodes, coarse_weights, q, epsilon)
            fine = assignment[result['original_to_coarse']]
            method_record['lifted_initial'] = measure(fine, edges, nodes, weights, q, epsilon)
            assert method_record['coarse_initial']['native_cut'] == method_record['lifted_initial']['native_cut']
            np.testing.assert_allclose(method_record['coarse_initial']['block_loads'], method_record['lifted_initial']['block_loads'])
            arrays[method + '_coarse_assignment'] = assignment
            stage = 'vcycle'
            refiner = RecordingRefiner(q, epsilon, passes=2)
            method_record['refinement_stages'] = refiner.records
            start = time.perf_counter()
            final = vcycle_uncoarsen(assignment, result['hierarchy_stack'], edges, q, refiner,
                verbose=False, node_weights=nodes, hyperedge_weights=weights)
            method_record['vcycle_seconds'] = time.perf_counter() - start
            method_record['final'] = measure(final, edges, nodes, weights, q, epsilon)
            assert method_record['final']['feasible']
            arrays[method + '_final_assignment'] = final
            method_record['status'] = 'success'
            print(f"{name} {method}: initial={method_record['coarse_initial']['native_cut']} final={method_record['final']['native_cut']} loads={method_record['final']['block_loads'].tolist()}", flush=True)
        except Exception as exc:
            method_record.update(status='failed', failed_stage=stage, failed_stage_seconds=time.perf_counter()-start,
                error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            print(f"{name} {method} FAILED {stage}: {type(exc).__name__}: {exc}", flush=True)
        save(output / (name + '.json'), record)
        np.savez_compressed(output / (name + '-assignments.npz'), **arrays)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/hypergraph/results/fem-repair-20260924')
    parser.add_argument('--saved-instances', type=Path, default=ROOT / 'benchmarks/hypergraph/results/ogp-diagnostic-20260924/objective/instances.jsonl')
    parser.add_argument('--ibm-directory', type=Path, default=Path('/private/tmp/ising-hgr.K5jm2Y'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        parser.error('output already has a summary; choose a new directory')
    torch.set_num_threads(1)
    before = sources()
    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'invocation': shlex.join([sys.executable, *sys.argv]),
        'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
            'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'platform': platform.platform()},
        'source_sha256_before': before,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'submodule_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'lib/qubo-solver', text=True).strip(),
        'saved_input_sha256': digest(args.saved_instances),
        'scope': 'Current production native FemCoarsenSolver; no independent optimizer, no KaHyPar comparison, no universal advantage claim.',
        'effective_fem_defaults': {'method': 'fem', 'map_type': 'native', 'optimizer': 'adam',
            'learning_rate': 0.08, 'betamin': 0.5, 'betamax': 50., 'anneal': 'exp', 'dtype': 'torch.float64',
            'h_factor': .1, 'imbalance_weight': 5., 'exact_balance_max_nodes': 20, 'balance_search_budget': 200000},
    }
    save(args.output / 'environment.json', report)
    report['saved_instances'] = production_small(args.saved_instances, args.output)
    report['weighted_small'] = []
    for n, q in itertools.product((8, 12), (2, 3)):
        seed = 41000 + 100 * n + q
        rng = np.random.default_rng(seed)
        edges = [sorted(rng.choice(n, size=int(rng.integers(2, 5)), replace=False).tolist()) for _ in range(2*n)]
        nodes = np.tile([1., 2.], n // 2)
        weights = rng.integers(1, 8, size=len(edges)).astype(float) / 2
        name = f'weighted-n{n:02d}-q{q}'
        save(args.output / (name + '-input.json'), {'edges': edges, 'node_weights': nodes, 'hyperedge_weights': weights, 'seed': seed})
        result = pipeline(name, edges, nodes, weights, q, seed, 0., max(q+1, n//2), 32, 300, args.output, True)
        result['input_sha256'] = digest(args.output / (name + '-input.json'))
        report['weighted_small'].append(result)
    report['real_instances'] = []
    for name in ('ibm01', 'ibm02'):
        path = args.ibm_directory / (name + '.hgr')
        n, edges = read_unweighted_hgr(path)
        result = pipeline(name, edges, np.ones(n), np.ones(len(edges)), 4, 30, .03, 200, 8, 150, args.output)
        result.update(input_path=str(path), input_sha256=digest(path))
        report['real_instances'].append(result)
        save(args.output / 'progress.json', report)
    report['source_sha256_after'] = sources()
    report['sources_unchanged_during_run'] = before == report['source_sha256_after']
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(args.output / 'summary.json', report)
    print('sources_unchanged_during_run=' + str(report['sources_unchanged_during_run']), flush=True)
    print(f"Saved {args.output / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
