"""Native-vs-clique diagnostic, exact oracles, and an independent MF reference.

This does not repair or benchmark the broken legacy FemCoarsenSolver pipeline.
All experiments use tiny synthetic, unit-weight, exactly balanced bisections.
The reference optimizer is explicit PyTorch autograd on logits; its results must
not be described as results from the repository's current FEM implementation.

Run: python benchmarks/hypergraph/ogp_objective_diagnostic.py --instances 10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # registers the external solver package
from src.hyper_solver import FemCoarsenSolver


def expected_km1(p, edges, weights=None):
    """Exact categorical product-distribution expectation; shape (..., n, k)."""
    if weights is None:
        weights = [1.0] * len(edges)
    if len(weights) != len(edges):
        raise ValueError('one weight per edge is required')
    value = p[..., 0, 0] * 0
    for edge, weight in zip(edges, weights):
        if not edge:
            raise ValueError('empty hyperedges must be removed')
        value = value + weight * ((1 - (1 - p[..., edge, :]).prod(dim=-2)).sum(dim=-1) - 1)
    return value


def make_instance(n, family, seed, planted_probability=0.85):
    rng = np.random.default_rng(seed)
    edges = []
    for _ in range(2 * n):
        pool = np.arange(n)
        if family == 'planted' and rng.random() < planted_probability:
            offset = int(rng.integers(2)) * (n // 2)
            pool = np.arange(offset, offset + n // 2)
        edges.append(sorted(rng.choice(pool, 4, replace=False).tolist()))
    # Hide the planted contiguous label ordering from deterministic rounding.
    permutation = rng.permutation(n)
    return np.sort(permutation[np.asarray(edges, dtype=np.int64)], axis=-1)


def balanced_assignments(n):
    # Label 0 at vertex 0 removes only the global label-flip symmetry.
    choices = list(itertools.combinations(range(1, n), n // 2))
    states = np.zeros((len(choices), n), dtype=np.int8)
    for i, choice in enumerate(choices):
        states[i, list(choice)] = 1
    return states


def costs(states, edges):
    counts = np.asarray(states)[..., edges].sum(axis=-1)
    native = ((counts > 0) & (counts < 4)).sum(axis=-1)
    clique = (counts * (4 - counts) / 3.0).sum(axis=-1)
    return native.astype(float), clique


def topk_round(p):
    out = np.zeros(p.shape, dtype=np.int8)
    order = np.argsort(-p, axis=-1, kind='stable')[:, :p.shape[-1] // 2]
    np.put_along_axis(out, order, 1, axis=-1)
    return out


def swap_descent(state, edges):
    """Exact best-improving balanced 1-for-1 swap, strict improvements only."""
    state = state.copy()
    current = float(costs(state[None], edges)[0][0])
    while True:
        pairs = list(itertools.product(np.flatnonzero(state == 0), np.flatnonzero(state == 1)))
        candidates = np.repeat(state[None], len(pairs), axis=0)
        for i, (u, v) in enumerate(pairs):
            candidates[i, [u, v]] = candidates[i, [v, u]]
        values = costs(candidates, edges)[0]
        best = int(np.argmin(values))
        if values[best] >= current:
            return state
        state, current = candidates[best], float(values[best])


def optimize_reference(edges, n, method, seed, trials, steps, lr, penalty):
    generator = torch.Generator().manual_seed(seed)
    logits = (0.1 * torch.randn(trials, n, generator=generator, dtype=torch.float64)).requires_grad_()
    initial = torch.sigmoid(logits).detach().numpy()
    optimizer = torch.optim.Adam([logits], lr=lr)
    edge_index = torch.as_tensor(edges, dtype=torch.long)
    start = time.perf_counter()
    for step in range(steps):
        temperature = 2.0 * (0.02 / 2.0) ** (step / max(steps - 1, 1))
        p = torch.sigmoid(logits)
        pe = p[:, edge_index]
        native = (1 - pe.prod(dim=-1) - (1 - pe).prod(dim=-1)).sum(dim=-1)
        if method == 'clique':
            energy = sum((pe[..., a] + pe[..., b] - 2 * pe[..., a] * pe[..., b]).sum(dim=-1)
                         for a, b in itertools.combinations(range(4), 2)) / 3.0
        else:
            energy = native if method == 'native' else native.detach()
        # Expected-load penalty, NOT a guarantee of sample-wise feasibility.
        balance = penalty * (p.sum(dim=-1) - n / 2) ** 2
        entropy = -(p * torch.log(p + 1e-15) + (1 - p) * torch.log(1 - p + 1e-15)).sum(dim=-1)
        loss = (energy + balance - temperature * entropy).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.sigmoid(logits).detach().numpy(), initial, time.perf_counter() - start


def formula_audit():
    torch.manual_seed(13)
    errors = []
    for k in (2, 3):
        n = 4
        edges, weights = [[0, 1, 2, 3], [0, 2], [1]], [1.7, 0.6, 2.0]
        logits = torch.randn(n, k, dtype=torch.float64, requires_grad=True)
        p = logits.softmax(-1)
        closed = expected_km1(p, edges, weights)
        exhaustive = logits.sum() * 0
        for labels in itertools.product(range(k), repeat=n):
            c = sum(w * (len({labels[v] for v in e}) - 1) for e, w in zip(edges, weights))
            probability = p[torch.arange(n), torch.tensor(labels)].prod()
            exhaustive = exhaustive + c * probability
        gc = torch.autograd.grad(closed, logits, retain_graph=True)[0]
        ge = torch.autograd.grad(exhaustive, logits)[0]
        errors.append({'k': k, 'value_error': abs(float(closed.detach() - exhaustive.detach())),
                       'gradient_max_error': float((gc - ge).abs().max())})
    q = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    gradcheck = torch.autograd.gradcheck(lambda h: expected_km1(h.softmax(-1), [[0, 1, 2, 3], [1, 2]]), (q,))
    p = q.softmax(-1)
    assignment = p.argmax(dim=-1).cpu().numpy()
    detached_cost = float(len(set(assignment)) - 1)
    return {'exact_expectation_checks': errors, 'finite_difference_gradcheck': bool(gradcheck),
            'isolated_argmax_numpy_cut_requires_grad': torch.as_tensor(detached_cost).requires_grad,
            'four_pin_native_costs': [1, 1], 'four_pin_normalized_clique_costs': [1, 4 / 3],
            'three_swaps_true_delta': 1, 'three_swaps_quadratic_truncation_delta': 0}


def runtime_audit():
    result = {'legacy_initial_partition': {}, 'scope': 'Current checkout only; no historical run reconstructed.'}
    for method in ('fem', 'pubo'):
        try:
            value = FemCoarsenSolver().initial_partition([[0, 1, 2, 3], [2, 3, 4, 5]],
                        torch.ones(6), 2, method=method, num_trials=2, num_steps=5)
            result['legacy_initial_partition'][method] = {'returned': value.tolist()}
        except Exception as exc:
            result['legacy_initial_partition'][method] = {'exception': type(exc).__name__, 'message': str(exc)}
    # Test the actual external solver with two objectives and identical seed.
    from fem import FEM
    final_probabilities = []
    traces = []
    for coefficient in (1.0, -7.0):
        def infer(_, p):
            final_probabilities.append(p.detach().clone())
            return (p[..., 1] > 0.5).to(torch.int64)
        case = FEM.from_couplings('customize', 6, 0, torch.zeros(6, 6),
                    customize_expected_func=lambda _, p, c=coefficient: c * p[..., 1].sum(-1),
                    customize_infer_func=infer)
        case.set_up_solver(4, 12, dev='cpu', q=2, manual_grad=False, seed=17, anneal='lin')
        initial = case.solver.p.clone()
        configs, values = case.solve()
        repeated_softmax = initial.clone()
        for _ in range(12):
            repeated_softmax = repeated_softmax.softmax(-1)
        traces.append({'coefficient': coefficient, 'configs': configs.tolist(), 'values': values.tolist(),
                       'max_error_vs_repeated_softmax': float((repeated_softmax - final_probabilities[-1]).abs().max())})
    difference = float((final_probabilities[0] - final_probabilities[1]).abs().max())
    matches_softmax_only = difference == 0 and all(x['max_error_vs_repeated_softmax'] == 0 for x in traces)
    result['external_manual_grad_false'] = {'probes': traces,
        'max_probability_difference_between_objectives': difference,
        'matches_objective_independent_repeated_softmax': matches_softmax_only,
        'interpretation': ('The probe matches repeated softmax without an objective-dependent update.'
                           if matches_softmax_only else 'Behavior changed; inspect the recorded traces and current solver source.')}
    return result


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / 'src/hyper_solver.py', ROOT / 'lib/qubo-solver/src/fem/solver_fem.py']
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    checks = formula_audit()
    assert checks['finite_difference_gradcheck']
    assert max(x['value_error'] for x in checks['exact_expectation_checks']) < 1e-12
    assert max(x['gradient_max_error'] for x in checks['exact_expectation_checks']) < 1e-12
    audit = runtime_audit()
    rows, exact_rows = [], []
    instances_file = (out / 'instances.jsonl').open('w')
    pool_file = (out / 'solution_pool.jsonl').open('w')
    for n in args.sizes:
        states = balanced_assignments(n)
        for family in ('random', 'planted'):
            for instance in range(args.instances):
                seed = args.seed + instance + n * 1000 + (100000 if family == 'planted' else 0)
                edges = make_instance(n, family, seed)
                native, clique = costs(states, edges)
                optimum = float(native.min())
                clique_best = np.isclose(clique, clique.min(), rtol=0, atol=1e-10)
                exact_row = {'n': n, 'family': family, 'seed': seed, 'optimum': optimum,
                    'clique_optimum': float(clique.min()), 'clique_minimizer_count': int(clique_best.sum()),
                    'clique_minimizer_best_native_gap': float(native[clique_best].min() - optimum),
                    'clique_minimizer_mean_native_gap': float(native[clique_best].mean() - optimum),
                    'clique_minimizer_worst_native_gap': float(native[clique_best].max() - optimum)}
                exact_rows.append(exact_row)
                instances_file.write(json.dumps({**exact_row, 'edges': edges.tolist()}) + '\n')
                for method in ('native', 'clique', 'detached_cut'):
                    p, initial, elapsed = optimize_reference(edges, n, method, seed, args.trials,
                        args.steps, args.learning_rate, args.balance_penalty)
                    raw = (p >= 0.5).astype(np.int8)
                    rounded = topk_round(p)
                    refined = np.stack([swap_descent(x, edges) for x in rounded])
                    raw_cuts = costs(raw, edges)[0]
                    rounded_cuts = costs(rounded, edges)[0]
                    refined_cuts = costs(refined, edges)[0]
                    feasible = raw.sum(-1) == n // 2
                    row = {'n': n, 'family': family, 'seed': seed, 'method': method, 'optimum': optimum,
                        'raw_feasible_fraction': float(feasible.mean()),
                        'raw_best_feasible_gap': float(raw_cuts[feasible].min() - optimum) if feasible.any() else None,
                        'rounded_best_gap': float(rounded_cuts.min() - optimum),
                        'rounded_mean_gap': float(rounded_cuts.mean() - optimum),
                        'rounded_optimal_fraction': float((rounded_cuts == optimum).mean()),
                        'refined_best_gap': float(refined_cuts.min() - optimum),
                        'refined_mean_gap': float(refined_cuts.mean() - optimum),
                        'random_topk_best_gap': float(costs(topk_round(initial), edges)[0].min() - optimum),
                        'optimize_seconds': elapsed}
                    rows.append(row)
                    pool_file.write(json.dumps({'n': n, 'family': family, 'seed': seed, 'method': method,
                        'probabilities': p.tolist(), 'raw': raw.tolist(), 'rounded': rounded.tolist(),
                        'refined': refined.tolist(), 'raw_native_cut': raw_cuts.tolist(),
                        'rounded_native_cut': rounded_cuts.tolist(), 'refined_native_cut': refined_cuts.tolist()}) + '\n')
            print(f'completed n={n} family={family} instances={args.instances}', flush=True)
    instances_file.close()
    pool_file.close()
    for name, data in [('per_method.csv', rows), ('exact_representation.csv', exact_rows)]:
        with (out / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = []
    representation = []
    for n in args.sizes:
        for family in ('random', 'planted'):
            selected_exact = [r for r in exact_rows if r['n'] == n and r['family'] == family]
            representation.append({'n': n, 'family': family, 'instances': len(selected_exact),
                'all_clique_optima_native_suboptimal_instances': sum(r['clique_minimizer_best_native_gap'] > 0 for r in selected_exact),
                'some_clique_optima_native_suboptimal_instances': sum(r['clique_minimizer_worst_native_gap'] > 0 for r in selected_exact),
                'mean_best_native_gap_among_clique_optima': float(np.mean([r['clique_minimizer_best_native_gap'] for r in selected_exact]))})
            for method in ('native', 'clique', 'detached_cut'):
                selected = [r for r in rows if r['n'] == n and r['family'] == family and r['method'] == method]
                entry = {'n': n, 'family': family, 'method': method, 'instances': len(selected)}
                for key in ('raw_feasible_fraction', 'rounded_best_gap', 'rounded_mean_gap', 'refined_best_gap', 'refined_mean_gap', 'random_topk_best_gap', 'optimize_seconds'):
                    entry['mean_' + key] = float(np.mean([r[key] for r in selected]))
                entry['rounded_exact_hit_instances'] = sum(r['rounded_best_gap'] == 0 for r in selected)
                entry['refined_exact_hit_instances'] = sum(r['refined_best_gap'] == 0 for r in selected)
                summary.append(entry)
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if before != after:
        raise RuntimeError('source changed during this run; rerun on a stable checkout')
    report = {'config': vars(args), 'scope': 'Independent autograd mean-field reference on synthetic bisections; not legacy HIP/FEM end-to-end.',
        'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                        'numpy': np.__version__, 'torch': torch.__version__, 'torch_threads': torch.get_num_threads()},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': after, 'source_unchanged_during_run': before == after,
        'formula_audit': checks, 'runtime_audit': audit, 'exact_representation': representation, 'optimizer_summary': summary}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'formula_audit': checks, 'exact_representation': representation, 'optimizer_summary': summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=int, default=10)
    parser.add_argument('--sizes', type=int, nargs='+', default=[12, 16])
    parser.add_argument('--seed', type=int, default=4000)
    parser.add_argument('--trials', type=int, default=32)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.08)
    parser.add_argument('--balance-penalty', type=float, default=5.0)
    parser.add_argument('--output', default=str(ROOT / 'benchmarks/hypergraph/results/ogp-diagnostic-20260924/objective'))
    args = parser.parse_args()
    if min(args.instances, args.trials, args.steps) < 1 or any(n < 8 or n > 20 or n % 2 for n in args.sizes):
        parser.error('positive counts and even sizes from 8 through 20 are required')
    run(args)


if __name__ == '__main__':
    main()
