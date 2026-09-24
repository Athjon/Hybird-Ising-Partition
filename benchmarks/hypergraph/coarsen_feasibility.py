"""Weighted small-instance ablation of the exact global coarsening guard.

Run from the repository root::

    python benchmarks/hypergraph/coarsen_feasibility.py --instances 300

All original instances are exactly balance-feasible. Coarse cuts are solved
by enumeration; infeasible coarse results are reported separately.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hypergraph.coarsen_quotient import exact_best_cut, make_instance
from src.hyper_solver import KahyparLikeSolver
from src.partition.hyper_quotient import balanced_packing_feasible


def feasible_weights(seed):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        weights = rng.integers(1, 5, size=8).astype(float)
        if balanced_packing_feasible(weights, 2, float(weights.sum()) / 2):
            return weights
    raise RuntimeError('could not sample a balance-feasible weight vector')


def run_family(family, count, seed):
    solver = KahyparLikeSolver()
    keys = ('hem_local', 'hem_guarded', 'boundary_local', 'boundary_guarded')
    records = {
        key: {'infeasible': 0, 'cut': [], 'loss': [], 'nodes': [], 'ms': [],
              'rescued_loss': []}
        for key in keys
    }
    paired = {'hem': [], 'boundary': []}

    for trial in range(count):
        edges = make_instance(family, seed + trial)
        weights = feasible_weights(seed + 100000 + trial)
        optimum = exact_best_cut(edges, np.arange(8), weights, epsilon=0.0)
        assert np.isfinite(optimum)
        cuts = {}
        for mode in ('hem', 'boundary'):
            for guarded in (False, True):
                key = f'{mode}_{"guarded" if guarded else "local"}'
                start = time.perf_counter()
                result = solver.coarsen(
                    edges, 8, 2, coarsen_to=4, seed=trial,
                    node_weights=weights, epsilon=0.0,
                    score_mode=mode, enforce_balance_cap=True,
                    enforce_global_feasibility=guarded,
                    num_pilots=4, pilot_refine_passes=2,
                )
                records[key]['ms'].append((time.perf_counter() - start) * 1000)
                coarse_weights = result['coarse_node_weights'].numpy()
                records[key]['nodes'].append(len(coarse_weights))
                cut = exact_best_cut(
                    edges, result['original_to_coarse'], coarse_weights,
                    epsilon=0.0,
                )
                if guarded and not np.isfinite(cut):
                    raise AssertionError('global guard returned an infeasible quotient')
                if np.isfinite(cut) and cut < optimum - 1e-12:
                    raise AssertionError('coarse optimum beat the original optimum')
                if not np.isfinite(cut):
                    records[key]['infeasible'] += 1
                else:
                    records[key]['cut'].append(cut)
                    records[key]['loss'].append(cut - optimum)
                cuts[key] = cut
            local, guarded_cut = cuts[f'{mode}_local'], cuts[f'{mode}_guarded']
            if np.isfinite(local):
                paired[mode].append((local, guarded_cut))
            else:
                records[f'{mode}_guarded']['rescued_loss'].append(guarded_cut - optimum)

    for key, record in records.items():
        mode = key.split('_')[0]
        pairs = np.asarray(paired[mode], dtype=float)
        gain = float(np.mean(pairs[:, 0] - pairs[:, 1])) if len(pairs) else float('nan')
        print(' '.join(f'{k}={v}' for k, v in {
            'family': family,
            'method': key,
            'instances': count,
            'infeasible': record['infeasible'],
            'mean_nodes': float(np.mean(record['nodes'])),
            'mean_cut_feasible_only': float(np.mean(record['cut'])) if record['cut'] else float('nan'),
            'mean_contraction_loss_feasible_only': float(np.mean(record['loss'])) if record['loss'] else float('nan'),
            'rescued_instances': len(record['rescued_loss']),
            'rescued_mean_contraction_loss': float(np.mean(record['rescued_loss'])) if record['rescued_loss'] else float('nan'),
            'mean_ms': float(np.mean(record['ms'])),
            'paired_local_minus_guarded_cut': gain,
        }.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=int, default=300)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.instances < 1:
        parser.error('--instances must be positive')
    for family in ('random', 'planted'):
        run_family(family, args.instances, args.seed)


if __name__ == '__main__':
    main()
