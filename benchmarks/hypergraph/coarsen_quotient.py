"""Reproducible small-instance ablation for quotient coarsening.

Each coarse graph is solved exactly by enumeration.  This isolates the loss
caused by contractions from FEM/SBM and refinement behavior.

Run from the repository root::

    python benchmarks/hypergraph/coarsen_quotient.py --instances 80
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hyper_solver import KahyparLikeSolver
from src.partition.hyper_quotient import connectivity_cost


def make_instance(family: str, seed: int, planted_probability: float = 0.85):
    rng = np.random.default_rng(seed)
    edges = []
    for _ in range(14):
        size = int(rng.integers(2, 5))
        if family == 'planted' and rng.random() < planted_probability:
            half = np.arange(0, 4) if rng.random() < 0.5 else np.arange(4, 8)
            edge = rng.choice(half, size=size, replace=False)
        else:
            edge = rng.choice(8, size=size, replace=False)
        edges.append(sorted(int(v) for v in edge))
    return edges


def exact_best_cut(edges, vertex_map, coarse_weights, epsilon=0.25):
    best = float('inf')
    limit = (1.0 + epsilon) * float(sum(coarse_weights)) / 2.0
    for labels in itertools.product(range(2), repeat=len(coarse_weights)):
        block_weights = np.bincount(labels, weights=coarse_weights, minlength=2)
        if np.max(block_weights) > limit + 1e-12:
            continue
        fine_labels = np.asarray(labels, dtype=np.int64)[vertex_map]
        best = min(best, connectivity_cost(fine_labels, edges))
    return best


def run_family(family, count, seed, num_pilots, boundary_weight,
               pilot_refine_passes, planted_probability=0.85):
    solver = KahyparLikeSolver()
    differences = []
    times = [[], []]
    infeasible = [0, 0]
    coarse_sizes = [[], []]
    original_optima = []

    for trial in range(count):
        edges = make_instance(family, seed + trial, planted_probability)
        original_optimum = exact_best_cut(edges, np.arange(8), np.ones(8))
        cuts = []
        for method_index, method in enumerate(('hem', 'boundary')):
            start = time.perf_counter()
            result = solver.coarsen(
                edges, 8, 2, coarsen_to=4, seed=trial,
                score_mode=method, enforce_balance_cap=True,
                epsilon=0.25, num_pilots=num_pilots,
                boundary_weight=boundary_weight,
                pilot_refine_passes=pilot_refine_passes,
            )
            times[method_index].append((time.perf_counter() - start) * 1000.0)
            coarse_weights = result['coarse_node_weights'].numpy()
            coarse_sizes[method_index].append(len(coarse_weights))
            cut = exact_best_cut(
                edges, result['original_to_coarse'], coarse_weights,
            )
            if not np.isfinite(cut):
                infeasible[method_index] += 1
            elif cut + 1e-12 < original_optimum:
                raise AssertionError('coarsening cannot improve on the original exact optimum')
            cuts.append(cut)
        if all(np.isfinite(cut) for cut in cuts):
            differences.append(tuple(cuts))
            original_optima.append(original_optimum)

    values = np.asarray(differences, dtype=np.float64)
    gains = values[:, 0] - values[:, 1] if len(values) else np.empty(0)
    gain_mean = float(np.mean(gains)) if len(gains) else float('nan')
    # Paired normal-approximation interval; instance generation is the sampling unit.
    margin = 1.96 * float(np.std(gains, ddof=1)) / np.sqrt(len(gains)) if len(gains) > 1 else float('nan')
    wins = int(np.sum(values[:, 1] < values[:, 0])) if len(values) else 0
    ties = int(np.sum(values[:, 1] == values[:, 0])) if len(values) else 0
    losses = int(np.sum(values[:, 1] > values[:, 0])) if len(values) else 0
    return {
        'family': family,
        'instances': count,
        'comparable': len(values),
        'original_mean_cut': float(np.mean(original_optima)) if original_optima else float('nan'),
        'hem_mean_cut': float(np.mean(values[:, 0])) if len(values) else float('nan'),
        'boundary_mean_cut': float(np.mean(values[:, 1])) if len(values) else float('nan'),
        'hem_mean_contraction_loss': float(np.mean(values[:, 0] - original_optima)) if len(values) else float('nan'),
        'boundary_mean_contraction_loss': float(np.mean(values[:, 1] - original_optima)) if len(values) else float('nan'),
        'boundary_gain_mean': gain_mean,
        'boundary_gain_ci_low': gain_mean - margin,
        'boundary_gain_ci_high': gain_mean + margin,
        'boundary_wins': wins,
        'ties': ties,
        'boundary_losses': losses,
        'hem_infeasible': infeasible[0],
        'boundary_infeasible': infeasible[1],
        'hem_mean_ms': float(np.mean(times[0])),
        'boundary_mean_ms': float(np.mean(times[1])),
        'hem_mean_nodes': float(np.mean(coarse_sizes[0])),
        'boundary_mean_nodes': float(np.mean(coarse_sizes[1])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=int, default=80)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num-pilots', type=int, default=4)
    parser.add_argument('--boundary-weight', type=float, default=0.4)
    parser.add_argument('--pilot-refine-passes', type=int, default=2)
    parser.add_argument('--planted-probability', type=float, default=0.85)
    args = parser.parse_args()
    if args.instances < 1:
        parser.error('--instances must be positive')
    if not 0.0 <= args.planted_probability <= 1.0:
        parser.error('--planted-probability must be between zero and one')

    for family in ('random', 'planted'):
        result = run_family(
            family, args.instances, args.seed, args.num_pilots,
            args.boundary_weight, args.pilot_refine_passes,
            args.planted_probability,
        )
        print(' '.join(f'{key}={value}' for key, value in result.items()))


if __name__ == '__main__':
    main()
