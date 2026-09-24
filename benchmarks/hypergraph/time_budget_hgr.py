"""Compare one boundary run with HEM restarts at the same time budget.

The input must be an unweighted hMETIS .hgr file. A reproducible real case is
mt-KaHyPar's ibm01.hgr at commit eee7b7a03dbbd565a39f6cb2679b083e484c905e:
https://raw.githubusercontent.com/kahypar/mt-kahypar/eee7b7a03dbbd565a39f6cb2679b083e484c905e/lib/examples/ibm01.hgr
SHA-256: 40f7f7c4dfd96c06b0570f696e67b9c667ac2cbdf4d0858da5e690f4f4d5ac72

A second case, ibm02.hgr, is available from the ISPD benchmark collection at
TILOS-AI-Institute/HypergraphPartitioning commit ff614601a9b8f7853e21019e1fd320f44e445f3c:
https://raw.githubusercontent.com/TILOS-AI-Institute/HypergraphPartitioning/ff614601a9b8f7853e21019e1fd320f44e445f3c/benchmark/ISPD_benchmark/ibm02.hgr
SHA-256: ff09f3be9ed84a8c13257f1655555938072cdf01fae40f1548795763981eae05

Example::

    python benchmarks/hypergraph/time_budget_hgr.py /path/to/ibm01.hgr \
        --trials 8 --pilot-refine-passes 0

The strict HEM result includes only runs completed within the boundary time.
The bracketed result includes one extra HEM run that crosses the deadline,
making the granularity of whole-run comparisons explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hyper_solver import KahyparLikeSolver, HyperRefineSolver, vcycle_uncoarsen
from src.partition.hyper_quotient import connectivity_cost


def read_unweighted_hgr(path):
    with open(path, encoding='utf-8') as stream:
        lines = (line.strip() for line in stream)
        lines = (line for line in lines if line and not line.startswith('%'))
        header = [int(value) for value in next(lines).split()]
        if len(header) < 2 or (len(header) > 2 and header[2] != 0):
            raise ValueError('only unweighted hMETIS files are supported')
        num_edges, num_vertices = header[:2]
        edges = [[int(pin) - 1 for pin in line.split()] for line in lines]
    if len(edges) != num_edges:
        raise ValueError(f'header says {num_edges} edges, found {len(edges)}')
    if any(pin < 0 or pin >= num_vertices for edge in edges for pin in edge):
        raise ValueError('hyperedge contains an out-of-range vertex')
    return num_vertices, edges


def solve_once(edges, num_vertices, seed, mode, args, solver, refiner):
    start = time.perf_counter()
    result = solver.coarsen(
        edges, num_vertices, args.q, coarsen_to=args.coarsen_to,
        seed=seed, score_mode=mode, enforce_balance_cap=True,
        epsilon=args.epsilon, num_pilots=args.num_pilots,
        pilot_refine_passes=args.pilot_refine_passes,
    )
    coarse_assignment = solver.initial_partition_greedy(
        result['coarse_hyperedges'], result['coarse_node_weights'], args.q,
        epsilon=args.epsilon, seed=seed,
    )
    assignment = vcycle_uncoarsen(
        coarse_assignment, result['hierarchy_stack'], edges, args.q,
        refiner, verbose=False,
    )
    cut = connectivity_cost(assignment, edges)
    block_weights = np.bincount(assignment, minlength=args.q)
    max_excess = float(block_weights.max() / (num_vertices / args.q) - 1)
    if max_excess > args.epsilon + 1e-12:
        raise AssertionError('result violates the maximum block capacity')
    return {
        'cut': cut,
        'elapsed': time.perf_counter() - start,
        'max_excess': max_excess,
        'coarse_vertices': len(result['coarse_groups']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hgr', type=Path)
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--q', type=int, default=4)
    parser.add_argument('--coarsen-to', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--flow-passes', type=int, default=2)
    parser.add_argument('--num-pilots', type=int, default=4)
    parser.add_argument('--pilot-refine-passes', type=int, default=0)
    parser.add_argument('--sha256', help='expected SHA-256 of the input file')
    args = parser.parse_args()
    if args.trials < 1 or args.q < 2 or args.coarsen_to < 1:
        parser.error('trials, q, and coarsen-to must be positive (q at least 2)')
    if args.sha256:
        digest = hashlib.sha256(args.hgr.read_bytes()).hexdigest()
        if digest.lower() != args.sha256.lower():
            parser.error(f'SHA-256 mismatch: {digest}')
    num_vertices, edges = read_unweighted_hgr(args.hgr)
    solver = KahyparLikeSolver()
    refiner = HyperRefineSolver()
    refiner.update_params(
        flow_passes=args.flow_passes, max_imbalance=args.epsilon,
        repair_balance=False,
    )

    results = []
    for trial in range(args.trials):
        boundary_seed = args.seed + trial
        boundary = solve_once(
            edges, num_vertices, boundary_seed, 'boundary', args, solver, refiner,
        )
        budget = boundary['elapsed']
        spent = 0.0
        within_best = float('inf')
        one_extra_best = float('inf')
        completed = 0
        attempted = 0
        while spent < budget:
            hem_seed = args.seed + 10000 + 1000 * trial + attempted
            hem = solve_once(edges, num_vertices, hem_seed, 'hem', args, solver, refiner)
            attempted += 1
            spent += hem['elapsed']
            one_extra_best = min(one_extra_best, hem['cut'])
            if spent <= budget:
                completed += 1
                within_best = min(within_best, hem['cut'])
        results.append((boundary['cut'], within_best, one_extra_best, budget, completed))
        print(
            f'trial={trial} boundary_seed={boundary_seed} '
            f'boundary_cut={boundary["cut"]:.0f} boundary_s={budget:.3f} '
            f'hem_completed={completed} hem_best_within={within_best:.0f} '
            f'hem_best_with_one_extra={one_extra_best:.0f} '
            f'hem_attempts={attempted} hem_attempted_s={spent:.3f}',
            flush=True,
        )

    values = np.asarray(results)
    comparable = np.isfinite(values[:, 1])
    print(
        f'summary trials={args.trials} comparable={int(comparable.sum())} '
        f'boundary_mean_cut={values[:, 0].mean():.3f} '
        f'hem_best_within_mean={values[comparable, 1].mean() if comparable.any() else float("nan"):.3f} '
        f'hem_best_one_extra_mean={values[:, 2].mean():.3f} '
        f'boundary_wins_within={int(np.sum(values[comparable, 0] < values[comparable, 1]))} '
        f'hem_wins_within={int(np.sum(values[comparable, 0] > values[comparable, 1]))} '
        f'ties_within={int(np.sum(values[comparable, 0] == values[comparable, 1]))} '
        f'mean_budget_s={values[:, 3].mean():.3f} '
        f'mean_hem_completed={values[:, 4].mean():.3f}'
    )


if __name__ == '__main__':
    main()
