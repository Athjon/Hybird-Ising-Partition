"""Complete finite-size overlap diagnostics for synthetic balanced hypergraphs.

This is an oracle experiment, not an HIP performance benchmark or a proof of
asymptotic OGP. All vertices/edges have unit weight, every edge has four pins,
and the two blocks have exactly n/2 vertices. We enumerate one representative
per global label flip (vertex 0 in block 0), then count *every* unordered pair
of different representatives. No solution or pair is sampled.

Run from the repository root, with Python >= 3.9 and NumPy::

    python benchmarks/hypergraph/ogp_exact_geometry.py

The default output is benchmarks/hypergraph/results/
ogp-diagnostic-20260924/geometry. The random/planted ensembles below are
synthetic controls and are not the manuscript's real benchmark instances.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'results/ogp-diagnostic-20260924/geometry'
MAX_N = 16


def balanced_masks(n):
    """Bit 0 is zero; exactly half of the bits are one."""
    if n < 4 or n > MAX_N or n % 2:
        raise ValueError('n must be even, between 4 and 16')
    return np.asarray([
        sum(1 << v for v in ones)
        for ones in itertools.combinations(range(1, n), n // 2)
    ], dtype=np.uint32)


def popcount_table(n):
    # bin().count is intentionally compatible with Python 3.9.
    return np.fromiter((bin(i).count('1') for i in range(1 << n)),
                       dtype=np.int16, count=1 << n)


def make_instance(family, n, seed, edge_density=2, planted_probability=0.75):
    """Generate an explicitly specified multihypergraph.

    There are edge_density*n independently drawn four-pin edges. Repetitions
    are retained, equivalently giving repeated nets integer weights. Random
    draws are uniform over all four-subsets. Planted draws choose a uniform
    block of the fixed balanced split with probability planted_probability,
    then a uniform four-subset inside it; otherwise draw globally. The planted
    split is not asserted to be optimal or unique.
    """
    if family not in ('random', 'planted'):
        raise ValueError('unknown family')
    if n < 8 or n % 2:
        raise ValueError('four-pin planted ensembles need even n >= 8')
    rng = np.random.default_rng(seed)
    edges = []
    for _ in range(edge_density * n):
        vertices = np.arange(n)
        if family == 'planted' and rng.random() < planted_probability:
            block = int(rng.integers(2))
            vertices = np.arange(block * (n // 2), (block + 1) * (n // 2))
        edges.append(sorted(int(v) for v in rng.choice(vertices, 4, replace=False)))
    return edges


def native_costs(masks, edges):
    """Exact native binary cut-net = connectivity-minus-one, integer weights."""
    costs = np.zeros(len(masks), dtype=np.int64)
    for edge in edges:
        edge_mask = sum(1 << v for v in set(edge))
        intersection = masks & edge_mask
        costs += (intersection != 0) & (intersection != edge_mask)
    return costs


def complete_pair_histogram(masks, n, table=None, max_block_entries=262144):
    """Counts all i < j pairs by integer numerator of |1 - 2d/n|.

    Blocks bound pair-array memory; they do not subsample. A histogram includes
    zero counts at impossible numerators so arithmetic stays exact. Diagonal
    self-pairs are excluded and can be added explicitly by the caller.
    """
    masks = np.asarray(masks, dtype=np.uint32)
    if max_block_entries < 1:
        raise ValueError('max_block_entries must be positive')
    count = len(masks)
    hist = np.zeros(n + 1, dtype=np.int64)
    if count < 2:
        return hist
    if table is None:
        table = popcount_table(n)
    rows_per_block = max(1, max_block_entries // count)
    columns = np.arange(count)[None, :]
    for start in range(0, count, rows_per_block):
        stop = min(count, start + rows_per_block)
        distance = table[masks[start:stop, None] ^ masks[None, :]]
        numerator = np.abs(n - 2 * distance)
        upper = columns > np.arange(start, stop)[:, None]
        hist += np.bincount(numerator[upper], minlength=n + 1)
    expected_pairs = count * (count - 1) // 2
    if int(hist.sum()) != expected_pairs:
        raise AssertionError('pair enumeration is incomplete')
    return hist


def support_description(hist, feasible_hist, n, unique_count):
    support = [int(v) for v in np.flatnonzero(hist)]
    feasible_support = [int(v) for v in np.flatnonzero(feasible_hist)]
    missing = sorted(set(feasible_support) - set(support))
    gaps = []
    # This is an additional conservative diagnostic, not a necessary condition
    # for OGP. Standard all-pairs overlap geometry also permits q=1 self-pairs;
    # its separate diagnostic is computed below.
    for lower, upper in zip(support, support[1:]):
        absent = [v for v in feasible_support if lower < v < upper]
        if absent:
            gaps.append({
                'lower_supported_numerator': lower,
                'upper_supported_numerator': upper,
                'missing_feasible_numerators': absent,
            })
    with_self = sorted(set(support) | ({n} if unique_count else set()))
    feasible_with_self = sorted(set(feasible_support) | {n})
    gaps_including_self = []
    for lower, upper in zip(with_self, with_self[1:]):
        absent = [v for v in feasible_with_self if lower < v < upper]
        if absent:
            gaps_including_self.append({
                'lower_supported_numerator': lower,
                'upper_supported_numerator': upper,
                'missing_feasible_numerators': absent,
                'upper_boundary_is_self_overlap': upper == n,
            })
    nontrivial_set = unique_count >= 2 and any(value < n for value in support)
    nontrivial_including_self_gap = nontrivial_set and bool(gaps_including_self)
    return {
        'overlap_denominator': n,
        'distinct_pair_histogram': {str(i): int(hist[i]) for i in support},
        'distinct_pair_support_numerators': support,
        'distinct_pair_support': [i / n for i in support],
        'including_self_support_numerators': with_self,
        'including_self_support': [i / n for i in with_self],
        'missing_feasible_distinct_pair_numerators': missing,
        'missing_feasible_including_self_numerators': sorted(set(feasible_with_self) - set(with_self)),
        'interior_gaps_bounded_by_distinct_pair_support': gaps,
        # Retain the original field for data compatibility, with an explicit
        # alias so downstream consumers can see which definition it uses.
        'has_interior_gap': bool(gaps),
        'has_distinct_bounded_interior_gap': bool(gaps),
        'interior_gaps_including_self': gaps_including_self,
        'has_nontrivial_including_self_interior_gap': nontrivial_including_self_gap,
        'has_nontrivial_gap_with_self_upper_boundary': nontrivial_set and any(
            gap['upper_boundary_is_self_overlap'] for gap in gaps_including_self),
        'unique_near_optimal_count': unique_count,
        'distinct_unordered_pairs_evaluated': unique_count * (unique_count - 1) // 2,
        'self_pairs_not_in_distinct_histogram': unique_count,
    }


def brute_force_pair_histogram(masks, n):
    """Independent slow oracle: explicit spin dot products, no popcount table."""
    spins = [[1 if int(mask) & (1 << v) else -1 for v in range(n)] for mask in masks]
    hist = np.zeros(n + 1, dtype=np.int64)
    for a, b in itertools.combinations(spins, 2):
        hist[abs(sum(x * y for x, y in zip(a, b)))] += 1
    return hist


def pair_witness(masks, n, numerator):
    for i, first in enumerate(masks):
        for second in masks[i + 1:]:
            distance = bin(int(first) ^ int(second)).count('1')
            if abs(n - 2 * distance) == numerator:
                return [int(first), int(second)]
    return None


def validation_checks(baselines):
    """Verify a real tiny instance and a deliberately gapped solution-set fixture."""
    n = 8
    masks = balanced_masks(n)
    edges = make_instance('random', n, 20260924)
    costs = native_costs(masks, edges)
    # Native cost is independently recomputed through explicit edge labels.
    scalar_costs = [sum(len({(int(mask) >> v) & 1 for v in edge}) - 1
                        for edge in edges) for mask in masks]
    if not np.array_equal(costs, scalar_costs):
        raise AssertionError('native objective disagrees with scalar oracle')
    threshold_checks = []
    for epsilon in (0, 1, 2):
        near = masks[costs <= int(costs.min()) + epsilon]
        fast = complete_pair_histogram(near, n)
        slow = brute_force_pair_histogram(near, n)
        if not np.array_equal(fast, slow):
            raise AssertionError('blocked pair count differs from brute-force query')
        threshold_checks.append({'epsilon': epsilon, 'count': len(near),
                                 'pair_count': int(slow.sum()), 'passed': True})

    # This is a synthetic SET fixture, not an extra ensemble result. Every mask
    # is strictly balanced. Its pair support is {0, 8}/12, with feasible 4/12
    # absent. Both bounding points come from different solution classes.
    n = 12
    fixture = np.asarray([sum(1 << v for v in group) for group in
                          ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7],
                           [1, 2, 3, 8, 9, 10])], dtype=np.uint32)
    hist = complete_pair_histogram(fixture, n, max_block_entries=2)
    if not np.array_equal(hist, brute_force_pair_histogram(fixture, n)):
        raise AssertionError('fixture pair queries disagree')
    full = baselines[n] if n in baselines else complete_pair_histogram(balanced_masks(n), n)
    desc = support_description(hist, full, n, len(fixture))
    if desc['distinct_pair_support_numerators'] != [0, 8] or not desc['has_interior_gap']:
        raise AssertionError('gapped fixture was not recognized')
    witnesses = {str(q): pair_witness(fixture, n, q) for q in (0, 4, 8)}
    if witnesses['0'] is None or witnesses['8'] is None or witnesses['4'] is not None:
        raise AssertionError('gap is not bounded by witnessed support')
    two_classes = fixture[[0, 2]]
    two_hist = brute_force_pair_histogram(two_classes, n)
    two_desc = support_description(two_hist, full, n, len(two_classes))
    if two_desc['has_distinct_bounded_interior_gap'] or not two_desc['has_nontrivial_including_self_interior_gap']:
        raise AssertionError('including-self gap was conflated with distinct-bounded gap')
    return {
        'tiny_native_instance': {'n': 8, 'seed': 20260924, 'family': 'random',
                                 'cost_oracle_passed': True,
                                 'threshold_pair_queries': threshold_checks},
        'gapped_set_fixture': {'not_an_ensemble_instance': True, 'n': n,
                              'masks': fixture.tolist(), 'description': desc,
                              'pair_witness_by_numerator': witnesses,
                              'all_pair_queries_complete': True},
        'two_class_self_bounded_gap_fixture': {
            'not_an_ensemble_instance': True, 'n': n,
            'masks': two_classes.tolist(), 'description': two_desc,
            'low_overlap_distinct_pair': two_classes.tolist(),
            'high_overlap_self_pair': [int(two_classes[0])] * 2,
            'all_pair_queries_complete': True,
        },
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(args):
    started = time.perf_counter()
    output = Path(args.output)
    (output / 'instances').mkdir(parents=True, exist_ok=True)
    rows, baseline_hists, baseline_records, all_records = [], {}, [], []
    for n in args.sizes:
        masks = balanced_masks(n)
        table = popcount_table(n)
        full_hist = complete_pair_histogram(masks, n, table, args.max_block_entries)
        baseline_hists[n] = full_hist
        baseline_records.append({
            'n': n, 'feasible_classes_modulo_global_flip': len(masks),
            'feasible_labeled_assignments': 2 * len(masks),
            'expected_feasible_labeled_assignments': math.comb(n, n // 2),
            **support_description(full_hist, full_hist, n, len(masks)),
        })
        for family_index, family in enumerate(('random', 'planted')):
            for instance_index in range(args.instances):
                seed = args.seed + family_index * 1000000 + n * 1000 + instance_index
                instance_id = f'{family}-n{n:02d}-i{instance_index:03d}'
                tick = time.perf_counter()
                edges = make_instance(family, n, seed, args.edge_density,
                                      args.planted_probability)
                costs = native_costs(masks, edges)
                optimum = int(costs.min())
                thresholds = []
                for epsilon in args.epsilons:
                    near = masks[costs <= optimum + epsilon]
                    hist = complete_pair_histogram(near, n, table, args.max_block_entries)
                    desc = support_description(hist, full_hist, n, len(near))
                    witnesses = []
                    for gap in desc['interior_gaps_bounded_by_distinct_pair_support']:
                        witnesses.append({
                            'lower_pair': pair_witness(near, n, gap['lower_supported_numerator']),
                            'upper_pair': pair_witness(near, n, gap['upper_supported_numerator']),
                        })
                    including_self_witnesses = []
                    for gap in desc['interior_gaps_including_self']:
                        including_self_witnesses.append({
                            'lower_pair': pair_witness(near, n, gap['lower_supported_numerator']),
                            'upper_pair': ([int(near[0])] * 2 if gap['upper_boundary_is_self_overlap']
                                           else pair_witness(near, n, gap['upper_supported_numerator'])),
                        })
                    threshold = {'epsilon_absolute': epsilon, 'energy_ceiling': optimum + epsilon,
                                 **desc, 'gap_boundary_pair_witnesses': witnesses,
                                 'including_self_gap_boundary_pair_witnesses': including_self_witnesses}
                    thresholds.append(threshold)
                    rows.append({
                        'instance_id': instance_id, 'family': family, 'n': n,
                        'seed': seed, 'edges': len(edges), 'exact_optimum': optimum,
                        'epsilon_absolute': epsilon,
                        'unique_near_optimal_count': len(near),
                        'distinct_unordered_pairs_evaluated': desc['distinct_unordered_pairs_evaluated'],
                        'distinct_pair_support_numerators': json.dumps(desc['distinct_pair_support_numerators']),
                        'including_self_support_numerators': json.dumps(desc['including_self_support_numerators']),
                        'missing_feasible_distinct_pair_numerators': json.dumps(desc['missing_feasible_distinct_pair_numerators']),
                        'has_interior_gap': desc['has_interior_gap'],
                        'interior_gaps': json.dumps(desc['interior_gaps_bounded_by_distinct_pair_support']),
                        'has_distinct_bounded_interior_gap': desc['has_distinct_bounded_interior_gap'],
                        'has_nontrivial_including_self_interior_gap': desc['has_nontrivial_including_self_interior_gap'],
                        'has_nontrivial_gap_with_self_upper_boundary': desc['has_nontrivial_gap_with_self_upper_boundary'],
                        'including_self_interior_gaps': json.dumps(desc['interior_gaps_including_self']),
                    })
                record = {
                    'instance_id': instance_id, 'family': family, 'n': n, 'seed': seed,
                    'hyperedges': edges, 'hyperedge_weights': [1] * len(edges),
                    'node_weights': [1] * n, 'balance': 'exactly n/2 per block',
                    'exact_optimum': optimum,
                    'unique_hyperedges': len({tuple(e) for e in edges}),
                    'energy_histogram': {str(k): v for k, v in sorted(Counter(costs.tolist()).items())},
                    'feasible_class_masks': masks.tolist(),
                    'native_energy_by_feasible_mask': costs.tolist(),
                    'all_assignments_enumerated': True, 'all_pairs_enumerated': True,
                    'thresholds': thresholds, 'elapsed_seconds': time.perf_counter() - tick,
                }
                write_json(output / 'instances' / f'{instance_id}.json', record)
                all_records.append(record)
        print(f'n={n}: enumerated {len(masks)} feasible classes, '
              f'{int(full_hist.sum())} feasible distinct pairs, '
              f'{2 * args.instances} instances', flush=True)

    write_json(output / 'feasible_baselines.json', baseline_records)
    write_json(output / 'validation.json', validation_checks(baseline_hists))
    with (output / 'instances.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for n in args.sizes:
        for family in ('random', 'planted'):
            instance_records = [r for r in all_records if r['n'] == n and r['family'] == family]
            for epsilon in args.epsilons:
                subset = [r for r in rows if r['n'] == n and r['family'] == family
                          and r['epsilon_absolute'] == epsilon]
                summary.append({
                    'family': family, 'n': n, 'epsilon_absolute': epsilon,
                    'instances': len(subset),
                    'mean_exact_optimum': float(np.mean([r['exact_optimum'] for r in instance_records])),
                    'mean_unique_near_optimal_count': float(np.mean([r['unique_near_optimal_count'] for r in subset])),
                    'min_unique_near_optimal_count': min(r['unique_near_optimal_count'] for r in subset),
                    'max_unique_near_optimal_count': max(r['unique_near_optimal_count'] for r in subset),
                    'singleton_near_optimal_sets': sum(r['unique_near_optimal_count'] == 1 for r in subset),
                    'instances_missing_feasible_lattice_points': sum(bool(json.loads(r['missing_feasible_distinct_pair_numerators'])) for r in subset),
                    'instances_with_interior_gap': sum(r['has_interior_gap'] for r in subset),
                    'instances_with_distinct_bounded_interior_gap': sum(r['has_distinct_bounded_interior_gap'] for r in subset),
                    'instances_with_nontrivial_including_self_interior_gap': sum(r['has_nontrivial_including_self_interior_gap'] for r in subset),
                    'instances_with_nontrivial_gap_with_self_upper_boundary': sum(r['has_nontrivial_gap_with_self_upper_boundary'] for r in subset),
                    'distinct_pairs_evaluated': sum(r['distinct_unordered_pairs_evaluated'] for r in subset),
                })
    manifest = {
        'experiment': 'complete finite-size native binary hypergraph overlap geometry',
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'python': platform.python_version(), 'numpy': np.__version__,
        'overlap': 'q=abs(n-2*HammingDistance)/n, label-flip invariant',
        'near_optimal_set': 'native cut <= exact optimum + absolute epsilon',
        'edge_distribution': 'edge_density*n independent 4-subsets with replacement; repeated hyperedges retained',
        'planted_distribution': 'with planted_probability choose a uniform planted half then uniform 4-subset; otherwise uniform global 4-subset',
        'seed_formula': 'base_seed + (0 for random, 1000000 for planted) + n*1000 + instance_index',
        'pair_scope': 'all unordered pairs of different solution classes; self support separately reported',
        'interior_gap_definition': 'missing full-feasible distinct-pair lattice point strictly between two observed distinct-pair support values',
        'including_self_gap_definition': 'missing full-feasible overlap lattice point between two support values allowing q=1 self-pairs; count only sets with >=2 solution classes and a q<1 distinct pair',
        'legacy_field_semantics': 'has_interior_gap and instances_with_interior_gap retain the original distinct-bounded diagnostic; it is NOT an OGP necessary condition',
        'no_subsampling': True, 'maximum_n_guard': MAX_N,
        'limitations': [
            'Synthetic ensembles are not manuscript benchmark instances.',
            'n<=16 and three absolute thresholds cannot establish asymptotic OGP.',
            'A finite-size gap does not imply an algorithmic lower bound.',
            'The distinct-bounded diagnostic is stricter than the standard all-pairs overlap-gap definition; its absence is not absence of OGP.',
            'An including-self gap can reflect only a sparse low-energy window; it does not establish an algorithmic barrier.',
            'Singleton near-optimal sets are excluded from nontrivial gap evidence.',
            'Neither finite-size diagnostic establishes or excludes ensemble OGP at other sizes or thresholds.',
            'No HIP, FEM, SBM, MCMC or KaHyPar performance is evaluated here.',
            'Planted n=8 has only one internal 4-subset per block; retained repeated nets make this especially simple.',
            'Fixed absolute epsilon changes relative accuracy as n and m change.',
        ],
        'elapsed_seconds': time.perf_counter() - started,
        'total_native_instances': len(all_records), 'total_threshold_rows': len(rows),
        'total_distinct_near_optimal_pairs_evaluated': sum(r['distinct_unordered_pairs_evaluated'] for r in rows),
        'total_threshold_rows_with_interior_gap': sum(r['has_interior_gap'] for r in rows),
        'total_threshold_rows_with_distinct_bounded_interior_gap': sum(r['has_distinct_bounded_interior_gap'] for r in rows),
        'total_threshold_rows_with_nontrivial_including_self_interior_gap': sum(r['has_nontrivial_including_self_interior_gap'] for r in rows),
        'total_threshold_rows_with_nontrivial_gap_with_self_upper_boundary': sum(r['has_nontrivial_gap_with_self_upper_boundary'] for r in rows),
    }
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'summary.json', summary)
    lines = [
        '# Exact finite-size native hypergraph overlap diagnostic', '',
        'Synthetic control ensembles; these are not the manuscript benchmark instances. '
        'This experiment does not establish or refute asymptotic OGP.', '',
        f'Configuration: n={args.sizes}; {args.instances} instances per family and size; '
        f'{args.edge_density}n independently drawn four-pin unit-weight hyperedges; '
        f'planted within-block probability {args.planted_probability}; seed base {args.seed}.', '',
        'All nodes have unit weight. Each block contains exactly n/2 vertices. '
        'Native cut-net equals km1 for this binary setting. The complete feasible set is '
        'enumerated with bit 0 fixed to zero to remove global label flips. '
        'q = |n - 2d| / n. Every unordered pair of different equivalence classes is counted. '
        'Two gap diagnostics are reported: one conservatively requires both boundaries '
        'to come from different-class pairs, while the other also admits self-pairs q=1 '
        'as in the standard all-pairs definition. Singleton sets are excluded from '
        'nontrivial including-self gap evidence.', '',
        '| Family | n | Absolute epsilon | Mean optimum | Mean near-optimal classes | Singleton sets | Missing any feasible q | Distinct-bounded gaps | Including-self gaps (nontrivial) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summary:
        lines.append(f"| {r['family']} | {r['n']} | {r['epsilon_absolute']} | "
                     f"{r['mean_exact_optimum']:.2f} | {r['mean_unique_near_optimal_count']:.2f} | "
                     f"{r['singleton_near_optimal_sets']}/{r['instances']} | "
                     f"{r['instances_missing_feasible_lattice_points']}/{r['instances']} | "
                     f"{r['instances_with_distinct_bounded_interior_gap']}/{r['instances']} | "
                     f"{r['instances_with_nontrivial_including_self_interior_gap']}/{r['instances']} |")
    lines += ['',
              '“Missing any feasible q” includes empty or narrow support and is not evidence '
              'of separated clusters. “Distinct-bounded gaps” requires two actually attained '
              'distinct-pair overlap values bounding an absent full-feasible lattice point. '
              'This stricter condition is NOT necessary for OGP. “Including-self gaps” '
              'also permits q=1 as the upper boundary and requires at least two solution '
              'classes with a q<1 pair; a singleton is not counted.', '',
              f"Across {len(rows)} instance/threshold combinations, distinct-bounded gaps: "
              f"{manifest['total_threshold_rows_with_distinct_bounded_interior_gap']}; "
              f"nontrivial including-self gaps: "
              f"{manifest['total_threshold_rows_with_nontrivial_including_self_interior_gap']}; "
              f"gaps with q=1 self-overlap upper boundary: "
              f"{manifest['total_threshold_rows_with_nontrivial_gap_with_self_upper_boundary']}. "
              'A self-bounded gap may simply reflect a sparse low-energy window, '
              'and does not establish an algorithmic barrier.', '',
              f"Processed {manifest['total_native_instances']} instances and "
              f"{manifest['total_distinct_near_optimal_pairs_evaluated']:,} near-optimal distinct pairs "
              '(summed over thresholds, so nested sets repeat pairs). All are exhaustive.', '',
              'Validation compares exact native costs with an independent scalar evaluator, '
              'and complete pair counts for a real n=8 instance with explicit spin-dot-product '
              'enumeration. A separate hand-built balanced solution-set fixture checks a '
              'distinct-bounded gap with witnesses on both sides, and a two-class fixture '
              'verifies a gap bounded above by q=1. These fixtures are not included in ensemble results.', '',
              'Files: `instances/*.json` contain hyperedges, all feasible masks and energies, '
              'threshold supports/counts and gap witnesses; `instances.csv` has threshold rows; '
              '`feasible_baselines.json` has complete feasible-pair supports; `validation.json`, '
              '`summary.json`, and `manifest.json` record checks, aggregation and provenance.', '',
              'Limitations:', '']
    lines += ['- ' + item for item in manifest['limitations']]
    lines += ['', 'Reproduce from repository root:', '', '```sh',
              'python benchmarks/hypergraph/ogp_exact_geometry.py', '```', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in (
        'elapsed_seconds', 'total_native_instances', 'total_threshold_rows',
        'total_distinct_near_optimal_pairs_evaluated',
        'total_threshold_rows_with_distinct_bounded_interior_gap',
        'total_threshold_rows_with_nontrivial_including_self_interior_gap',
        'total_threshold_rows_with_nontrivial_gap_with_self_upper_boundary')}, indent=2))
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[8, 12, 16])
    parser.add_argument('--instances', type=int, default=10)
    parser.add_argument('--epsilons', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--edge-density', type=int, default=2)
    parser.add_argument('--planted-probability', type=float, default=0.75)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--max-block-entries', type=int, default=262144)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(n < 8 or n > MAX_N or n % 2 for n in args.sizes):
        parser.error('sizes must be even, >=8 and <=16 for complete enumeration')
    if args.instances < 1 or args.edge_density < 1 or args.max_block_entries < 1:
        parser.error('instances, edge-density, and max-block-entries must be positive')
    if any(epsilon < 0 for epsilon in args.epsilons):
        parser.error('absolute epsilons must be nonnegative')
    if not 0 <= args.planted_probability <= 1:
        parser.error('planted-probability must be between zero and one')
    args.sizes = sorted(set(args.sizes))
    args.epsilons = sorted(set(args.epsilons))
    return args


if __name__ == '__main__':
    run(parse_args())
