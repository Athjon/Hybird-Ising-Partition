"""Recompute paired seed summaries and figures from saved follow-up runs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text())


def success_cut(unit, method):
    if method == 'restarts':
        value = unit.get('greedy_restarts', {})
        return value['winner']['native_cut'] if value.get('status') == 'success' else None
    value = unit.get('single_runs', {}).get(method, {})
    return value['final']['native_cut'] if value.get('status') == 'success' else None


def paired(rows, control, rng):
    pairs = [(success_cut(r, 'fem'), success_cut(r, control)) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return {'pairs': 0}
    values = np.array(pairs)
    delta = values[:, 1] - values[:, 0]
    boot = rng.choice(delta, size=(20_000, len(delta)), replace=True).mean(axis=1)
    return {'pairs': len(pairs), 'fem_wins': int(np.sum(delta > 0)),
            'ties': int(np.sum(delta == 0)), 'fem_losses': int(np.sum(delta < 0)),
            'mean_fem_cut': float(values[:, 0].mean()), 'mean_control_cut': float(values[:, 1].mean()),
            'mean_control_minus_fem': float(delta.mean()),
            'median_control_minus_fem': float(np.median(delta)),
            'mean_delta_seed_bootstrap_95_percentile': np.quantile(boot, [.025, .975]).tolist()}


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    timing_path = args.multiseed / 'summary.json'
    mechanisms_path = args.search / 'summary.json'
    timing, mechanisms = load(timing_path), load(mechanisms_path)
    assert timing['sources_unchanged_during_run'] and timing['inputs_unchanged_during_run']
    assert mechanisms['sources_unchanged_during_run']
    rng = np.random.default_rng(20260924)
    report = {'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (timing_path, mechanisms_path)},
              'bootstrap': '20,000 paired resamples of seeds, RNG seed 20260924; conditional on one fixed input and paired successful returns. Not confidence across datasets.',
              'real_instances': [], 'mechanism_instances': []}
    for instance in sorted({r['instance'] for r in timing['units']}):
        rows = [r for r in timing['units'] if r['instance'] == instance]
        result = {'instance': instance, 'seeds': [r['seed'] for r in rows], 'methods': {},
                  'paired_single_greedy': paired(rows, 'greedy', rng),
                  'paired_restarts': paired(rows, 'restarts', rng)}
        for method in ('greedy', 'fem', 'restarts'):
            cuts = [success_cut(r, method) for r in rows]
            cuts = [c for c in cuts if c is not None]
            statuses = [(r.get('greedy_restarts', {}) if method == 'restarts'
                         else r.get('single_runs', {}).get(method, {})).get('status', r['status'])
                        for r in rows]
            result['methods'][method] = {'statuses': dict(Counter(statuses)),
                'mean_cut_successful_only': float(np.mean(cuts)) if cuts else None,
                'median_cut_successful_only': float(np.median(cuts)) if cuts else None}
            if method != 'restarts':
                elapsed = [r.get('single_runs', {}).get(method, {}).get('pipeline_seconds') for r in rows]
                elapsed = [v for v in elapsed if v is not None]
                result['methods'][method]['median_pipeline_seconds'] = float(np.median(elapsed)) if elapsed else None
        result['restart_counts_in_budget'] = [r.get('greedy_restarts', {}).get('eligible_completed_runs', 0) for r in rows]
        result['restart_overshoot_seconds'] = [r.get('greedy_restarts', {}).get('overshoot_seconds', 0) for r in rows]
        report['real_instances'].append(result)
    for record in mechanisms['instances']:
        result = {k: record[k] for k in ('name', 'family', 'initial_cost', 'exact_optimum',
                  'local_reachability', 'exact_swap_barrier', 'aggregate') if k in record}
        result['proposal_rates'] = {}
        for method in ('zero_local', 'anneal_local', 'anneal_block'):
            rows = [r for r in record['runs'] if r['method'] == method]
            totals = Counter()
            for row in rows:
                totals.update({k: v for k, v in row['proposal_statistics'].items() if k != 'by_kind'})
            result['proposal_rates'][method] = {k: totals[k]/totals['attempted']
                for k in ('accepted', 'capacity_rejected', 'uphill_accepted', 'no_op')}
        report['mechanism_instances'].append(result)
    (args.output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    for ax, instance in zip(axes, ('ibm01', 'ibm02')):
        rows = [r for r in timing['units'] if r['instance'] == instance]
        for method, label, color, marker in (
                ('greedy', 'Single greedy', '#7a7a7a', 'o'),
                ('fem', 'Native FEM', '#2166ac', 'o'),
                ('restarts', 'Greedy at FEM time budget', '#b35b14', 'x')):
            values = [success_cut(r, method) for r in rows]
            values = [np.nan if v is None else v for v in values]
            ax.plot([r['seed'] for r in rows], values, label=label, color=color,
                    marker=marker, linewidth=1.3, markersize=5)
        missing = sum(success_cut(r, 'restarts') is None for r in rows)
        ax.set_title(f'{instance.upper()} | restart no-result: {missing}/{len(rows)}')
        ax.set_xlabel('Hierarchy / solver seed')
        ax.set_ylabel('Final native km1 (lower is better)')
        ax.set_xticks([r['seed'] for r in rows])
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(args.output / 'multiseed-cuts.png', dpi=180)
    fig.savefig(args.output / 'multiseed-cuts.pdf')
    plt.close(fig)
    print(json.dumps(report['real_instances'], indent=2))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--multiseed', type=Path, default=root / 'benchmarks/hypergraph/results/fem-multiseed-20260924-v2')
    parser.add_argument('--search', type=Path, default=root / 'benchmarks/hypergraph/results/native-search-20260924')
    parser.add_argument('--output', type=Path, default=root / 'benchmarks/hypergraph/results/ogp-search-followup-20260924')
    main(parser.parse_args())
