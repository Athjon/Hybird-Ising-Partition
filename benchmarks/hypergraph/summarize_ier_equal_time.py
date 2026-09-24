"""Aggregate saved equal-time checkpoints and draw source-backed quality curves.

Does not run solvers or use post-deadline candidates. The primary mean includes
the explicitly recorded common fallback when no V-cycle completed in time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    base = args.results
    source = base / 'summary.json'
    summary = json.loads(source.read_text())
    arms = list(summary['config']['arms'])
    budgets = summary['config']['budgets_seconds']
    rows, aggregates, comparisons = [], [], []
    for case in summary['cases']:
        if case['status'] != 'completed' or set(case['arms']) != set(arms):
            raise ValueError(f'incomplete case: {case["instance"]}, seed {case["seed"]}')
        for arm in arms:
            previous = float('inf')
            for checkpoint in case['arms'][arm]['checkpoints']:
                best = checkpoint['best_available']
                if best is None:
                    raise ValueError('no verified fallback or completed output at a checkpoint')
                if best['available_seconds'] > checkpoint['budget_seconds']:
                    raise ValueError('post-deadline candidate in checkpoint')
                if best['native_cut'] > previous:
                    raise ValueError('best-available curve increased')
                previous = best['native_cut']
                rows.append({'instance': case['instance'], 'seed': case['seed'], 'arm': arm,
                    'budget_seconds': checkpoint['budget_seconds'], 'native_cut': best['native_cut'],
                    'source': best['source'], 'run_id': best['run_id'],
                    'available_seconds': best['available_seconds'],
                    'optimized_status': checkpoint['optimized_status'],
                    'eligible_completed_runs': checkpoint['eligible_completed_runs'],
                    'eligible_completed_vcycles': checkpoint['eligible_completed_vcycles'],
                    'eligible_flow_continuations': checkpoint['eligible_flow_continuations']})
    for instance in summary['config']['instances']:
        for budget in budgets:
            for arm in arms:
                selected = [r for r in rows if r['instance'] == instance and
                            r['budget_seconds'] == budget and r['arm'] == arm]
                if len(selected) != len(summary['config']['seeds']):
                    raise ValueError('missing or duplicated paired observation')
                values = [r['native_cut'] for r in selected]
                aggregates.append({'instance': instance, 'budget_seconds': budget, 'arm': arm,
                    'cases': len(selected), 'mean_best_available_native_cut': float(np.mean(values)),
                    'median_best_available_native_cut': float(np.median(values)),
                    'completed_cases': sum(r['optimized_status'] == 'success' for r in selected),
                    'initial_only_cases': sum(r['source'] == 'initial_only' for r in selected),
                    'total_eligible_vcycles': sum(r['eligible_completed_vcycles'] for r in selected),
                    'total_eligible_flow_continuations': sum(r['eligible_flow_continuations'] for r in selected)})
            fem = {r['seed']: r for r in rows if r['instance'] == instance and
                   r['budget_seconds'] == budget and r['arm'] == 'fem'}
            for comparator in ('pairs', 'deterministic', 'flow'):
                other = {r['seed']: r for r in rows if r['instance'] == instance and
                         r['budget_seconds'] == budget and r['arm'] == comparator}
                differences = [fem[seed]['native_cut'] - other[seed]['native_cut'] for seed in fem]
                comparisons.append({'instance': instance, 'budget_seconds': budget,
                    'comparator': comparator, 'fem_wins': sum(d < 0 for d in differences),
                    'ties': sum(d == 0 for d in differences), 'fem_losses': sum(d > 0 for d in differences),
                    'mean_fem_minus_comparator': float(np.mean(differences)),
                    'paired_differences_by_seed': {str(seed): fem[seed]['native_cut'] -
                                                  other[seed]['native_cut'] for seed in fem}})
    failures = [{'instance': c['instance'], 'seed': c['seed'], 'arm': arm,
                 'validation_summary': c['arms'][arm].get('validation_summary')}
                for c in summary['cases'] for arm in arms if c['arms'][arm].get('has_failures')]
    result = {'summary_sha256': sha(source), 'script_sha256': sha(__file__),
        'primary_budget_seconds': summary['config']['primary_budget_seconds'],
        'arms_with_failures': failures,
        'metric': 'Best verified objective available by the deadline, including the given coarse-lift fallback.',
        'limitations': ['Seeds per fixed input are paired cases, not independent graph instances.',
                       'Budget checkpoints reuse the same timed run and are not independent samples.',
                       'Means include initial_only outcomes; completion counts are reported separately.'],
        'rows': rows, 'aggregates': aggregates, 'comparisons': comparisons}
    (base / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    instances = summary['config']['instances']
    colors = {'fem': '#2457a7', 'pairs': '#d56a18', 'deterministic': '#37815a', 'flow': '#6f647b'}
    labels = {'fem': 'FEM + FM', 'pairs': 'Pairs + FM', 'deterministic': 'Shared candidates + FM',
              'flow': 'FM V-cycle + polishing'}
    fig, axes = plt.subplots(1, len(instances), figsize=(11, 4.5), squeeze=False)
    for ax, instance in zip(axes[0], instances):
        for arm in arms:
            selected = [r for r in aggregates if r['instance'] == instance and r['arm'] == arm]
            ax.plot([r['budget_seconds'] for r in selected],
                    [r['mean_best_available_native_cut'] for r in selected],
                    marker='o', color=colors[arm], label=labels[arm], linewidth=1.8)
        ax.set_title(instance.upper() + f' | {len(summary["config"]["seeds"])} paired starts')
        ax.set_xlabel('Wall-time budget after common coarse input (s)')
        ax.set_ylabel('Mean best available native km1 (lower is better)')
        ax.set_xticks(budgets)
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8, loc='best')
    fig.suptitle('Equal-time complete V-cycle strategies', fontsize=13)
    fig.text(.5, .015, 'Unfinished V-cycles return the common initial fallback; no post-deadline outputs. '
             'Coarsening and coarse initialization excluded.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .045, 1, .95])
    fig.savefig(base / 'equal-time-quality.png', dpi=180)
    fig.savefig(base / 'equal-time-quality.pdf')
    plt.close(fig)
    primary = summary['config']['primary_budget_seconds']
    fig, axes = plt.subplots(1, len(instances), figsize=(10, 4.5), squeeze=False)
    for ax, instance in zip(axes[0], instances):
        paired = []
        for index, seed in enumerate(summary['config']['seeds']):
            values = {r['arm']: r['native_cut'] for r in rows if r['instance'] == instance
                      and r['seed'] == seed and r['budget_seconds'] == primary}
            if values['flow'] <= 0:
                raise ValueError('relative primary plot requires a positive flow reference')
            ratios = [100 * values[arm] / values['flow'] for arm in arms]
            paired.append(ratios)
            ax.plot(range(len(arms)), ratios, marker='o', linewidth=1, alpha=.65,
                    color=plt.get_cmap('tab10')(index % 10), label=f'Seed {seed}')
        ax.plot(range(len(arms)), np.mean(paired, axis=0), color='black', marker='D',
                linewidth=2.2, label='Mean paired ratio')
        ax.axhline(100, color='gray', linewidth=.8, linestyle='--')
        ax.set_xticks(range(len(arms)), ['FEM', 'Pairs', 'Shared\ncandidates', 'FM reference'])
        ax.set_title(instance.upper())
        ax.set_ylabel('Native km1 (% of paired FM reference; lower is better)')
        ax.grid(axis='y', alpha=.2)
    handles, legend_labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8, loc='lower center', ncol=6,
               bbox_to_anchor=(.5, .055))
    fig.suptitle(f'Primary comparison: {primary:g}-second deadline', fontsize=13)
    fig.text(.5, .012, 'Each colored line shares one hierarchy and coarse start across methods. '
             'Given coarse setup excluded; only eligible outputs used.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .145, 1, .95])
    fig.savefig(base / 'equal-time-primary.png', dpi=180)
    fig.savefig(base / 'equal-time-primary.pdf')
    plt.close(fig)
    print(json.dumps({'aggregates': aggregates, 'comparisons': comparisons}, indent=2))


if __name__ == '__main__':
    main()
