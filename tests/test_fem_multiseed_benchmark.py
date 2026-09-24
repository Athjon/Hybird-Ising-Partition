"""Strict-deadline and artifact checks without running timed solver workloads."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hypergraph.compare_fem_multiseed import (
    choose_restart_winner, collect_restarts, measure, persist_unit,
)


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


def runner(clock, durations, cuts, statuses=None):
    statuses = statuses or ['success'] * len(durations)

    def run(attempt):
        clock.now += durations[attempt]
        return {'run_id': f'restart_{attempt:04d}', 'solver_seed': 1000 + attempt,
                'finished_at': clock(), 'pipeline_seconds': durations[attempt],
                'status': statuses[attempt],
                'final': {'native_cut': cuts[attempt], 'block_loads': [2, 2], 'feasible': True}}
    return run


def test_over_budget_better_solution_cannot_win():
    clock = Clock()
    runs = collect_restarts(runner(clock, [.5, .75], [8., 1.]), 1., clock=clock)
    choose_restart_winner(runs)
    assert runs['attempts'] == 2
    assert runs['eligible_completed_runs'] == 1
    assert runs['winner']['native_cut'] == 8.
    assert runs['records'][1]['completion_seconds'] == 1.25
    assert not runs['records'][1]['completed_within_budget']
    assert runs['overshoot_seconds'] == .25


def test_first_run_too_slow_is_no_result_not_borrowed_candidate():
    clock = Clock()
    runs = collect_restarts(runner(clock, [1.25], [1.]), 1., clock=clock)
    choose_restart_winner(runs)
    assert runs['status'] == 'no_result'
    assert runs['winner_run_id'] is None and runs['winner'] is None
    assert runs['eligible_completed_runs'] == 0
    assert len(runs['records']) == 1


def test_exact_deadline_completion_counts_and_no_extra_run_starts():
    clock = Clock()
    runs = collect_restarts(runner(clock, [.25, .75], [7., 3.]), 1., clock=clock)
    choose_restart_winner(runs)
    assert runs['attempts'] == 2
    assert runs['eligible_completed_runs'] == 2
    assert runs['records'][-1]['completion_seconds'] == 1.
    assert runs['winner']['native_cut'] == 3.
    assert runs['overshoot_seconds'] == 0.


def test_failures_consume_budget_and_are_retained_but_cannot_win():
    clock = Clock()
    runs = collect_restarts(runner(clock, [.25, .5, .5], [0., 8., 2.],
                            statuses=['failed', 'success', 'success']), 1., clock=clock)
    choose_restart_winner(runs)
    assert runs['attempts'] == 3
    assert runs['records'][0]['completed_within_budget']
    assert runs['records'][1]['completion_seconds'] == .75
    assert runs['eligible_completed_runs'] == 1
    assert runs['winner']['native_cut'] == 8.
    assert runs['overshoot_seconds'] == .25


def test_validation_failure_cannot_be_selected():
    clock = Clock()
    runs = collect_restarts(runner(clock, [.5, .5], [2., 8.]), 1., clock=clock)
    runs['records'][0]['status'] = 'validation_failed'
    choose_restart_winner(runs)
    assert runs['winner']['native_cut'] == 8.


def test_zero_budget_does_not_run_and_is_no_result():
    clock = Clock()
    runs = collect_restarts(lambda _: pytest.fail('zero budget must not start a pipeline'), 0., clock=clock)
    choose_restart_winner(runs)
    assert runs['status'] == 'no_result' and runs['attempts'] == 0


def test_assignment_artifact_references_survive_serialization(tmp_path):
    coarse = {'original_to_coarse': np.array([0, 0, 1, 1]), 'coarse_node_weights': np.array([2., 2.])}
    run = {'run_id': 'single_fem', 'status': 'success',
           'coarse_assignment': np.array([0, 1]), 'final_assignment': np.array([0, 1, 0, 1])}
    unit = {'single_runs': {'fem': run}}
    persist_unit(tmp_path, unit, coarse)
    saved = json.loads((tmp_path / 'runs.json').read_text())['single_runs']['fem']
    assert 'coarse_assignment' not in saved and 'final_assignment' not in saved
    arrays = np.load(tmp_path / 'assignments.npz')
    for field in ('coarse_assignment', 'final_assignment'):
        ref = saved['assignment_artifacts'][field]
        assert ref['file'] == 'assignments.npz'
        np.testing.assert_array_equal(arrays[ref['key']], run[field])


def test_measure_accepts_actual_torch_coarse_weights():
    record = measure(np.array([0, 1]), [[0, 1]],
                     torch.tensor([2., 2.]), torch.tensor([3.]), 2, 0.)
    assert record['native_cut'] == 3.
    assert record['capacity'] == 2.
    assert record['feasible']
    assert record['independent_cut_and_loads_verified']
    np.testing.assert_array_equal(record['block_loads'], [2., 2.])
