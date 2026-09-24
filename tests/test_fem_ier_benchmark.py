"""IER benchmark control logic; no production or timed benchmark runs."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hypergraph import validate_fem_ier as bench


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


def test_a_b_receive_independent_identical_starts_and_explicit_selector_controls(monkeypatch):
    clock = Clock()
    calls = []

    class FakeRefiner:
        def update_params(self, **options):
            self.options = options

        def refine(self, initial, edges, q, **kwargs):
            calls.append((initial.copy(), self.options.copy()))
            initial[[0, 1]] = initial[[1, 0]]
            self.last_result = {'history': [{'candidate_count': 1}]}
            clock.now += .25
            return initial

    monkeypatch.setattr(bench, 'HyperRefineSolver', FakeRefiner)
    initial = np.array([0, 1, 0, 1])
    for arm in ('fem_ier', 'random'):
        record = bench.run_refinement(arm, initial, [], np.ones(4), np.ones(0), 2, 0., 30, 4, clock=clock)
        bench.verify_refinement(record, [], np.ones(4), np.ones(0), 2, 0.)
        assert record['status'] == 'success' and record['pipeline_seconds'] == .25
    np.testing.assert_array_equal(initial, [0, 1, 0, 1])
    for start, options in calls:
        np.testing.assert_array_equal(start, initial)
        assert options['ier_rounds'] == 4
        assert options['ier_num_trials'] == 8 and options['ier_num_steps'] == 100
        assert options['ier_random_samples'] == 800
        assert options['ier_max_moves'] == 24 and options['ier_boundary_pool'] == 96
        assert options['ier_pool_strategy'] == 'local'
        assert options['seed'] == 30
    assert calls[0][1]['ier_backend'] == 'fem'
    assert calls[1][1]['ier_backend'] == 'random'


@pytest.mark.parametrize('durations,budget,expected_status', [([.25, .5], .5, 'success'), ([.75], .5, 'no_result')])
def test_extra_flow_continues_only_its_own_state_and_excludes_overshoot(monkeypatch, durations, budget, expected_status):
    clock, starts = Clock(), []
    initial = np.array([0, 0, 1, 1])
    improved = np.array([0, 1, 0, 1])

    def fake_run(arm, start, edges, nodes, weights, q, epsilon, seed, rounds, **kwargs):
        attempt = len(starts)
        starts.append(start.copy())
        begin = clock()
        clock.now += durations[attempt]
        return {'arm': 'flow', 'initial_assignment': start.copy(), 'final_assignment': improved.copy(),
                'status': 'success', 'solver_seed': seed, 'started_at': begin,
                'finished_at': clock(), 'pipeline_seconds': durations[attempt], 'last_result': None}

    monkeypatch.setattr(bench, 'run_refinement', fake_run)
    result = bench.additional_flow(initial, [[0, 2], [1, 3]], np.ones(4), np.ones(2),
                                   2, 0., 30, 2, budget, clock=clock)
    assert result['status'] == expected_status
    np.testing.assert_array_equal(initial, [0, 0, 1, 1])
    np.testing.assert_array_equal(starts[0], initial)
    assert not result['records'][-1]['completed_within_budget']
    if len(starts) == 2:
        np.testing.assert_array_equal(starts[1], improved)
        assert result['eligible_completed_runs'] == 1
        assert result['winner_run_id'] == 'flow_0000'
        assert result['winner']['native_cut'] == 0.
    else:
        assert result['winner'] is None
        assert result['eligible_completed_runs'] == 0


def test_saved_coarse_assignment_cannot_use_a_different_hierarchy():
    coarse = {'original_to_coarse': np.array([0, 0, 1, 1]), 'coarse_node_weights': torch.tensor([2., 2.])}
    saved = {'mapping': np.array([0, 1, 0, 1]), 'coarse_nodes': np.array([2., 2.]), 'coarse_assignment': np.array([0, 1])}
    with pytest.raises(AssertionError, match='reconstructed hierarchy'):
        bench.check_hierarchy(coarse, saved)
    saved['mapping'] = np.array([0, 0, 1, 1])
    bench.check_hierarchy(coarse, saved)


def test_nested_initial_final_assignments_are_saved_completely(tmp_path):
    initial, final = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    case = {'initial_assignment': initial, 'arms': {'fem_ier': {'initial_assignment': initial.copy(),
            'final_assignment': final, 'last_result': {'history': [{'candidate_count': 3}]}}},
            'additional_flow': {'records': [{'initial_assignment': initial.copy(), 'final_assignment': final.copy()}]}}
    bench.persist(tmp_path, case)
    saved = json.loads((tmp_path / 'result.json').read_text())
    arrays = np.load(tmp_path / 'assignments.npz')
    refs = [(saved['initial_assignment'], initial),
            (saved['arms']['fem_ier']['final_assignment'], final),
            (saved['additional_flow']['records'][0]['initial_assignment'], initial)]
    for reference, expected in refs:
        assert reference['artifact'] == 'assignments.npz'
        np.testing.assert_array_equal(arrays[reference['key']], expected)
    assert saved['arms']['fem_ier']['last_result']['history'][0]['candidate_count'] == 3
