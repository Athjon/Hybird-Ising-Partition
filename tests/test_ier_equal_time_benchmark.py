"""Control, timing-boundary and artifact tests; no formal timed experiment."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.hypergraph import compare_ier_equal_time as bench


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, value):
        self.now += value


def fallback(clock, duration=.125, cut=10.):
    def prepare():
        clock.advance(duration)
        return {'assignment': np.array([0, 0, 1, 1]),
                'native_metrics': {'native_cut': cut, 'feasible': True, 'block_loads': [2., 2.]}}
    return prepare


def fake_runner(clock, durations, cuts, *, statuses=None, states=None, calls=None):
    calls = [] if calls is None else calls
    statuses = ['success'] * len(durations) if statuses is None else statuses

    def run(kind, seed, continuation):
        i = len(calls)
        initial = np.array([0, 0, 1, 1]) if continuation is None else continuation.copy()
        calls.append({'kind': kind, 'seed': seed, 'continuation': None if continuation is None else continuation.copy()})
        clock.advance(durations[i])
        final = np.array([0, 1, 0, 1]) if states is None else np.asarray(states[i]).copy()
        return {'status': statuses[i], 'initial_assignment': initial, 'final_assignment': final,
                'native_metrics': {'native_cut': cuts[i], 'feasible': True, 'block_loads': [2., 2.]}}
    return run


def mark_fake_verified(result):
    if 'fallback' in result:
        result['fallback']['verified'] = result['fallback']['native_metrics'].copy()
    for record in result['records']:
        if record['status'] == 'success':
            record['final'] = record['native_metrics'].copy()
    bench.choose_checkpoints(result)


def test_shared_checkpoints_are_from_one_trace_and_overshoot_cannot_win():
    clock, calls = Clock(), []
    run = fake_runner(clock, [.25, .5, .5], [8., 3., 1.], calls=calls)
    result = bench.collect_arm('fem', 30, [.25, .75, 1.25], fallback(clock), run, clock=clock)
    mark_fake_verified(result)
    assert [r['completion_seconds'] for r in result['records']] == [.375, .875, 1.375]
    points = result['checkpoints']
    assert [p['optimized_status'] for p in points] == ['no_result', 'success', 'success']
    assert [p['eligible_completed_vcycles'] for p in points] == [0, 1, 2]
    assert all(p['eligible_flow_continuations'] == 0 for p in points)
    assert [p['best_completed']['native_cut'] if p['best_completed'] else None for p in points] == [None, 8., 3.]
    assert points[0]['best_available']['source'] == 'initial_only'
    assert points[0]['best_available']['native_cut'] == 10.
    assert result['records'][-1]['completed_within_max_budget'] is False
    assert result['overshoot_seconds'] == .125
    assert [c['seed'] for c in calls] == [30, 30 + 1000003, 30 + 2 * 1000003]
    assert all(c['kind'] == 'vcycle' and c['continuation'] is None for c in calls)


def test_exact_deadline_counts_and_does_not_start_another_attempt():
    clock = Clock()
    result = bench.collect_arm('pairs', 31, [.75], fallback(clock, .25),
        fake_runner(clock, [.5], [8.]), clock=clock)
    mark_fake_verified(result)
    assert len(result['records']) == 1
    assert result['records'][0]['completion_seconds'] == .75
    assert result['checkpoints'][0]['best_completed']['native_cut'] == 8.


def test_first_full_return_after_deadline_is_no_result_with_separate_initial_only():
    clock = Clock()
    result = bench.collect_arm('fem', 30, [.5], fallback(clock),
        fake_runner(clock, [.5], [1.]), clock=clock)
    mark_fake_verified(result)
    point = result['checkpoints'][0]
    assert point['optimized_status'] == 'no_result' and point['best_completed'] is None
    assert point['best_available']['source'] == 'initial_only'
    assert point['best_available']['native_cut'] == 10.


def test_fallback_itself_is_not_free_or_available_before_scoring_completes():
    clock = Clock()
    result = bench.collect_arm('fem', 30, [.25, .5], fallback(clock, .75),
        lambda *_: pytest.fail('late fallback leaves no time to launch'), clock=clock)
    mark_fake_verified(result)
    assert result['fallback']['available_seconds'] == .75
    assert result['records'] == []
    assert all(p['best_available'] is None and p['optimized_status'] == 'no_result' for p in result['checkpoints'])


def test_scoring_and_incumbent_comparisons_precede_availability():
    clock = Clock()

    class Cost(float):
        def __lt__(self, other):
            clock.advance(.0625)
            return float(self) < other

    def once(*_):
        clock.advance(.25)  # solver work
        clock.advance(.125)  # production native scoring, still inside run_once
        return {'status': 'success', 'initial_assignment': np.array([0, 0, 1, 1]),
                'final_assignment': np.array([0, 1, 0, 1]),
                'native_metrics': {'native_cut': Cost(1.), 'feasible': True, 'block_loads': [2., 2.]}}

    result = bench.collect_arm('fem', 30, [.5], fallback(clock, .125), once, clock=clock)
    # Four comparisons are currently required for the two incumbent updates.
    assert result['records'][0]['completion_seconds'] > .5
    mark_fake_verified(result)
    assert result['checkpoints'][0]['optimized_status'] == 'no_result'


def test_failures_consume_budget_and_are_retained_but_never_win():
    clock = Clock()
    result = bench.collect_arm('deterministic', 30, [1.], fallback(clock, 0.),
        fake_runner(clock, [.25, .5, .5], [0., 8., 1.], statuses=['failed', 'success', 'success']), clock=clock)
    mark_fake_verified(result)
    assert len(result['records']) == 3
    assert result['records'][0]['status'] == 'failed'
    assert result['checkpoints'][0]['best_completed']['native_cut'] == 8.


def test_external_validation_failure_excludes_a_completed_candidate():
    clock = Clock()
    result = bench.collect_arm('pairs', 30, [1.], fallback(clock, 0.),
        fake_runner(clock, [.5, .5], [1., 8.]), clock=clock)
    result['records'][0]['status'] = 'validation_failed'
    mark_fake_verified(result)
    assert result['checkpoints'][0]['best_completed']['native_cut'] == 8.


def test_flow_continues_its_own_final_and_requires_assignment_not_cut_fixed_point():
    clock, calls = Clock(), []
    states = [[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]]
    result = bench.collect_arm('flow', 30, [2.], fallback(clock, 0.),
        fake_runner(clock, [.25, .25, .25], [1., 1., 1.], states=states, calls=calls), clock=clock)
    mark_fake_verified(result)
    assert [c['kind'] for c in calls] == ['vcycle', 'flow_continuation', 'flow_continuation']
    assert calls[0]['continuation'] is None
    np.testing.assert_array_equal(calls[1]['continuation'], states[0])
    np.testing.assert_array_equal(calls[2]['continuation'], states[1])
    assert [r['exact_assignment_fixed_point'] for r in result['records']] == [False, False, True]
    assert result['stopped_reason'] == 'exact_assignment_fixed_point'
    point = result['checkpoints'][0]
    assert point['eligible_completed_vcycles'] == 1 and point['eligible_flow_continuations'] == 2


def test_flow_first_failure_never_starts_a_foreign_or_missing_continuation():
    clock = Clock()
    result = bench.collect_arm('flow', 30, [1.], fallback(clock, 0.),
        fake_runner(clock, [.25], [0.], statuses=['failed']), clock=clock)
    assert len(result['records']) == 1 and result['stopped_reason'] == 'flow_failed'


@pytest.mark.parametrize('arm', bench.ARMS)
def test_pool_frontend_parameters_and_mutation_isolated_restarts(monkeypatch, arm):
    starts, options = [], []

    class FakeRecorder:
        def __init__(self, graph_ids):
            self.stages = []

        def update_params(self, **params):
            options.append(params)

    def fake_vcycle(start, hierarchy, edges, q, refiner, **kwargs):
        starts.append(start.copy())
        start[[0, 2]] = start[[2, 0]]
        return start

    monkeypatch.setattr(bench, 'StateRecorder', FakeRecorder)
    monkeypatch.setattr(bench, 'vcycle_uncoarsen', fake_vcycle)
    original = np.array([0, 0, 1, 1])
    coarse = {'original_to_coarse': np.arange(4), 'hierarchy_stack': []}
    for seed in [30, 30 + 1000003]:
        record = bench.run_attempt(arm, 'vcycle', seed, None, original, coarse, [], np.ones(4), np.ones(0), 2, 0.)
        assert record['status'] == 'success'
    np.testing.assert_array_equal(original, [0, 0, 1, 1])
    for start in starts:
        np.testing.assert_array_equal(start, original)
    for i, p in enumerate(options):
        assert p['seed'] == 30 + i * 1000003
        assert p['ier_rounds'] == 2 and p['ier_max_moves'] == 24 and p['ier_boundary_pool'] == 96
        assert p['ier_pool_strategy'] == 'local'
        assert p['ier_num_trials'] == 8 and p['ier_num_steps'] == 100 and p['ier_random_samples'] == 800
        assert p['flow_passes'] == 2
        assert p['mode_cycle'] == (('flow',) if arm == 'flow' else ('fem_ier', 'flow'))
        assert p['ier_backend'] == ('fem' if arm == 'flow' else arm)


def test_arm_order_rotates_and_balances_positions():
    orders = [bench.arm_order(i) for i in range(10)]
    for arm in bench.ARMS:
        counts = [sum(order[position] == arm for order in orders) for position in range(4)]
        assert max(counts) - min(counts) <= 1
    assert len(set(orders[:4])) == 4


def test_state_recorder_keeps_initial_final_and_distinct_diagnostics(monkeypatch):
    def fake_refine(self, assignment, *args, **kwargs):
        assignment[[0, 2]] = assignment[[2, 0]]
        self.last_result = {'serial': len(self.stages)}
        return assignment

    monkeypatch.setattr(bench.HyperRefineSolver, 'refine', fake_refine)
    recorder = bench.StateRecorder(['level_0', 'original'])
    start = np.array([0, 0, 1, 1])
    recorder.refine(start, [], 2)
    recorder.refine(start, [], 2)
    np.testing.assert_array_equal(recorder.stages[0]['initial_assignment'], [0, 0, 1, 1])
    np.testing.assert_array_equal(recorder.stages[0]['final_assignment'], [1, 0, 0, 1])
    assert recorder.stages[0]['last_result'] == {'serial': 1}
    assert recorder.stages[1]['last_result'] == {'serial': 2}


def test_recursive_artifacts_keep_fallback_attempt_and_every_stage_state(tmp_path):
    initial, final = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    value = {'fallback': {'assignment': initial}, 'records': [{'coarse_assignment': np.array([0, 1]),
        'initial_assignment': initial, 'final_assignment': final, 'stages': [
            {'initial_assignment': initial.copy(), 'final_assignment': final.copy(),
             'last_result': {'stages': [{'history': [{'moves': [{'0': 1, '1': 0}]}]}]}}]}]}
    bench.persist_bundle(tmp_path, 'fem', value)
    saved = json.loads((tmp_path / 'fem.json').read_text())
    with np.load(tmp_path / 'fem.npz') as arrays:
        refs = [(saved['fallback']['assignment'], initial),
                (saved['records'][0]['stages'][0]['initial_assignment'], initial),
                (saved['records'][0]['stages'][0]['final_assignment'], final)]
        for ref, expected in refs:
            assert ref['artifact'] == 'fem.npz'
            np.testing.assert_array_equal(arrays[ref['key']], expected)
    with pytest.raises(FileExistsError):
        bench.persist_bundle(tmp_path, 'fem', value)


def test_hierarchy_artifact_has_all_weighted_graphs_and_projection_maps():
    level = {'num_nodes': 4, 'remap': {0: 0, 1: 0, 2: 1, 3: 1},
             'groups': [[0], [1], [2], [3]], 'node_weights': np.ones(4),
             'hyperedges': [[0, 2], [1, 3]], 'hyperedge_weights': np.array([2., 3.])}
    coarse = {'original_to_coarse': np.array([0, 0, 1, 1]), 'hierarchy_stack': [level],
              'coarse_node_weights': np.array([2., 2.]), 'coarse_hyperedges': [[0, 1]],
              'coarse_hyperedge_weights': np.array([5.])}
    graphs = bench.graph_data(coarse, level['hyperedges'], level['node_weights'], level['hyperedge_weights'])
    packed = bench.hierarchy_artifact(np.array([0, 1]), coarse, graphs)
    assert set(packed['graphs']) == {'original', 'level_0'}
    np.testing.assert_array_equal(packed['hierarchy_stack'][0]['remap'], [0, 0, 1, 1])
    np.testing.assert_array_equal(packed['graphs']['level_0']['hyperedges']['offsets'], [0, 2, 4])
    np.testing.assert_array_equal(packed['graphs']['level_0']['hyperedge_weights'], [2., 3.])


def test_protocol_and_all_new_driver_sources_are_hashed():
    sources = bench.source_hashes()
    assert 'benchmarks/hypergraph/IER_EQUAL_TIME_PROTOCOL.md' in sources
    assert 'benchmarks/hypergraph/compare_ier_equal_time.py' in sources
    assert 'src/partition/hyper_ier.py' in sources


@pytest.mark.parametrize('arm', ['pairs', 'deterministic', 'flow'])
def test_real_tiny_vcycle_stage_projection_audit_and_failure_reporting(arm):
    edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2, 4], [1, 3, 5], [2, 6], [3, 7]]
    nodes, weights = np.ones(8), np.arange(1., 9.)
    solver = bench.KahyparLikeSolver()
    coarse = solver.coarsen(edges, 8, 2, coarsen_to=4, seed=30, score_mode='hem',
        enforce_balance_cap=True, epsilon=0., node_weights=nodes, hyperedge_weights=weights)
    start = solver.initial_partition_greedy(coarse['coarse_hyperedges'], coarse['coarse_node_weights'],
        2, seed=30, epsilon=0., hyperedge_weights=coarse['coarse_hyperedge_weights'])
    clock = Clock()

    def prepare():
        lifted = np.asarray(start)[coarse['original_to_coarse']]
        return {'assignment': lifted.copy(),
                'native_metrics': bench.production_metrics(lifted, edges, nodes, weights, 2, 0.)}

    def once(kind, seed, continuation):
        record = bench.run_attempt(arm, kind, seed, continuation, start, coarse, edges, nodes, weights, 2, 0.)
        clock.advance(.25)
        return record

    result = bench.collect_arm(arm, 30, [.25], prepare, once, clock=clock)
    graphs = bench.graph_data(coarse, edges, nodes, weights)
    bench.verify_arm(result, start, coarse, graphs, 2, 0.)
    assert result['has_failures'] is False
    assert result['validation_summary']['successful_runs'] == 1
    assert result['checkpoints'][0]['optimized_status'] == 'success'
    record = result['records'][0]
    assert len(record['stages']) == len(coarse['hierarchy_stack']) + 1
    assert record['independent_stage_and_projection_verification']
    # A corrupted saved projection must invalidate the complete result.
    record['stages'][0]['initial_assignment'] = 1 - record['stages'][0]['initial_assignment']
    bench.verify_arm(result, start, coarse, graphs, 2, 0.)
    assert result['has_failures'] is True
    assert result['validation_summary']['validation_failed_runs'] == 1
    assert result['checkpoints'][0]['optimized_status'] == 'no_result'


def test_outer_restart_seed_stride_does_not_reuse_adjacent_inner_round_seed():
    inner_stride = 104729
    assert bench.SEED_STRIDE == 1000003
    seeds = [30 + bench.SEED_STRIDE * attempt + inner_stride * round_index
             for attempt in range(20) for round_index in range(2)]
    assert len(set(seeds)) == len(seeds)
