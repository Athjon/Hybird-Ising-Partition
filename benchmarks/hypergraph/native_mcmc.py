"""Native weighted-km1 Metropolis search inside the capacity-feasible set.

This research baseline does not alter the production solver. Its fixed-beta
kernel mixes state-independent proposals: a uniform node/target-label move,
a uniform ordered pair of distinct nodes to swap, and optionally a uniform
fixed-size subset on which two uniformly chosen labels are transposed.
Infeasible proposals are self-loops; there is no repair or resampling.

All proposals are symmetric, so ordinary Metropolis acceptance satisfies
Gibbs detailed balance at fixed finite beta. This does NOT establish
irreducibility, mixing, or equilibration. In particular, strict weighted
capacities may disconnect the single-move/pair-swap neighborhood. Annealing
histories and best-so-far states are optimization trajectories, not certified
Gibbs samples.

Example::

    chain = NativeChain(edges, node_weights, edge_weights, q, epsilon, start, seed=7)
    result = run_annealing(chain, 1000, 0.1, 5.0, block_probability=0.25,
                           block_sizes=(3,), history_stride=100)

Requires only NumPy and the standard library.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _weights(values, size, name):
    result = np.ones(size, dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f'{name} must contain {size} finite nonnegative weights')
    result = result.copy()
    result.flags.writeable = False
    return result


def _beta(beta):
    beta = float(beta)
    if math.isnan(beta) or beta < 0:
        raise ValueError('beta must be nonnegative; +inf is permitted for zero temperature')
    return beta


def metropolis_acceptance(delta, beta):
    """Acceptance probability; +inf beta accepts only downhill/flat moves."""
    beta = _beta(beta)
    delta = float(delta)
    if not math.isfinite(delta):
        raise ValueError('delta must be finite')
    if delta <= 0 or beta == 0:
        return 1.0
    return math.exp(-beta * delta)


@dataclass(frozen=True)
class Proposal:
    """Read-only proposal description, valid only at its originating state."""

    version: int
    updates: tuple
    new_loads: tuple
    edge_changes: tuple
    delta: float
    feasible: bool


class NativeChain:
    """Capacity-constrained Metropolis kernel with incident-edge delta updates.

    ``edge_weights=None`` means unit weights. The initial assignment must
    already be feasible. ``accepted`` counts actual state changes, excluding
    accepted mathematical self-loops caused by identical labels.
    """

    def __init__(self, edges, node_weights, edge_weights, q, epsilon, assignment, seed=1):
        self.q = _integer(q, 'q', minimum=2)
        labels = np.asarray(assignment)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('assignment must be a one-dimensional integer array')
        self.n = len(labels)
        if self.n < 1 or np.any(labels < 0) or np.any(labels >= self.q):
            raise ValueError('assignment must contain at least one node with labels in [0, q)')
        self._assignment = labels.astype(np.int64, copy=True)
        self.node_weights = _weights(node_weights, self.n, 'node_weights')
        self.edge_weights = _weights(edge_weights, len(edges), 'edge_weights')
        self.epsilon = float(epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError('epsilon must be finite and nonnegative')
        total = math.fsum(self.node_weights)
        self.capacity = (1 + self.epsilon) * total / self.q
        if not math.isfinite(total) or not math.isfinite(self.capacity):
            raise ValueError('total node weight and capacity must be finite')
        self.tolerance = 1e-10 * max(total, self.capacity)
        self._loads = np.bincount(self._assignment, weights=self.node_weights, minlength=self.q)
        if np.any(self._loads > self.capacity + self.tolerance):
            raise ValueError('initial assignment exceeds the upper block capacity')

        canonical = []
        self.incident = [[] for _ in range(self.n)]
        for edge_id, edge in enumerate(edges):
            vertices = sorted({_integer(v, 'hyperedge vertex') for v in edge})
            if any(v >= self.n for v in vertices):
                raise ValueError('hyperedge vertex id is outside [0, n)')
            canonical.append(tuple(vertices))
            for vertex in vertices:
                self.incident[vertex].append(edge_id)
        self.edges = tuple(canonical)
        self._edge_counts = np.zeros((len(self.edges), self.q), dtype=np.int64)
        for edge_id, edge in enumerate(self.edges):
            if edge:
                self._edge_counts[edge_id] = np.bincount(self._assignment[list(edge)], minlength=self.q)
        self._occupied = np.count_nonzero(self._edge_counts, axis=1)
        self._energy = self.native_energy()
        if not math.isfinite(self._energy):
            raise ValueError('native energy must be finite')
        self._energy_correction = 0.0
        self._best_energy = self._energy
        self._best_assignment = self._assignment.copy()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._version = 0
        self._counters = Counter({key: 0 for key in
                                  ('attempted', 'accepted', 'uphill_accepted', 'capacity_rejected',
                                   'metropolis_rejected', 'no_op', 'self_loops')})
        self._by_kind = {kind: self._counters.copy() for kind in ('move', 'swap', 'block')}

    @property
    def assignment(self):
        return self._assignment.copy()

    @property
    def loads(self):
        return self._loads.copy()

    @property
    def energy(self):
        return float(self._energy)

    @property
    def best_assignment(self):
        return self._best_assignment.copy()

    @property
    def best_energy(self):
        return float(self._best_energy)

    @property
    def stats(self):
        return {**dict(self._counters), 'by_kind': {kind: dict(counts)
                                                  for kind, counts in self._by_kind.items()}}

    def native_energy(self, assignment=None):
        """Independent full native evaluation, useful for audits and reporting."""
        labels = self._assignment if assignment is None else np.asarray(assignment)
        return math.fsum(float(weight) * (len({int(labels[v]) for v in edge}) - 1)
                         for edge, weight in zip(self.edges, self.edge_weights) if len(edge) > 1)

    def proposal(self, changes):
        """Evaluate an arbitrary joint ``{vertex: new_label}`` without mutation.

        Each touched edge is evaluated once, after accumulating *all* changed
        pins' label-count deltas. Shared hyperedges are never double-counted.
        This generic evaluator alone does not define a symmetric proposal
        distribution; ``step`` uses only the symmetric mechanisms documented.
        """
        updates = []
        for vertex, label in changes.items():
            vertex = _integer(vertex, 'vertex')
            label = _integer(label, 'target label')
            if vertex >= self.n or label >= self.q:
                raise ValueError('vertex or target label is out of range')
            if self._assignment[vertex] != label:
                updates.append((vertex, label))
        updates.sort()
        new_loads = self._loads.copy()
        edge_deltas = {}
        for vertex, new_label in updates:
            old_label = int(self._assignment[vertex])
            weight = self.node_weights[vertex]
            new_loads[old_label] -= weight
            new_loads[new_label] += weight
            for edge_id in self.incident[vertex]:
                counts = edge_deltas.setdefault(edge_id, Counter())
                counts[old_label] -= 1
                counts[new_label] += 1
        edge_changes = []
        contributions = []
        for edge_id, changes_by_label in sorted(edge_deltas.items()):
            occupied_change = 0
            clean = []
            for label, delta in sorted(changes_by_label.items()):
                if not delta:
                    continue
                before = int(self._edge_counts[edge_id, label])
                after = before + delta
                if after < 0:
                    raise AssertionError('negative hyperedge label count')
                occupied_change += int(after > 0) - int(before > 0)
                clean.append((label, delta))
            edge_changes.append((edge_id, tuple(clean), occupied_change))
            contributions.append(float(self.edge_weights[edge_id]) * occupied_change)
        delta = math.fsum(contributions)
        if not math.isfinite(delta):
            raise ValueError('native energy delta must be finite')
        feasible = bool(np.all(new_loads <= self.capacity + self.tolerance))
        return Proposal(self._version, tuple(updates), tuple(new_loads),
                        tuple(edge_changes), delta, feasible)

    def move_proposal(self, vertex, target_label):
        return self.proposal({vertex: target_label})

    def swap_proposal(self, first, second):
        first = _integer(first, 'first node')
        second = _integer(second, 'second node')
        if first >= self.n or second >= self.n:
            raise ValueError('swap node is out of range')
        return self.proposal({first: int(self._assignment[second]),
                              second: int(self._assignment[first])})

    def block_proposal(self, vertices, first_label, second_label):
        first_label = _integer(first_label, 'first label')
        second_label = _integer(second_label, 'second label')
        if first_label >= self.q or second_label >= self.q or first_label == second_label:
            raise ValueError('block transposition requires two distinct labels in [0, q)')
        selected = [_integer(v, 'block vertex') for v in vertices]
        if len(set(selected)) != len(selected) or any(v >= self.n for v in selected):
            raise ValueError('block vertices must be a distinct subset of [0, n)')
        changes = {}
        for vertex in selected:
            label = int(self._assignment[vertex])
            if label == first_label:
                changes[vertex] = second_label
            elif label == second_label:
                changes[vertex] = first_label
        return self.proposal(changes)

    def _apply(self, proposal):
        if proposal.version != self._version:
            raise ValueError('proposal is stale after a state change')
        if not proposal.feasible:
            raise ValueError('cannot apply a capacity-infeasible proposal')
        for vertex, label in proposal.updates:
            self._assignment[vertex] = label
        self._loads = np.asarray(proposal.new_loads, dtype=np.float64)
        for edge_id, changes, occupied_change in proposal.edge_changes:
            for label, delta in changes:
                self._edge_counts[edge_id, label] += delta
            self._occupied[edge_id] += occupied_change
        # Compensated accumulation limits drift. Acceptance uses the local
        # exact-native delta directly, never this accumulated score.
        corrected_delta = proposal.delta - self._energy_correction
        updated = self._energy + corrected_delta
        self._energy_correction = (updated - self._energy) - corrected_delta
        self._energy = updated
        self._version += 1
        if self._energy < self._best_energy:
            # Best-so-far reporting gets an independent full evaluation.
            exact = self.native_energy()
            self._energy, self._energy_correction = exact, 0.0
            if exact < self._best_energy:
                self._best_energy = exact
                self._best_assignment = self._assignment.copy()

    def _mixture(self, swap_probability, block_probability, block_sizes):
        swap_probability, block_probability = float(swap_probability), float(block_probability)
        if (not math.isfinite(swap_probability) or not math.isfinite(block_probability)
                or swap_probability < 0 or block_probability < 0
                or swap_probability + block_probability > 1):
            raise ValueError('swap_probability and block_probability must be nonnegative and sum to <= 1')
        if block_sizes is None:
            sizes = tuple(size for size in (2, 3, 4, 6) if size <= self.n)
            if not sizes:
                sizes = (1,)
        else:
            sizes = tuple(_integer(size, 'block size', minimum=1) for size in block_sizes)
            if not sizes or len(set(sizes)) != len(sizes) or any(size > self.n for size in sizes):
                raise ValueError('explicit block_sizes must be distinct integers in [1, n]')
        return swap_probability, block_probability, sizes

    def step(self, beta, swap_probability=.5, block_probability=0.0, block_sizes=None):
        """Attempt exactly one proposal; illegal/no-op/rejected proposals stay put.

        Move targets are uniform over *all* labels, including the current one.
        Swap pairs are uniform ordered distinct vertices (n=1 is a no-op).
        Block size is uniform over the supplied sizes; the subset is uniform
        and independent of current labels. Default sizes are 2,3,4,6 clipped
        to n (or 1 at n=1). Explicit out-of-range sizes raise.
        """
        beta = _beta(beta)
        swap_probability, block_probability, sizes = self._mixture(
            swap_probability, block_probability, block_sizes,
        )
        choice = self.rng.random()
        if choice < block_probability:
            kind = 'block'
            size = sizes[int(self.rng.integers(len(sizes)))]
            selected = self.rng.choice(self.n, size=size, replace=False)
            first = int(self.rng.integers(self.q))
            second = int(self.rng.integers(self.q - 1))
            second += second >= first
            proposal = self.block_proposal(selected, first, second)
        elif choice < block_probability + swap_probability:
            kind = 'swap'
            if self.n == 1:
                proposal = self.proposal({})
            else:
                first = int(self.rng.integers(self.n))
                second = int(self.rng.integers(self.n - 1))
                second += second >= first
                proposal = self.swap_proposal(first, second)
        else:
            kind = 'move'
            proposal = self.move_proposal(int(self.rng.integers(self.n)),
                                          int(self.rng.integers(self.q)))
        accepted = False
        if not proposal.feasible:
            reason = 'capacity_rejected'
        elif not proposal.updates:
            reason = 'no_op'
        elif self.rng.random() < metropolis_acceptance(proposal.delta, beta):
            self._apply(proposal)
            accepted, reason = True, 'accepted'
        else:
            reason = 'metropolis_rejected'
        for counts in (self._counters, self._by_kind[kind]):
            counts['attempted'] += 1
            counts[reason] += 1
            if accepted and proposal.delta > 0:
                counts['uphill_accepted'] += 1
            if not accepted:
                counts['self_loops'] += 1
        return {'kind': kind, 'accepted': accepted, 'reason': reason,
                'feasible': proposal.feasible, 'delta': proposal.delta,
                'changed_vertices': len(proposal.updates) if accepted else 0,
                'energy': self.energy, 'best_energy': self.best_energy}

    def assert_consistent(self):
        """Audit cached counts, loads and score against the current assignment."""
        counts = np.zeros_like(self._edge_counts)
        for edge_id, edge in enumerate(self.edges):
            if edge:
                counts[edge_id] = np.bincount(self._assignment[list(edge)], minlength=self.q)
        np.testing.assert_array_equal(counts, self._edge_counts)
        np.testing.assert_array_equal(np.count_nonzero(counts, axis=1), self._occupied)
        loads = np.bincount(self._assignment, weights=self.node_weights, minlength=self.q)
        np.testing.assert_allclose(loads, self._loads, rtol=1e-12, atol=self.tolerance)
        if np.any(loads > self.capacity + self.tolerance):
            raise AssertionError('current state violates capacity')
        np.testing.assert_allclose(self.energy, self.native_energy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(self.best_energy, self.native_energy(self._best_assignment),
                                   rtol=1e-12, atol=1e-12)


def run_annealing(chain, steps, beta_min, beta_max, *, swap_probability=.5,
                  block_probability=0.0, block_sizes=None, history_stride=100,
                  record_assignments=True):
    """Linear-beta optimization trajectory; return best/final states and logs.

    Constant finite endpoints give a fixed-beta run; both endpoints +inf give
    zero-temperature local search. A mixed finite/infinite schedule is invalid.
    Counters and best-so-far values include any earlier calls on this chain.
    ``history_stride=1`` records every state, including rejected self-loops.
    """
    steps = _integer(steps, 'steps')
    history_stride = _integer(history_stride, 'history_stride', minimum=1)
    beta_min, beta_max = _beta(beta_min), _beta(beta_max)
    if math.isinf(beta_min) != math.isinf(beta_max) or beta_max < beta_min:
        raise ValueError('beta endpoints must both be finite or both +inf, with beta_max >= beta_min')
    _, _, sizes = chain._mixture(swap_probability, block_probability, block_sizes)
    history = []

    def record(step, beta):
        row = {'step': step, 'beta': beta, 'energy': chain.native_energy(),
               'best_energy': chain.best_energy, 'loads': chain.loads.tolist(),
               'counters': dict(chain._counters)}
        if record_assignments:
            row['assignment'] = chain.assignment.tolist()
        history.append(row)

    record(0, beta_min)
    for index in range(steps):
        fraction = index / (steps - 1) if steps > 1 else 0.0
        beta = beta_min if math.isinf(beta_min) else beta_min + fraction * (beta_max - beta_min)
        chain.step(beta, swap_probability, block_probability, sizes)
        if (index + 1) % history_stride == 0 or index + 1 == steps:
            record(index + 1, beta)
    chain.assert_consistent()
    return {
        'best_assignment': chain.best_assignment, 'best_energy': chain.best_energy,
        'final_assignment': chain.assignment, 'final_energy': chain.native_energy(),
        'stats': chain.stats, 'history': history,
        'configuration': {'steps': steps, 'beta_min': beta_min, 'beta_max': beta_max,
                          'schedule': 'linear_beta', 'swap_probability': swap_probability,
                          'block_probability': block_probability, 'block_sizes': list(sizes),
                          'history_stride': history_stride, 'seed': chain.seed},
        'interpretation': 'optimization trajectory and best-so-far states; not certified Gibbs samples',
    }
