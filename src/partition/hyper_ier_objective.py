"""Exact native km1 expectation over disjoint, balanced joint-move atoms.

One binary selector controls an entire move, including all of that move's
pins in a hyperedge. Pins controlled by one selector are correlated. Only
different selectors are independent under the categorical FEM distribution.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from numbers import Integral

import numpy as np
import torch

from src.partition.hyper_objective import capacity_limits


def _index(value, name, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < 0 or value >= upper):
        raise ValueError(f'{name} must be an integer in [0, {upper})')
    return int(value)


def _weights(values, size, name):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    result = np.ones(size, dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f'{name} must contain {size} finite nonnegative weights')
    result = result.copy()
    result.flags.writeable = False
    return result


class JointMoveObjective:
    """Native weighted hypergraph cost for a set of balanced joint moves.

    ``num_nodes`` is the number of original vertices, while ``num_moves``
    (also ``num_variables``) is the number of binary FEM variables. A move
    maps each changed vertex to its new original partition label. Vertices
    cannot occur in two moves and no move entry may be a no-op.

    Each atom preserves every block's load within the same relative numeric
    tolerance as the production capacity constraint. A second, conservative
    bound checks that these tiny residuals cannot accumulate into a capacity
    violation under any subset of selectors. Integer/equal-weight atoms have
    exactly zero residual and trivially satisfy this extra check.

    ``expectation(_, p)`` accepts (batch, num_moves, 2) categorical probabilities.
    ``energy(_, selectors)`` accepts hard binary (batch, num_moves) selectors,
    or categorical configurations whose last axis is decoded by argmax.
    ``apply(selectors)`` accepts one hard selector vector and returns a full
    original-vertex assignment. An empty move set is legal and constant.
    """

    def __init__(self, assignment, hyperedges, moves, q, node_weights=None,
                 hyperedge_weights=None, epsilon=.03):
        if isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral) or q < 2:
            raise ValueError('q must be an integer >= 2')
        self.q = int(q)
        if torch.is_tensor(assignment):
            assignment = assignment.detach().cpu().numpy()
        labels = np.asarray(assignment)
        if labels.ndim != 1 or (labels.size and not np.issubdtype(labels.dtype, np.integer)):
            raise ValueError('assignment must be a one-dimensional integer array')
        if np.any(labels < 0) or np.any(labels >= self.q):
            raise ValueError('assignment labels must lie in [0, q)')
        self.assignment = labels.astype(np.int64, copy=True)
        self.assignment.flags.writeable = False
        self.num_nodes = len(self.assignment)
        self.node_weights = _weights(node_weights, self.num_nodes, 'node_weights')
        self.hyperedge_weights = _weights(hyperedge_weights, len(hyperedges), 'hyperedge_weights')
        self.epsilon = float(epsilon)
        self.capacity, self.tolerance = capacity_limits(self.node_weights, self.q, self.epsilon)
        self.initial_loads = self._loads(self.assignment)
        if np.any(self.initial_loads > self.capacity + self.tolerance):
            raise ValueError('initial assignment exceeds upper block capacity')

        self.hyperedges = tuple(tuple(sorted({_index(v, 'hyperedge vertex', self.num_nodes)
                                             for v in edge})) for edge in hyperedges)
        vertex_to_move = np.full(self.num_nodes, -1, dtype=np.int64)
        on_labels = self.assignment.copy()
        self._moves = []
        load_deltas = []
        for move_id, move in enumerate(moves):
            if not isinstance(move, Mapping) or not move:
                raise ValueError('each move must be a nonempty vertex-to-label mapping')
            entries = []
            terms = [[] for _ in range(self.q)]
            for vertex, target in move.items():
                vertex = _index(vertex, 'move vertex', self.num_nodes)
                target = _index(target, 'move target label', self.q)
                old = int(self.assignment[vertex])
                if old == target:
                    raise ValueError('every move entry must actually change its vertex label')
                if vertex_to_move[vertex] >= 0:
                    raise ValueError('move vertices must be disjoint across all atoms')
                vertex_to_move[vertex] = move_id
                on_labels[vertex] = target
                entries.append((vertex, target))
                weight = float(self.node_weights[vertex])
                terms[old].append(-weight)
                terms[target].append(weight)
            delta = np.asarray([math.fsum(parts) for parts in terms])
            if np.any(np.abs(delta) > self.tolerance):
                raise ValueError('each move must preserve every block load within capacity tolerance')
            load_deltas.append(delta)
            self._moves.append(tuple(sorted(entries)))
        self._moves = tuple(self._moves)
        self.num_moves = len(self._moves)
        self.num_variables = self.num_moves
        self.move_load_deltas = np.asarray(load_deltas, dtype=np.float64).reshape(self.num_moves, self.q)
        worst_loads = self.initial_loads + np.asarray([
            math.fsum(float(delta[block]) for delta in self.move_load_deltas if delta[block] > 0)
            for block in range(self.q)
        ])
        if np.any(worst_loads > self.capacity + self.tolerance):
            raise ValueError('move load residuals can accumulate above capacity for a selector subset')
        self.worst_subset_loads = worst_loads

        # Group by number of *distinct atoms* touching an edge. The cached
        # shapes are (edges_in_group, touched_atoms[, original_q]), never E*M.
        groups = defaultdict(list)
        constants = []
        for edge, weight in zip(self.hyperedges, self.hyperedge_weights):
            if len(edge) < 2:
                continue
            controlled = defaultdict(list)
            fixed_absence = np.ones(self.q, dtype=bool)
            for vertex in edge:
                atom = int(vertex_to_move[vertex])
                if atom < 0:
                    fixed_absence[self.assignment[vertex]] = False
                else:
                    controlled[atom].append(vertex)
            if not controlled:
                constants.append(float(weight) * (len({int(self.assignment[v]) for v in edge}) - 1))
                continue
            atoms = sorted(controlled)
            off_absence = np.ones((len(atoms), self.q), dtype=bool)
            on_absence = np.ones((len(atoms), self.q), dtype=bool)
            for index, atom in enumerate(atoms):
                pins = controlled[atom]
                off_absence[index, self.assignment[pins]] = False
                on_absence[index, on_labels[pins]] = False
            groups[len(atoms)].append((atoms, fixed_absence, off_absence, on_absence, float(weight)))
        self.constant_cost = math.fsum(constants)
        if not math.isfinite(self.constant_cost):
            raise ValueError('constant native cost must be finite')
        self._groups = []
        for touched in sorted(groups):
            rows = groups[touched]
            self._groups.append({
                'indices': np.asarray([row[0] for row in rows], dtype=np.int64),
                'fixed_absence': np.asarray([row[1] for row in rows], dtype=bool),
                'off_absence': np.asarray([row[2] for row in rows], dtype=bool),
                'on_absence': np.asarray([row[3] for row in rows], dtype=bool),
                'weights': np.asarray([row[4] for row in rows], dtype=np.float64),
            })
        self._cache = {}

    @property
    def moves(self):
        return [dict(entries) for entries in self._moves]

    def interaction_summary(self):
        """Count hyperedges by distinct controlling atoms, not polynomial degree.

        Only edges with at least two distinct pins enter the histogram; empty
        and singleton nets have identically zero km1 contribution. An edge
        touching t atoms has interaction *potential* bounded by t, but its
        actual selector polynomial may have lower degree or be constant due
        to label-count cancellations or fixed pins. Zero-weight edges are
        included in the first histogram and excluded from the second.
        """
        total_nontrivial = sum(len(edge) > 1 for edge in self.hyperedges)
        positive_nontrivial = sum(len(edge) > 1 and weight > 0
                                  for edge, weight in zip(self.hyperedges, self.hyperedge_weights))
        histogram = {0: total_nontrivial}
        positive_histogram = {0: positive_nontrivial}
        for group in self._groups:
            touched = int(group['indices'].shape[1])
            count = len(group['indices'])
            positive_count = int(np.count_nonzero(group['weights'] > 0))
            histogram[touched] = histogram.get(touched, 0) + count
            positive_histogram[touched] = positive_histogram.get(touched, 0) + positive_count
            histogram[0] -= count
            positive_histogram[0] -= positive_count

        def buckets(counts):
            return {'0': int(counts.get(0, 0)), '1': int(counts.get(1, 0)),
                    '2': int(counts.get(2, 0)),
                    '3+': int(sum(count for touched, count in counts.items() if touched >= 3))}

        return {
            'num_moves': self.num_moves,
            'total_hyperedges': len(self.hyperedges),
            'nontrivial_hyperedges': total_nontrivial,
            'excluded_empty_or_singleton_hyperedges': len(self.hyperedges) - total_nontrivial,
            'edges_touching_atoms': buckets(histogram),
            'positive_weight_edges_touching_atoms': buckets(positive_histogram),
            'exact_touched_atom_histogram': {str(key): int(value) for key, value in sorted(histogram.items())},
            'max_atoms_touching_one_hyperedge': max(histogram),
            'interpretation': 'interaction potential upper bounds only; not actual polynomial degree or nonzero interaction counts',
        }

    def _loads(self, labels):
        return np.asarray([math.fsum(float(w) for w, label in zip(self.node_weights, labels)
                                     if label == block) for block in range(self.q)])

    def _tensors(self, p):
        key = (p.device, p.dtype)
        if key not in self._cache:
            self._cache[key] = [{
                'indices': torch.as_tensor(group['indices'], device=p.device, dtype=torch.long),
                'fixed_absence': torch.as_tensor(group['fixed_absence'], device=p.device),
                'off_absence': torch.as_tensor(group['off_absence'], device=p.device),
                'on_absence': torch.as_tensor(group['on_absence'], device=p.device),
                'weights': torch.as_tensor(group['weights'], device=p.device, dtype=p.dtype),
            } for group in self._groups]
        return self._cache[key]

    def _check_probability_shape(self, p):
        if (not torch.is_tensor(p) or p.ndim != 3
                or tuple(p.shape[1:]) != (self.num_moves, 2)):
            raise ValueError(f'expected selector probabilities shaped (batch, {self.num_moves}, 2)')
        if not p.is_floating_point():
            raise ValueError('selector probabilities must have floating dtype')

    def expectation(self, _, p):
        self._check_probability_shape(p)
        value = p.sum(dim=(1, 2)) * 0 + self.constant_cost
        for group in self._tensors(p):
            # Each r is shared by every pin controlled by this atom.
            r = p[:, group['indices'], 1, None]
            absence_factors = ((1 - r) * group['off_absence'][None]
                              + r * group['on_absence'][None])
            absence = group['fixed_absence'][None] * absence_factors.prod(dim=2)
            edge_cost = (1 - absence).sum(dim=-1) - 1
            value = value + (edge_cost * group['weights']).sum(dim=-1)
        return value

    def inference(self, _, p):
        self._check_probability_shape(p)
        return p.argmax(dim=-1)

    def _hard_selectors(self, selectors, one_dimensional=False):
        tensor = selectors if torch.is_tensor(selectors) else torch.as_tensor(selectors)
        if tensor.ndim == 3 and not one_dimensional:
            if tuple(tensor.shape[1:]) != (self.num_moves, 2):
                raise ValueError('categorical selector configurations have the wrong shape')
            tensor = tensor.argmax(dim=-1)
        if one_dimensional:
            if tensor.ndim != 1 or len(tensor) != self.num_moves:
                raise ValueError(f'apply expects one selector vector of length {self.num_moves}')
        else:
            if tensor.ndim == 1:
                tensor = tensor[None]
            if tensor.ndim != 2 or tensor.shape[1] != self.num_moves:
                raise ValueError(f'energy expects (batch, {self.num_moves}) binary selectors')
        if torch.any((tensor != 0) & (tensor != 1)):
            raise ValueError('hard selectors must be binary')
        return tensor.to(dtype=torch.long)

    def energy(self, _, selectors):
        selectors = self._hard_selectors(selectors)
        probabilities = torch.nn.functional.one_hot(selectors, num_classes=2).to(dtype=torch.float64)
        return self.expectation(None, probabilities)

    def apply(self, selectors):
        selected = self._hard_selectors(selectors, one_dimensional=True).detach().cpu().numpy()
        result = self.assignment.copy()
        for atom in np.flatnonzero(selected):
            for vertex, label in self._moves[int(atom)]:
                result[vertex] = label
        if np.any(self._loads(result) > self.capacity + self.tolerance):
            raise AssertionError('validated joint-move subset violated capacity')
        return result

    def exact_best(self, max_moves=20):
        """Completely enumerate a small selector space in bounded batches.

        Returns ``selectors``, full ``assignment``, native ``cost``, the number
        ``evaluated``, and ``complete=True``. Refuses, rather than samples, when
        num_moves exceeds max_moves. Ties choose the lowest integer bit mask.
        """
        if isinstance(max_moves, bool) or not isinstance(max_moves, Integral) or max_moves < 0:
            raise ValueError('max_moves must be a nonnegative integer')
        if self.num_moves > max_moves:
            raise ValueError('joint selector exact search exceeds max_moves')
        if self.num_moves >= 63:
            raise ValueError('exact selector enumeration supports fewer than 63 moves')
        total = 1 << self.num_moves
        largest = max((group['off_absence'].size for group in self._groups), default=1)
        batch_size = max(1, min(256, 1_000_000 // largest))
        best_cost, best_selectors = math.inf, None
        with torch.no_grad():
            for start in range(0, total, batch_size):
                codes = np.arange(start, min(total, start + batch_size), dtype=np.int64)
                selectors = (codes[:, None] >> np.arange(self.num_moves, dtype=np.int64)) & 1
                costs = self.energy(None, selectors).cpu().numpy()
                index = int(np.argmin(costs))
                if costs[index] < best_cost:
                    best_cost = float(costs[index])
                    best_selectors = selectors[index].copy()
        return {'selectors': best_selectors, 'assignment': self.apply(best_selectors),
                'cost': best_cost, 'evaluated': total, 'complete': True}
