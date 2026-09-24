"""Differentiable native hypergraph objectives and checked capacity rounding."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch


class BalanceInfeasibleError(ValueError):
    """The specified upper capacities are provably infeasible."""


class BalanceSearchError(RuntimeError):
    """A bounded rounding search failed; feasibility is not decided."""


def _weights(values, size, name):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    result = np.ones(size, dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f'{name} must contain {size} finite nonnegative values')
    return result


def capacity_limits(node_weights, q, epsilon):
    if isinstance(q, bool) or int(q) != q or q < 1:
        raise ValueError('q must be a positive integer')
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and nonnegative')
    total = float(np.sum(node_weights))
    capacity = (1 + float(epsilon)) * total / int(q)
    if not np.isfinite(total) or not np.isfinite(capacity):
        raise ValueError('total node weight and block capacity must be finite')
    # Resource units may be arbitrarily small. An absolute floor of 1 here
    # would make tiny positive weights bypass the capacity constraint.
    tolerance = 1e-10 * max(total, capacity)
    return capacity, tolerance


def round_balanced_probabilities(probabilities, node_weights, q, epsilon=0.03,
                                 exact_max_nodes=20, search_budget=200000):
    """Return a capacity-feasible labeling, or an explicit failure.

    The constraint is max block load <= (1+epsilon)*total/q. Empty blocks
    are allowed. Greedy rounding is followed by bounded exact packing only
    on small inputs. Exhausting a budget NEVER implies infeasibility.
    """
    if torch.is_tensor(probabilities):
        probabilities = probabilities.detach().cpu().numpy()
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != q or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('probabilities must be a finite nonnegative (n, q) array')
    n = len(p)
    weights = _weights(node_weights, n, 'node_weights')
    cap, tol = capacity_limits(weights, q, epsilon)
    q = int(q)
    if exact_max_nodes < 0 or search_budget < 0:
        raise ValueError('exact_max_nodes and search_budget must be nonnegative')
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(p.sum(axis=1) <= 0):
        raise ValueError('every node must have positive probability mass')
    if np.any(weights > cap + tol):
        raise BalanceInfeasibleError('a node exceeds the maximum block capacity')

    def feasible(labels):
        return np.bincount(labels, weights=weights, minlength=q).max() <= cap + tol

    initial = p.argmax(axis=1).astype(np.int64)
    if feasible(initial):
        return initial
    scores = np.log(np.maximum(p, np.finfo(np.float64).tiny))

    # Equal-weight bisection admits an exact score-maximizing capacity sort.
    if q == 2 and weights[0] > 0 and np.all(weights == weights[0]):
        limit = min(n, math.floor((cap + tol) / weights[0]))
        lower = max(0, n - limit)
        if lower > limit:
            raise BalanceInfeasibleError('integer block capacities cannot hold all nodes')
        difference = scores[:, 1] - scores[:, 0]
        count = int(np.clip(np.count_nonzero(difference > 0), lower, limit))
        labels = np.zeros(n, dtype=np.int64)
        labels[np.argsort(-difference, kind='stable')[:count]] = 1
        if not feasible(labels):
            raise AssertionError('capacity rounding failed its postcondition')
        return labels

    confidence = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2] if q > 1 else np.zeros(n)
    order = np.lexsort((np.arange(n), -confidence, -weights))
    # Try preference, least-loaded and best-fit packing. A failed greedy
    # attempt is never a feasibility certificate.
    for strategy in ('probability', 'least_loaded', 'best_fit'):
        labels = np.full(n, -1, dtype=np.int64)
        loads = np.zeros(q)
        for vertex in order:
            allowed = np.flatnonzero(loads + weights[vertex] <= cap + tol)
            if not len(allowed):
                break
            if strategy == 'probability':
                block = max(allowed, key=lambda b: (scores[vertex, b], -loads[b]))
            elif strategy == 'least_loaded':
                block = min(allowed, key=lambda b: (loads[b], -scores[vertex, b]))
            else:
                block = max(allowed, key=lambda b: (loads[b], scores[vertex, b]))
            labels[vertex] = block
            loads[block] += weights[vertex]
        if np.all(labels >= 0) and feasible(labels):
            return labels

    if n > exact_max_nodes or search_budget == 0:
        raise BalanceSearchError('greedy rounding failed; exact packing was disabled or exceeds the size limit')
    labels = np.full(n, -1, dtype=np.int64)
    loads = np.zeros(q)
    failed = set()
    visited = 0

    def search(index):
        nonlocal visited
        visited += 1
        if visited > search_budget:
            raise BalanceSearchError('capacity search budget exhausted; feasibility is unknown')
        if index == n:
            return True
        key = (index, tuple(sorted(loads)))
        if key in failed:
            return False
        vertex = order[index]
        seen = set()
        for block in sorted(range(q), key=lambda b: (-scores[vertex, b], loads[b])):
            load = float(loads[block])
            if load in seen or load + weights[vertex] > cap + tol:
                continue
            seen.add(load)
            labels[vertex] = block
            loads[block] = load + weights[vertex]
            if search(index + 1):
                return True
            loads[block] = load
            labels[vertex] = -1
        failed.add(key)
        return False

    if not search(0):
        raise BalanceInfeasibleError('complete capacity search proved no feasible packing')
    if not feasible(labels):
        raise AssertionError('capacity search failed its postcondition')
    return labels


class HypergraphObjective:
    """Native km1 or an explicitly selected clique/star surrogate.

    All modes rank returned partitions by native km1. Star auxiliaries carry
    zero resource load and are removed before logical-node rounding.
    """

    def __init__(self, hyperedges, node_weights, q, *, hyperedge_weights=None,
                 imbalance_weight=5.0, map_type='native'):
        if torch.is_tensor(node_weights):
            node_weights = node_weights.detach().cpu().numpy()
        self.num_nodes = len(node_weights)
        self.node_weights = _weights(node_weights, self.num_nodes, 'node_weights')
        capacity_limits(self.node_weights, q, 0)
        self.balance_scale = float(self.node_weights.mean()) if self.num_nodes and self.node_weights.sum() > 0 else 1.0
        self.q = int(q)
        if self.q < 2:
            raise ValueError('FEM hypergraph partitioning requires q >= 2')
        if map_type not in ('native', 'clique', 'star'):
            raise ValueError("map_type must be 'native', 'clique', or 'star'")
        if not np.isfinite(imbalance_weight) or imbalance_weight < 0:
            raise ValueError('imbalance_weight must be finite and nonnegative')
        self.map_type = map_type
        self.imbalance_weight = float(imbalance_weight)
        self.hyperedge_weights = _weights(hyperedge_weights, len(hyperedges), 'hyperedge_weights')
        self.hyperedges = []
        groups = defaultdict(list)
        auxiliary = self.num_nodes
        for edge, weight in zip(hyperedges, self.hyperedge_weights):
            raw = list(edge)
            if any(isinstance(v, bool) or int(v) != v or v < 0 or v >= self.num_nodes for v in raw):
                raise ValueError('hyperedge vertex ids must be integers in [0, num_nodes)')
            canonical = sorted(set(int(v) for v in raw))
            self.hyperedges.append(canonical)
            if len(canonical) > 1:
                groups[len(canonical)].append((canonical, float(weight), auxiliary))
                auxiliary += 1
        self.num_variables = auxiliary if map_type == 'star' else self.num_nodes
        self._groups = list(groups.values())
        self._cache = {}

    def _tensors(self, p):
        key = (p.device, p.dtype)
        if key not in self._cache:
            self._cache[key] = [
                (torch.tensor([row[0] for row in group], device=p.device, dtype=torch.long),
                 torch.tensor([row[1] for row in group], device=p.device, dtype=p.dtype),
                 torch.tensor([row[2] for row in group], device=p.device, dtype=torch.long))
                for group in self._groups
            ]
        return self._cache[key]

    def cut_expectation(self, p):
        if p.ndim != 3 or p.shape[1:] != (self.num_variables, self.q):
            raise ValueError(f'expected probabilities shaped (batch, {self.num_variables}, {self.q})')
        value = p.sum(dim=(1, 2)) * 0
        for indices, weights, auxiliary in self._tensors(p):
            pe = p[:, indices, :]
            if self.map_type == 'native':
                edge_cost = (1 - (1 - pe).prod(dim=2)).sum(dim=-1) - 1
            elif self.map_type == 'clique':
                size = indices.shape[1]
                pair_dot = (pe.sum(dim=2).square().sum(dim=-1) - pe.square().sum(dim=(2, 3))) / 2
                edge_cost = (size * (size - 1) / 2 - pair_dot) / (size - 1)
            else:
                edge_cost = (1 - (pe * p[:, auxiliary, None, :]).sum(dim=-1)).sum(dim=-1)
            value = value + (edge_cost * weights).sum(dim=-1)
        return value

    def expectation(self, _, p):
        # Normalize only the soft penalty by average physical-node weight.
        # Rescaling resource units then leaves the optimization unchanged;
        # final discrete capacities still use the original physical weights.
        weights = torch.as_tensor(self.node_weights / self.balance_scale, device=p.device, dtype=p.dtype)
        loads = (p[:, :self.num_nodes] * weights[None, :, None]).sum(dim=1)
        target = weights.sum() / self.q
        penalty = (loads - target).square().sum(dim=-1)
        return self.cut_expectation(p) + self.imbalance_weight * penalty

    def inference(self, _, p):
        return p.argmax(dim=-1)

    def energy(self, _, configurations):
        """Original weighted km1, without a surrogate or soft balance penalty."""
        labels = configurations.argmax(dim=-1) if configurations.ndim == 3 else configurations
        result = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.float64)
        for edge, weight in zip(self.hyperedges, self.hyperedge_weights):
            if len(edge) > 1:
                occupied = torch.stack([(labels[:, edge] == b).any(dim=-1) for b in range(self.q)], dim=-1)
                result += weight * (occupied.sum(dim=-1) - 1)
        return result
