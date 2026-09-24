"""Input and upper-capacity checks shared by hypergraph refinement paths."""

import numpy as np


def validated_weights(weights, size, name):
    values = np.ones(size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if (values.shape != (size,) or not np.all(np.isfinite(values))
            or np.any(values < 0)):
        raise ValueError(f'{name} must contain {size} finite nonnegative weights')
    return values


def capacity_state(assignment, q, node_weights, epsilon):
    """Return validated labels, node weights, loads, upper capacity and tolerance.

    The partition constraint is an upper bound only. For more than two blocks,
    an absolute deviation from the ideal would impose an extra lower bound.
    """
    if isinstance(q, bool) or int(q) != q or q < 1:
        raise ValueError('q must be a positive integer')
    q = int(q)
    labels = np.asarray(assignment)
    if labels.ndim != 1 or (labels.size and not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError('assignment must be a one-dimensional integer array')
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= q):
        raise ValueError('assignment labels must lie in [0, q)')
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('max_imbalance must be finite and nonnegative')
    weights = validated_weights(node_weights, len(labels), 'node_weights')
    total = float(weights.sum())
    if not np.isfinite(total):
        raise ValueError('total node weight must be finite')
    from src.partition.hyper_objective import capacity_limits
    capacity, tolerance = capacity_limits(weights, q, epsilon)
    if not np.isfinite(capacity):
        raise ValueError('block capacity must be finite')
    loads = np.bincount(labels, weights=weights, minlength=q)
    return labels, weights, loads, capacity, tolerance


def normalized_hyperedges(hyperedges, num_nodes):
    """Keep edge identities while removing repeated pins within a hyperedge."""
    result = []
    for edge in hyperedges:
        normalized = []
        seen = set()
        for vertex in edge:
            if isinstance(vertex, bool) or int(vertex) != vertex or not 0 <= vertex < num_nodes:
                raise ValueError('hyperedge vertex ids must be integers in [0, num_nodes)')
            vertex = int(vertex)
            if vertex not in seen:
                normalized.append(vertex)
                seen.add(vertex)
        result.append(normalized)
    return result
