"""Objective-preserving quotients of weighted hypergraphs.

The vertex map identifies vertices.  Hyperedges retain their identities and
weights, except that edges mapped to fewer than two vertices contribute zero
to the connectivity objective and are omitted.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def quotient_hypergraph(
    hyperedges,
    vertex_map,
    vertex_weights=None,
    hyperedge_weights=None,
):
    """Return ``(edges, vertex_weights, edge_weights)`` on the quotient.

    ``vertex_map[v]`` is the coarse vertex containing original vertex ``v``.
    Coarse ids must be contiguous from zero.  Parallel hyperedges are retained
    so their individual weights remain available to later refinement stages.
    """
    vertex_map = np.asarray(vertex_map)
    if vertex_map.size and not np.issubdtype(vertex_map.dtype, np.integer):
        raise ValueError("vertex_map must contain integer coarse ids")
    vertex_map = vertex_map.astype(np.int64, copy=False)
    if vertex_map.ndim != 1 or np.any(vertex_map < 0):
        raise ValueError("vertex_map must be a one-dimensional nonnegative array")
    num_vertices = vertex_map.size
    num_coarse = int(vertex_map.max()) + 1 if num_vertices else 0
    if num_coarse and np.unique(vertex_map).size != num_coarse:
        raise ValueError("coarse vertex ids must be contiguous from zero")

    if vertex_weights is None:
        vertex_weights = np.ones(num_vertices, dtype=np.float64)
    else:
        vertex_weights = np.asarray(vertex_weights, dtype=np.float64)
    if (vertex_weights.shape != (num_vertices,) or
            not np.all(np.isfinite(vertex_weights)) or np.any(vertex_weights < 0)):
        raise ValueError("vertex_weights must contain one finite nonnegative value per vertex")

    if hyperedge_weights is None:
        hyperedge_weights = np.ones(len(hyperedges), dtype=np.float64)
    else:
        hyperedge_weights = np.asarray(hyperedge_weights, dtype=np.float64)
    if (hyperedge_weights.shape != (len(hyperedges),) or
            not np.all(np.isfinite(hyperedge_weights)) or np.any(hyperedge_weights < 0)):
        raise ValueError("hyperedge_weights must contain one finite nonnegative value per edge")

    coarse_vertex_weights = np.bincount(
        vertex_map, weights=vertex_weights, minlength=num_coarse,
    )
    coarse_edges = []
    coarse_edge_weights = []
    for edge, weight in zip(hyperedges, hyperedge_weights):
        mapped = []
        seen = set()
        for vertex in edge:
            vertex = int(vertex)
            if vertex < 0 or vertex >= num_vertices:
                raise ValueError(f"hyperedge contains invalid vertex {vertex}")
            coarse_vertex = int(vertex_map[vertex])
            if coarse_vertex not in seen:
                mapped.append(coarse_vertex)
                seen.add(coarse_vertex)
        if len(mapped) > 1:
            coarse_edges.append(mapped)
            coarse_edge_weights.append(float(weight))

    return coarse_edges, coarse_vertex_weights, np.asarray(coarse_edge_weights, dtype=np.float64)


def connectivity_cost(assignment, hyperedges, hyperedge_weights=None):
    """Weighted connectivity-minus-one objective, without balance penalty."""
    assignment = np.asarray(assignment, dtype=np.int64)
    if hyperedge_weights is None:
        hyperedge_weights = np.ones(len(hyperedges), dtype=np.float64)
    if len(hyperedges) != len(hyperedge_weights):
        raise ValueError("one weight is required per hyperedge")
    return sum(
        float(weight) * (len({int(assignment[v]) for v in edge}) - 1)
        for edge, weight in zip(hyperedges, hyperedge_weights)
        if len(edge) > 1
    )


def balanced_packing_feasible(vertex_weights, num_blocks, max_block_weight):
    """Exactly decide if indivisible vertex weights fit into capped blocks.

    This is a bin-packing decision problem and can take exponential time; use
    only on small coarse graphs. Empty blocks are allowed, as in the solver's
    maximum-block-weight balance constraint.
    """
    weights = np.asarray(vertex_weights, dtype=np.float64)
    if (weights.ndim != 1 or not np.all(np.isfinite(weights)) or
            np.any(weights < 0)):
        raise ValueError('vertex_weights must be finite and nonnegative')
    if num_blocks < 1:
        raise ValueError('num_blocks must be positive')
    capacity = float(max_block_weight)
    if not np.isfinite(capacity) or capacity < 0:
        raise ValueError('max_block_weight must be finite and nonnegative')
    tolerance = 1e-12 * max(1.0, capacity, float(weights.sum()))
    if np.any(weights > capacity + tolerance):
        return False
    if float(weights.sum()) > num_blocks * capacity + tolerance:
        return False

    ordered = tuple(sorted((float(w) for w in weights if w > 0), reverse=True))

    @lru_cache(maxsize=None)
    def search(index, loads):
        if index == len(ordered):
            return True
        weight = ordered[index]
        seen_loads = set()
        for block, load in enumerate(loads):
            if load in seen_loads:
                continue
            seen_loads.add(load)
            if load + weight > capacity + tolerance:
                continue
            next_loads = list(loads)
            next_loads[block] += weight
            if search(index + 1, tuple(sorted(next_loads))):
                return True
        return False

    return search(0, (0.0,) * num_blocks)
