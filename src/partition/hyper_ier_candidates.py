"""Bounded, disjoint, load-preserving moves for native hypergraph IER.

These moves are optimization proposals, not a symmetric sampling kernel.  A
binary variable may enable each returned move: disjoint vertices and a bound
on the sum of positive load errors make every such subset capacity-feasible.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import combinations
import math
from numbers import Integral

import numpy as np


# Relative to the exchanged resource, never an absolute unit-sized floor.
_BALANCE_RTOL = 1e-12


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _weights(value, size, name):
    try:
        weights = np.ones(size) if value is None else np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain finite nonnegative weights') from exc
    if weights.shape != (size,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f'{name} must contain {size} finite nonnegative weights')
    return weights


def _stratified_sample(vertices, labels, limit, rng):
    """Random round-robin across represented blocks, with a total size cap."""
    groups = defaultdict(list)
    for vertex in vertices:
        groups[int(labels[vertex])].append(int(vertex))
    for group in groups.values():
        rng.shuffle(group)
    active = list(groups)
    rng.shuffle(active)
    selected = []
    while active and len(selected) < limit:
        remaining = []
        for block in active:
            selected.append(groups[block].pop())
            if len(selected) == limit:
                return selected
            if groups[block]:
                remaining.append(block)
        active = remaining
    return selected


def _matches(sorted_masses, mass, rng, limit=4):
    """Uniformly retain a few equal-mass matches without materializing a bin."""
    radius = _BALANCE_RTOL * mass / (1 - _BALANCE_RTOL)
    lo = bisect_left(sorted_masses, max(0., mass - radius))
    hi = bisect_right(sorted_masses, mass + radius)
    if hi - lo <= limit:
        return range(lo, hi)
    return lo + rng.choice(hi - lo, size=limit, replace=False)


def _local_pool(edges, cut_edges, incident, boundary, labels, limit, rng):
    """Grow a bounded connected cut-net neighborhood, then fill/cover blocks."""
    if not cut_edges:
        return _stratified_sample(range(len(labels)), labels, limit, rng)
    is_cut = set(cut_edges)
    visited_edges, selected, pool, frontier = set(), set(), [], []

    def add_edge(edge_id):
        visited_edges.add(edge_id)
        fresh = [v for v in edges[edge_id] if v not in selected]
        added = _stratified_sample(fresh, labels, limit - len(pool), rng)
        pool.extend(added)
        selected.update(added)
        frontier.extend(added)

    add_edge(cut_edges[int(rng.integers(len(cut_edges)))])
    cursor = 0
    while cursor < len(frontier) and len(pool) < limit:
        vertex = frontier[cursor]
        cursor += 1
        neighbors = [edge_id for edge_id in incident[vertex]
                     if edge_id in is_cut and edge_id not in visited_edges]
        rng.shuffle(neighbors)
        for edge_id in neighbors:
            add_edge(edge_id)
            if len(pool) >= limit:
                break
    if len(pool) < limit:
        fresh = [int(v) for v in np.flatnonzero(boundary) if v not in selected]
        added = _stratified_sample(fresh, labels, limit - len(pool), rng)
        pool.extend(added)
        selected.update(added)
    if len(pool) < limit:
        fresh = [int(v) for v in np.flatnonzero(~boundary) if v not in selected]
        pool.extend(_stratified_sample(fresh, labels, limit - len(pool), rng))

    # A local component can omit a block entirely.  Exchange at most one
    # redundant local vertex per missing block for a global anchor.  When the
    # pool can hold all occupied blocks, each has a representative; boundary
    # anchors are preferred to internal/isolated ones.
    all_groups, boundary_groups = defaultdict(list), defaultdict(list)
    for vertex, label in enumerate(labels):
        all_groups[int(label)].append(vertex)
        if boundary[vertex]:
            boundary_groups[int(label)].append(vertex)
    represented = Counter(int(labels[v]) for v in pool)
    missing = [block for block in all_groups if block not in represented]
    rng.shuffle(missing)
    for block in missing:
        replacements = [i for i, v in enumerate(pool) if represented[int(labels[v])] > 1]
        if not replacements:
            break
        index = replacements[-1]
        represented[int(labels[pool[index]])] -= 1
        anchors = boundary_groups[block] or all_groups[block]
        pool[index] = anchors[int(rng.integers(len(anchors)))]
        represented[block] += 1
    return pool


def _native_delta(changes, labels, incident, counts, weights):
    """Score touched label counts, without rescanning every pin of large nets."""
    edge_deltas = defaultdict(Counter)
    for vertex, target in changes.items():
        for edge_id in incident[vertex]:
            edge_deltas[edge_id][int(labels[vertex])] -= 1
            edge_deltas[edge_id][target] += 1
    terms = []
    for edge_id, changed_counts in edge_deltas.items():
        occupied_delta = sum(
            int(counts[edge_id][label] + difference > 0) - int(counts[edge_id][label] > 0)
            for label, difference in changed_counts.items()
        )
        terms.append(float(weights[edge_id]) * occupied_delta)
    return math.fsum(terms)


def generate_balanced_moves(
    assignment, hyperedges, q, node_weights=None, hyperedge_weights=None,
    epsilon=.03, *, max_moves=32, boundary_pool=128, seed=1, allow_triples=True,
    pool_strategy='local',
):
    """Return disjoint ``{vertex: new_label}`` pair/triple transpositions.

    ``boundary_pool`` caps the TOTAL selected vertices, across all blocks.
    The default ``pool_strategy='local'`` starts at a random positive-weight
    cut net and grows through cut-net incidence, encouraging atoms to share
    nets and hence interact.  It fills shortfalls globally and replaces a few
    redundant vertices with anchors for missing blocks, if the cap permits.
    ``'global'`` instead samples boundary vertices stratified by block.  In
    both strategies, free slots are filled by other vertices, including
    isolated ones, and only a local RNG is used.  Related block pairs (sharing
    a cut net in the pool) have priority over fallback pairs.

    Equal-weight 1-for-1 exchanges and, optionally, equal-total-weight 2-for-1
    exchanges are considered.  Weight sums use a 1e-12 relative tolerance at
    the exchanged-resource scale.  Selection additionally bounds, for every
    block, initial load plus the sum of ALL positive move load errors by the
    upper capacity plus ``1e-10 * total_weight``.  Thus simultaneous subsets
    cannot accumulate tolerated errors into a capacity violation.

    Matching examines at most quadratic-size lists in the bounded pool, not
    all graph vertex pairs.  At most four reservoir pools of
    ``max(16, 4*max_moves, boundary_pool)`` atoms are scored by their exact
    native weighted-km1 deltas.  Scoring is real proposal-generation work;
    callers should include this function in optimization timing.  We retain
    non-improving atoms and randomly choose among the four best remaining
    ranked atoms, allowing joint improvements and seed diversity.  No claim
    of completeness or irreducibility is made.

    Inputs are read-only.  Empty inputs and ``q=1`` produce no moves after
    validation; zero-weight nodes, repeated pins and empty nets are supported.
    """
    q = _integer(q, 'q', 1)
    max_moves = _integer(max_moves, 'max_moves')
    boundary_pool = _integer(boundary_pool, 'boundary_pool')
    seed = _integer(seed, 'seed')
    if not isinstance(allow_triples, (bool, np.bool_)):
        raise ValueError('allow_triples must be boolean')
    if pool_strategy not in ('local', 'global'):
        raise ValueError("pool_strategy must be 'local' or 'global'")
    if np.ndim(epsilon) != 0:
        raise ValueError('epsilon must be a finite nonnegative scalar')
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be a finite nonnegative scalar')
    labels = np.asarray(assignment)
    if labels.ndim != 1 or (labels.size and not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError('assignment must be a one-dimensional integer array')
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= q):
        raise ValueError('assignment labels must lie in [0, q)')
    n = len(labels)
    nodes = _weights(node_weights, n, 'node_weights')
    # Match the production capacity formula, while summing each block and
    # each signed move residual with fsum as the joint objective does.
    with np.errstate(over='ignore'):
        total = float(nodes.sum())
    capacity = (1 + epsilon) * total / q
    if not math.isfinite(total) or not math.isfinite(capacity):
        raise ValueError('block capacity must be finite')
    tolerance = 1e-10 * max(total, capacity)
    load_parts = [[] for _ in range(q)]
    for label, weight in zip(labels, nodes):
        load_parts[int(label)].append(float(weight))
    loads = np.asarray([math.fsum(parts) for parts in load_parts])
    if np.any(loads > capacity + tolerance):
        raise ValueError('initial assignment exceeds the upper block capacity')

    edges, incident, counts = [], [[] for _ in range(n)], []
    for edge_id, edge in enumerate(hyperedges):
        vertices = sorted({_integer(v, 'hyperedge vertex') for v in edge})
        if vertices and vertices[-1] >= n:
            raise ValueError('hyperedge vertex outside [0, n)')
        edges.append(vertices)
        counts.append(Counter(int(labels[v]) for v in vertices))
        for vertex in vertices:
            incident[vertex].append(edge_id)
    weights = _weights(hyperedge_weights, len(edges), 'hyperedge_weights')
    if n < 2 or q < 2 or max_moves == 0 or boundary_pool < 2:
        return []
    rng = np.random.default_rng(seed)
    boundary = np.zeros(n, dtype=bool)
    cut_edges = []
    for edge_id, (edge, weight) in enumerate(zip(edges, weights)):
        if weight > 0 and len(counts[edge_id]) > 1:
            boundary[edge] = True
            cut_edges.append(edge_id)
    if pool_strategy == 'local':
        pool = _local_pool(edges, cut_edges, incident, boundary, labels, boundary_pool, rng)
    else:
        pool = _stratified_sample(np.flatnonzero(boundary), labels, boundary_pool, rng)
        pool += _stratified_sample(np.flatnonzero(~boundary), labels, boundary_pool - len(pool), rng)
    groups = defaultdict(list)
    for vertex in pool:
        groups[int(labels[vertex])].append(vertex)
    selected = set(pool)
    related = set()
    for edge_id in cut_edges:
        blocks = sorted({int(labels[v]) for v in edges[edge_id] if v in selected})
        related.update(combinations(blocks, 2))
    block_pairs = list(combinations(sorted(groups), 2))
    rng.shuffle(block_pairs)
    by_mass = {block: sorted((float(nodes[v]), v) for v in vertices)
               for block, vertices in groups.items()}
    pair_sums = {block: sorted((math.fsum((float(nodes[u]), float(nodes[v]))), u, v)
                              for u, v in combinations(vertices, 2))
                 for block, vertices in groups.items()} if allow_triples else {}
    masses = {block: [row[0] for row in rows] for block, rows in by_mass.items()}
    sum_masses = {block: [row[0] for row in rows] for block, rows in pair_sums.items()}

    # Separate reservoirs prevent abundant equal-weight pairs from excluding
    # all triples, and fallback pairs from excluding boundary-related atoms.
    reservoirs, seen = defaultdict(list), Counter()
    reservoir_size = max(16, 4 * max_moves, boundary_pool)

    def offer(outgoing, incoming, first, second):
        left = math.fsum(float(nodes[v]) for v in outgoing)
        right = math.fsum(float(nodes[v]) for v in incoming)
        error = math.fsum([float(nodes[v]) for v in incoming]
                          + [-float(nodes[v]) for v in outgoing])
        if abs(error) > _BALANCE_RTOL * max(left, right):
            return
        if abs(error) > tolerance:
            return
        changes = {v: second for v in outgoing}
        changes.update({v: first for v in incoming})
        key = (tuple(sorted((first, second))) in related, len(changes) == 3)
        atom = (changes, {first: error, second: -error}, key[0])
        seen[key] += 1
        reservoir = reservoirs[key]
        if len(reservoir) < reservoir_size:
            reservoir.append(atom)
        else:
            index = int(rng.integers(seen[key]))
            if index < reservoir_size:
                reservoir[index] = atom

    for first, second in block_pairs:
        for mass, vertex in by_mass[first]:
            for index in _matches(masses[second], mass, rng):
                offer((vertex,), (by_mass[second][index][1],), first, second)
        if allow_triples:
            for pair_block, single_block in ((first, second), (second, first)):
                for mass, vertex in by_mass[single_block]:
                    for index in _matches(sum_masses[pair_block], mass, rng):
                        _, one, two = pair_sums[pair_block][index]
                        offer((one, two), (vertex,), pair_block, single_block)

    ranked = []
    for reservoir in reservoirs.values():
        for changes, errors, is_related in reservoir:
            score = _native_delta(changes, labels, incident, counts, weights)
            if not math.isfinite(score):
                raise ValueError('native candidate delta must be finite')
            ranked.append((not is_related, score, float(rng.random()), changes, errors))
    ranked.sort(key=lambda row: row[:3])
    result, used = [], set()
    positive_errors = [[] for _ in range(q)]
    while ranked and len(result) < max_moves:
        # The small randomized top window also yields diversity when the
        # vertex pool itself is smaller than boundary_pool.
        index = int(rng.integers(min(4, len(ranked))))
        _, _, _, changes, errors = ranked.pop(index)
        if used.intersection(changes):
            continue
        if any(loads[block] + math.fsum(positive_errors[block] + [max(0., error)]) > capacity + tolerance
               for block, error in errors.items()):
            continue
        result.append(changes)
        used.update(changes)
        for block, error in errors.items():
            positive_errors[block].append(max(0., error))
    return result
