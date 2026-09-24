"""
Hypergraph solver classes.

Each solver type encapsulates a single concern:
- KahyparLikeSolver  → HEM / LSH coarsening (produces coarse structure)
- FemCoarsenSolver   → FEM-based initial partition on a coarsened hypergraph
- HyperRefineSolver  → local refinement on the full hypergraph

Usage::
    # 1. Coarsen once (returns hierarchy_stack for V-cycle)
    res = kahypar_solver.coarsen(hyperedges, num_nodes, q)

    # 2. Apply different initial partition strategies to the SAME coarse result
    greedy_assignment = kahypar_solver.initial_partition_greedy(
        res['coarse_hyperedges'], res['coarse_node_weights'], q)
    fem_assignment = fem_solver.initial_partition(
        res['coarse_hyperedges'], res['coarse_node_weights'], q)

    # 3. V-Cycle: project up through the hierarchy, refining at each level
    final = vcycle_uncoarsen(
        fem_assignment, res['hierarchy_stack'], hyperedges, q,
        refine_solver, verbose=True,
    )
"""

from __future__ import annotations
import heapq
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.partition.hyper_utils import (
    build_clique_expanded_graph,
    evaluate_kahypar_cut_value,
    greedy_initial_hypergraph_partition,
    greedy_refine_hypergraph_incremental,
)
from src.partition.hyper_quotient import balanced_packing_feasible, quotient_hypergraph


# ── Helper: build coarse hyperedges from groups ──────────────────────────


def _build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes):
    coarse_hyperedges = []
    for he in hyperedges:
        coarse_he = []
        seen = set()
        for v in he:
            if v < num_nodes:
                c = int(original_to_coarse[v])
                if c not in seen:
                    coarse_he.append(c)
                    seen.add(c)
        if len(coarse_he) > 1:
            coarse_hyperedges.append(coarse_he)
    return coarse_hyperedges


# ── Hypergraph solver base ───────────────────────────────────────────────


class HyperSolverBase:
    """Base class for hypergraph solvers."""

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self._config: Dict[str, Any] = {}

    def get_param(self, key: str, default=None):
        return self._config.get(key, default)

    def set_param(self, key: str, value):
        self._config[key] = value

    def update_params(self, **kwargs):
        self._config.update(kwargs)

    def get_all_params(self) -> Dict[str, Any]:
        return dict(self._config)


# ── KaHyPar-like coarsening solver ───────────────────────────────────────


class KahyparLikeSolver(HyperSolverBase):
    """HEM (heavy-edge matching) coarsening directly on hyperedges.

    This solver ONLY does coarsening – it returns the coarse hypergraph
    structure (groups, hyperedges, node weights) but does NOT produce an
    initial partition.  Use ``.initial_partition_greedy()`` for that, or
    pass the coarse structure to ``FemCoarsenSolver.initial_partition()``.
    """

    def coarsen(self, hyperedges, num_nodes, q, **overrides):
        """Run HEM coarsening rounds until ``coarsen_to`` is reached.

        ``score_mode='hem'`` preserves the existing matching rule.  The
        experimental ``'boundary'`` mode discounts candidate pairs that
        pilot partitions place in different blocks.  Optional
        ``enforce_balance_cap=True`` prevents a coarse vertex from exceeding
        one block's maximum allowed weight. The opt-in
        ``enforce_global_feasibility=True`` additionally rejects contractions
        that would make balanced block packing impossible; this exact check
        is limited to ``global_feasibility_max_nodes`` original vertices
        (default 20) and does not support LSH. ``node_weights`` and
        ``hyperedge_weights`` are accepted through ``overrides``.

        Returns a dict with keys:
            coarse_groups, coarse_hyperedges, coarse_node_weights,
            coarse_hyperedge_weights, original_to_coarse, coarse_graph,
            hierarchy_stack.
        Does NOT include ``initial_assignment``.
        """
        p = {**self._config, **overrides}
        rng = np.random.default_rng(p.get('seed', None))

        target_coarse = max(1, int(p.get('coarsen_to', 50)))
        verbose = p.get('verbose', False)
        score_mode = p.get('score_mode', 'hem')
        if score_mode not in ('hem', 'boundary'):
            raise ValueError("score_mode must be 'hem' or 'boundary'")
        boundary_weight = float(p.get('boundary_weight', 0.4))
        if not 0.0 <= boundary_weight <= 1.0:
            raise ValueError('boundary_weight must be between zero and one')
        if q <= 0:
            raise ValueError('q must be positive')
        global_feasibility = bool(p.get('enforce_global_feasibility', False))
        global_max_nodes = int(p.get('global_feasibility_max_nodes', 20))
        if global_feasibility and (global_max_nodes < 1 or num_nodes > global_max_nodes):
            raise ValueError(
                'exact global feasibility guard supports at most '
                'global_feasibility_max_nodes original vertices'
            )

        original_node_weights = np.asarray(
            p.get('node_weights', np.ones(num_nodes)), dtype=np.float64,
        )
        original_edge_weights = np.asarray(
            p.get('hyperedge_weights', np.ones(len(hyperedges))), dtype=np.float64,
        )
        original_edges, _, original_edge_weights = quotient_hypergraph(
            hyperedges, np.arange(num_nodes), original_node_weights,
            original_edge_weights,
        )

        enforce_cap = bool(p.get('enforce_balance_cap', score_mode == 'boundary')) or global_feasibility
        max_cluster_weight = p.get('max_cluster_weight', None)
        if max_cluster_weight is None and enforce_cap:
            max_cluster_weight = (1.0 + float(p.get('epsilon', 0.03))) * original_node_weights.sum() / q
        if max_cluster_weight is not None:
            max_cluster_weight = float(max_cluster_weight)
            if max_cluster_weight < 0 or np.any(original_node_weights > max_cluster_weight + 1e-12):
                raise ValueError('max_cluster_weight must accommodate every original vertex')
        if global_feasibility and not balanced_packing_feasible(
            original_node_weights, q, max_cluster_weight,
        ):
            raise ValueError('the original vertices have no feasible balanced partition')

        pilots = None
        if score_mode == 'boundary' and num_nodes:
            pilots = p.get('pilot_assignments', None)
            if pilots is None:
                num_pilots = max(1, int(p.get('num_pilots', 4)))
                pilot_refine_passes = int(p.get('pilot_refine_passes', 0))
                if pilot_refine_passes < 0:
                    raise ValueError('pilot_refine_passes must be nonnegative')
                pilot_seed = p.get('seed', None)
                pilots = []
                for i in range(num_pilots):
                    pilot = greedy_initial_hypergraph_partition(
                        original_edges, original_node_weights, q,
                        hyperedge_weights=original_edge_weights,
                        epsilon=float(p.get('epsilon', 0.03)),
                        seed=None if pilot_seed is None else int(pilot_seed) + i,
                    )
                    if pilot_refine_passes:
                        pilot = greedy_refine_hypergraph_incremental(
                            pilot, original_edges, original_edge_weights, q,
                            max_passes=pilot_refine_passes,
                            max_imbalance=float(p.get('epsilon', 0.03)),
                            node_weights=original_node_weights,
                        )
                    pilots.append(pilot)
            pilots = np.asarray(pilots, dtype=np.int64)
            if pilots.ndim == 1:
                pilots = pilots[None, :]
            if pilots.ndim != 2 or pilots.shape[1] != num_nodes or pilots.shape[0] == 0:
                raise ValueError('pilot_assignments must have shape (num_pilots, num_nodes)')
            if np.any(pilots < 0) or np.any(pilots >= q):
                raise ValueError('pilot labels must be in [0, q)')

        hierarchy_stack: list[dict] = []

        # ── Optionally pre-coarsen with LSH ───────────────────────────────
        use_lsh = p.get('use_lsh', False)
        if use_lsh and global_feasibility:
            raise ValueError('exact global feasibility guard requires use_lsh=False')
        if use_lsh and score_mode == 'boundary':
            raise ValueError('boundary scoring requires use_lsh=False; LSH groups form before pair scoring')
        if use_lsh:
            lsh_map, lsh_groups = _lsh_bucketize_vertices(
                original_edges, num_nodes,
                target_buckets=max(1, target_coarse * 4),
                seed=p.get('seed', None),
                verbose=verbose,
            )
            if max_cluster_weight is not None:
                capped_groups = []
                for group in lsh_groups:
                    chunk = []
                    chunk_weight = 0.0
                    for vertex in group:
                        weight = float(original_node_weights[vertex])
                        if chunk and chunk_weight + weight > max_cluster_weight + 1e-12:
                            capped_groups.append(chunk)
                            chunk = []
                            chunk_weight = 0.0
                        chunk.append(vertex)
                        chunk_weight += weight
                    if chunk:
                        capped_groups.append(chunk)
                lsh_groups = capped_groups
                for index, group in enumerate(lsh_groups):
                    lsh_map[group] = index
            if verbose:
                print(f"[kahypar_like] LSH pre-coarsen: {num_nodes} -> {len(lsh_groups)} buckets")
            current_hyperedges, current_node_weights, current_edge_weights = quotient_hypergraph(
                original_edges, lsh_map, original_node_weights, original_edge_weights,
            )
            current_groups = [list(g) for g in lsh_groups]
            if num_nodes != len(current_groups):
                hierarchy_stack.append({
                    'hyperedges': [list(edge) for edge in original_edges],
                    'hyperedge_weights': original_edge_weights.copy(),
                    'node_weights': original_node_weights.copy(),
                    'groups': [[i] for i in range(num_nodes)],
                    'remap': lsh_map.copy(),
                    'num_nodes': num_nodes,
                })
        else:
            current_hyperedges = original_edges
            current_edge_weights = original_edge_weights.copy()
            current_node_weights = original_node_weights.copy()
            current_groups = [[i] for i in range(num_nodes)]

        current_n = len(current_groups)
        if current_n == 0:
            empty_graph = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32), (0, 0),
            ).coalesce()
            return {
                'coarse_graph': empty_graph,
                'coarse_node_weights': torch.empty((0,), dtype=torch.float32),
                'coarse_groups': [],
                'original_to_coarse': np.empty((0,), dtype=np.int64),
                'coarse_hyperedges': [],
                'coarse_hyperedge_weights': np.empty((0,), dtype=np.float64),
                'hierarchy_stack': hierarchy_stack,
            }

        # ── HEM matching rounds ──────────────────────────────────────────
        # Build incidence ONCE — updated statefully through the loop
        vertex_to_edges, edge_vertices, edge_weights = _build_incidence(
            current_hyperedges, current_n, current_edge_weights,
        )

        round_id = 0
        while current_n > target_coarse:
            round_id += 1
            alive = np.ones(current_n, dtype=bool)
            matched = np.zeros(current_n, dtype=bool)
            partner = np.full(current_n, -1, dtype=np.int64)
            pilot_hist = None
            if pilots is not None:
                pilot_hist = np.zeros((current_n, pilots.shape[0], q), dtype=np.float64)
                for group_id, members in enumerate(current_groups):
                    for trial, labels in enumerate(pilots):
                        pilot_hist[group_id, trial] = np.bincount(
                            labels[members], minlength=q,
                        ) / len(members)

            # vertex_to_edges / edge_vertices are already up-to-date from
            # the previous round's stateful merge — no rebuild needed.
            order = rng.permutation(current_n)
            pair_count = 0
            packing_weights = current_node_weights.copy() if global_feasibility else None
            packing_active = np.ones(current_n, dtype=bool) if global_feasibility else None

            for u in order:
                if not alive[u] or matched[u]:
                    continue
                ratings = {}
                for eid in vertex_to_edges[u]:
                    verts = edge_vertices[eid]
                    if len(verts) < 2:
                        continue
                    contrib = float(edge_weights[eid]) / float(len(verts) - 1)
                    for v in verts:
                        if (v != u and alive[v] and not matched[v]
                                and (max_cluster_weight is None or
                                     current_node_weights[u] + current_node_weights[v]
                                     <= max_cluster_weight + 1e-12)):
                            ratings[v] = ratings.get(v, 0.0) + contrib
                if not ratings:
                    continue
                if pilot_hist is not None:
                    scored = {}
                    for v, rating in ratings.items():
                        same_probability = np.mean(np.sum(
                            pilot_hist[u] * pilot_hist[v], axis=1,
                        ))
                        disagreement = 1.0 - same_probability
                        scored[v] = rating * (1.0 - boundary_weight * disagreement)
                else:
                    scored = ratings
                v = None
                for candidate in sorted(scored, key=lambda item: (scored[item], -item), reverse=True):
                    if scored[candidate] <= 0.0:
                        break
                    if global_feasibility:
                        remaining = packing_weights[
                            packing_active & (np.arange(current_n) != u) &
                            (np.arange(current_n) != candidate)
                        ]
                        proposed = np.append(
                            remaining, packing_weights[u] + packing_weights[candidate],
                        )
                        if not balanced_packing_feasible(proposed, q, max_cluster_weight):
                            continue
                    v = candidate
                    break
                if v is None:
                    continue
                matched[u] = True
                matched[v] = True
                partner[u] = v
                partner[v] = u
                if global_feasibility:
                    packing_weights[u] += packing_weights[v]
                    packing_active[v] = False
                pair_count += 1
                if pair_count >= current_n - target_coarse:
                    break

            if pair_count == 0:
                break

            remap = np.full(current_n + pair_count, -1, dtype=np.int64)
            new_groups = []
            new_id = 0
            used = np.zeros(current_n, dtype=bool)
            for u in range(current_n):
                if not alive[u] or used[u]:
                    continue
                v = partner[u]
                if v != -1 and used[v]:
                    continue
                if v != -1:
                    used[u] = used[v] = True
                    remap[u] = remap[v] = new_id
                    new_groups.append(current_groups[u] + current_groups[v])
                else:
                    used[u] = True
                    remap[u] = new_id
                    new_groups.append(current_groups[u])
                new_id += 1

            # ── Combined pass: build new_hyperedges (hierarchy stack)   ──
            #     AND simultaneously build updated vertex_to_edges so that
            #     we never need _build_incidence again inside the loop.
            new_hyperedges = []
            new_edge_weights = []
            new_vertex_to_edges = [set() for _ in range(new_id)]

            for he, weight in zip(current_hyperedges, current_edge_weights):
                mapped = []
                seen = set()
                for v in he:
                    nv = remap[v]
                    if nv < 0 or nv in seen:
                        continue
                    mapped.append(int(nv))
                    seen.add(int(nv))
                if len(mapped) > 1:
                    eid = len(new_hyperedges)
                    new_hyperedges.append(mapped)
                    new_edge_weights.append(float(weight))
                    for mv in mapped:
                        new_vertex_to_edges[mv].add(eid)

            if new_id == current_n:
                break

            # ── Save hierarchy entry before transitioning ──
            hierarchy_stack.append({
                'hyperedges': [list(he) for he in current_hyperedges],
                'hyperedge_weights': current_edge_weights.copy(),
                'node_weights': current_node_weights.copy(),
                'groups': [list(g) for g in current_groups],
                'remap': remap.copy(),
                'num_nodes': current_n,
            })

            # ── Update incidence statefully for the next round ──
            # edge_vertices and new_hyperedges share the same structure
            # (deduped vertex lists per hyperedge), so we can reuse the
            # new_hyperedges list directly.
            edge_vertices = new_hyperedges
            vertex_to_edges = new_vertex_to_edges
            edge_weights = new_edge_weights

            current_hyperedges = new_hyperedges
            current_edge_weights = np.asarray(new_edge_weights, dtype=np.float64)
            current_node_weights = np.bincount(
                remap[:current_n], weights=current_node_weights, minlength=new_id,
            )
            current_groups = new_groups
            current_n = new_id

        # ── Build output ─────────────────────────────────────────────────
        original_to_coarse = np.empty(num_nodes, dtype=np.int64)
        for idx, members in enumerate(current_groups):
            for member in members:
                if member < num_nodes:
                    original_to_coarse[member] = idx

        coarse_hyperedges_out, coarse_weights, coarse_edge_weights = quotient_hypergraph(
            original_edges, original_to_coarse, original_node_weights,
            original_edge_weights,
        )
        coarse_graph = build_clique_expanded_graph(
            coarse_hyperedges_out, num_nodes=len(current_groups), normalize_weight=True,
            hyperedge_weights=coarse_edge_weights,
        )
        coarse_node_weights = torch.tensor(coarse_weights, dtype=torch.float32)

        return {
            'coarse_groups': current_groups,
            'coarse_hyperedges': coarse_hyperedges_out,
            'coarse_node_weights': coarse_node_weights,
            'original_to_coarse': original_to_coarse,
            'coarse_graph': coarse_graph,
            'coarse_hyperedge_weights': coarse_edge_weights,
            'hierarchy_stack': hierarchy_stack,
        }

    def initial_partition_greedy(self, coarse_hyperedges, coarse_node_weights, q, **overrides):
        """Greedy initial partition on the coarse hypergraph (respects node weights)."""
        p = {**self._config, **overrides}
        return greedy_initial_hypergraph_partition(
            coarse_hyperedges,
            coarse_node_weights.cpu().numpy() if torch.is_tensor(coarse_node_weights) else coarse_node_weights,
            q,
            hyperedge_weights=p.get('hyperedge_weights', [1.0] * len(coarse_hyperedges)),
            epsilon=p.get('epsilon', 0.03),
            seed=p.get('seed', None),
        )


# ── FEM initial-partition solver ─────────────────────────────────────────


class FemCoarsenSolver(HyperSolverBase):
    """Optimize native weighted km1, then return a capacity-feasible partition.

    ``method='fem'`` and the legacy ``method='pubo'`` both use the native
    differentiable categorical expectation by default. Explicit ``map_type``
    values ``'clique'``/``'star'`` select surrogate experiments for FEM only;
    all returned candidates are ranked by the native weighted objective.

    ``epsilon`` specifies the upper load cap (1+epsilon)*total/q. Weighted
    rounding may raise BalanceSearchError when its bounded packing search
    fails, or BalanceInfeasibleError only when infeasibility is proved.
    """

    def initial_partition(self, coarse_hyperedges, coarse_node_weights, q, **overrides):
        from fem import FEM
        from src.partition.hyper_objective import (
            HypergraphObjective, BalanceSearchError, capacity_limits,
            round_balanced_probabilities,
        )
        from src.partition.hyper_quotient import connectivity_cost

        options = {**self._config, **overrides}
        method = options.get('method', 'fem')
        if method not in ('fem', 'pubo'):
            raise ValueError("method must be 'fem' or 'pubo'")
        objective = HypergraphObjective(
            coarse_hyperedges, coarse_node_weights, q,
            hyperedge_weights=options.get('hyperedge_weights'),
            imbalance_weight=options.get('imbalance_weight', 5.0),
            map_type='native' if method == 'pubo' else options.get('map_type', 'native'),
        )
        q = objective.q
        epsilon = float(options.get('epsilon', options.get('max_imbalance', 0.03)))
        capacity, tolerance = capacity_limits(objective.node_weights, q, epsilon)
        rounding_options = dict(
            exact_max_nodes=int(options.get('exact_balance_max_nodes', 20)),
            search_budget=int(options.get('balance_search_budget', 200000)),
        )
        self.last_result = None
        if objective.num_nodes == 0:
            return np.empty(0, dtype=np.int64)
        # Obvious capacity impossibilities are diagnosed before optimization.
        if np.any(objective.node_weights > capacity + tolerance):
            from src.partition.hyper_objective import BalanceInfeasibleError
            raise BalanceInfeasibleError('a coarse node exceeds the maximum block capacity')

        # A customize problem does not use its coupling tensor. Avoid a dense
        # N-by-N dummy allocation for a native sparse hypergraph objective.
        case = FEM.from_couplings(
            'customize', objective.num_variables, len(objective.hyperedges), torch.empty(0),
            customize_expected_func=objective.expectation,
            customize_infer_func=objective.inference,
        )
        case.set_up_solver(
            int(options.get('num_trials', 16)), int(options.get('num_steps', 300)),
            dev=options.get('dev', 'cpu'), q=q, manual_grad=False,
            anneal=options.get('anneal', 'exp'),
            betamin=float(options.get('betamin', 0.5)),
            betamax=float(options.get('betamax', 50.0)),
            learning_rate=float(options.get('learning_rate', 0.08)),
            optimizer=options.get('optimizer', 'adam'),
            dtype=options.get('dtype', torch.float64),
            seed=int(options.get('seed', 1)), h_factor=float(options.get('h_factor', 0.1)),
            use_adaptive_annealing=bool(options.get('use_adaptive_annealing', False)),
            adaptive_A=float(options.get('adaptive_A', 0.5)),
            use_compile=bool(options.get('use_compile', False)),
        )
        case.solve()
        probabilities = case.solver.probabilities.detach().cpu().numpy()[:, :objective.num_nodes]
        candidates, candidate_costs, candidate_trials, failures = [], [], [], []
        for trial, probability in enumerate(probabilities):
            try:
                assignment = round_balanced_probabilities(
                    probability, objective.node_weights, q, epsilon, **rounding_options,
                )
            except BalanceSearchError as exc:
                failures.append({'trial': trial, 'message': str(exc)})
                continue
            cost = connectivity_cost(assignment, objective.hyperedges, objective.hyperedge_weights)
            candidates.append(assignment)
            candidate_costs.append(cost)
            candidate_trials.append(trial)
        if not candidates:
            raise BalanceSearchError(
                'no trial produced a capacity-feasible partition; increase the packing budget '
                'or use a less restrictive coarse hierarchy (feasibility is unknown)'
            )
        best = int(np.argmin(candidate_costs))
        assignment = candidates[best]
        loads = np.bincount(assignment, weights=objective.node_weights, minlength=q)
        if loads.max() > capacity + tolerance:
            raise AssertionError('FEM returned a partition above the load capacity')
        self.last_result = {
            'method': method, 'map_type': objective.map_type,
            'native_cut': float(candidate_costs[best]), 'block_loads': loads.copy(),
            'capacity': capacity, 'selected_trial': candidate_trials[best],
            'candidate_native_cuts': np.asarray(candidate_costs),
            'candidate_trials': candidate_trials, 'rounding_failures': failures,
        }
        return assignment.copy()


class _Q4PUBOWrapper:
    """Backward-compatible wrapper using native km1 for arbitrary q >= 2."""

    def __init__(self, hyperedges, node_weights, q, num_nodes, imbalance_weight=5.0,
                 hyperedge_weights=None):
        from src.partition.hyper_objective import HypergraphObjective
        if len(node_weights) != num_nodes:
            raise ValueError('num_nodes and node_weights disagree')
        self._objective = HypergraphObjective(
            hyperedges, node_weights, q, hyperedge_weights=hyperedge_weights,
            imbalance_weight=imbalance_weight,
        )

    def expectation(self, _, p):
        return self._objective.expectation(_, p)

    def inference(self, _, p):
        return self._objective.inference(_, p)


# ── Refinement solver ────────────────────────────────────────────────────


class HyperRefineSolver(HyperSolverBase):
    """Local refinement on the original hypergraph.

    When ``mode_cycle=('flow',)`` (default), runs simple FM (greedy
    incremental refinement). ``('fem_ier',)`` selects balanced move atoms
    with native hypergraph FEM; it can be composed with ``'flow'`` in either
    order. Other cycles (e.g. ``('mcts', 'flow')``) use the legacy hybrid path.

    Parameters (via ``update_params`` or ``**overrides``):
        mode_cycle      — tuple of modes: ('flow',) for FM, or hybrid
        rounds          — number of hybrid rounds (ignored for pure flow)
        flow_passes     — FM passes per flow stage
        max_imbalance   — balance constraint (default 0.05)
        repair_balance  — if True, actively repair balance after refinement
                          using ``_repair_balance_fast`` (default True).
                          Hybrid mode always repairs; this flag controls
                          the simple FM path.
        node_weights    — per-vertex weights for weighted balance (default None)
        hyperedge_weights — per-hyperedge objective weights (flow and fem_ier
                            support nonunit weights; default None means unit)
        ier_*           — opt-in joint-move FEM configuration: rounds,
                          max_moves, boundary_pool, num_trials, num_steps,
                          pool_strategy, backend, allow_triples; see hyper_ier.py
    """

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(config_dir)
        self._config['mode_cycle'] = ('flow',)

    def refine(self, assignment, hyperedges, q, node_weights=None,
               hyperedge_weights=None, **overrides):
        p = {**self._config, **overrides}
        mode_cycle = p.get('mode_cycle', ('flow',))
        repair = p.get('repair_balance', True)
        if node_weights is None:
            node_weights = p.get('node_weights')
        if hyperedge_weights is None:
            hyperedge_weights = p.get('hyperedge_weights')

        # Native FEM-IER is opt-in. Each binary variable controls an entire
        # balanced move atom; all selector combinations preserve capacity.
        if 'fem_ier' in mode_cycle:
            if any(mode not in ('fem_ier', 'flow') for mode in mode_cycle):
                raise ValueError("fem_ier can currently be combined only with 'flow'")
            from src.partition.hyper_ier import refine_fem_ier
            from src.partition.hyper_quotient import connectivity_cost
            result = np.asarray(assignment).copy()
            stages = []
            for stage_index, mode in enumerate(mode_cycle):
                if mode == 'flow':
                    result = _refine_flow(
                        result, hyperedges, q, max_passes=p.get('flow_passes', 5),
                        max_imbalance=p.get('max_imbalance', 0.05), repair_balance=repair,
                        verbose=p.get('verbose', False), node_weights=node_weights,
                        hyperedge_weights=hyperedge_weights)
                    stages.append({'method': 'flow', 'final_native_cut': float(
                        connectivity_cost(result, hyperedges, hyperedge_weights))})
                else:
                    result, diagnostics = refine_fem_ier(
                        result, hyperedges, q, node_weights=node_weights,
                        hyperedge_weights=hyperedge_weights, epsilon=p.get('max_imbalance', .05),
                        rounds=p.get('ier_rounds', 2), max_moves=p.get('ier_max_moves', 24),
                        boundary_pool=p.get('ier_boundary_pool', 96),
                        num_trials=p.get('ier_num_trials', 8), num_steps=p.get('ier_num_steps', 100),
                        seed=p.get('seed', 1) + stage_index,
                        backend=p.get('ier_backend', 'fem'), random_samples=p.get('ier_random_samples'),
                        allow_triples=p.get('ier_allow_triples', True),
                        pool_strategy=p.get('ier_pool_strategy', 'local'),
                        learning_rate=p.get('ier_learning_rate', .08),
                        betamin=p.get('ier_betamin', .5), betamax=p.get('ier_betamax', 50.),
                        dev=p.get('ier_device', 'cpu'), exact_max_moves=p.get('ier_exact_max_moves', 16))
                    stages.append(diagnostics)
            self.last_result = {'method': 'fem_ier_cycle', 'mode_cycle': list(mode_cycle),
                                'stages': stages, 'final_native_cut': float(
                                    connectivity_cost(result, hyperedges, hyperedge_weights))}
            return result

        # Simple FM mode
        if mode_cycle == ('flow',):
            return _refine_flow(
                assignment, hyperedges, q,
                max_passes=p.get('flow_passes', 5),
                max_imbalance=p.get('max_imbalance', 0.05),
                repair_balance=repair,
                verbose=p.get('verbose', False),
                node_weights=node_weights,
                hyperedge_weights=hyperedge_weights,
            )

        # Hybrid mode (MCTS / evolution / flow)
        from src.partition.hyper_refine_contract import (
            capacity_state, normalized_hyperedges, validated_weights,
        )
        assignment, node_weights, _, _, _ = capacity_state(
            assignment, q, node_weights, p.get('max_imbalance', 0.05),
        )
        hyperedges = normalized_hyperedges(hyperedges, len(assignment))
        edge_weights = validated_weights(hyperedge_weights, len(hyperedges), 'hyperedge_weights')
        if np.any(edge_weights != 1.0):
            raise NotImplementedError(
                'hybrid refinement does not yet support nonunit hyperedge_weights; '
                "use mode_cycle=('flow',) for weighted hypergraphs"
            )
        result = _refine_hybrid(
            assignment, hyperedges, q,
            mode_cycle=mode_cycle,
            rounds=p.get('rounds', 3),
            max_imbalance=p.get('max_imbalance', 0.05),
            flow_passes=p.get('flow_passes', 3),
            mcts_rollouts=p.get('mcts_rollouts', 16),
            mcts_depth=p.get('mcts_depth', 3),
            evolution_population=p.get('evolution_population', 8),
            evolution_generations=p.get('evolution_generations', 5),
            evolution_mutation=p.get('evolution_mutation', 0.1),
            skip_exploration_if_good=p.get('skip_exploration_if_good', True),
            verbose=p.get('verbose', False),
            node_weights=node_weights,
        )
        _, _, loads, capacity, tolerance = capacity_state(
            result, q, node_weights, p.get('max_imbalance', 0.05),
        )
        if np.any(loads > capacity + tolerance):
            raise RuntimeError(
                'hybrid refinement failed to return a capacity-feasible partition; '
                'this search failure does not prove the instance infeasible'
            )
        return result


# ═════════════════════════════════════════════════════════════════════════
#  Refine helpers (moved from src/partition/hyper_refine.py)
# ═════════════════════════════════════════════════════════════════════════


def _refine_flow(assignment, hyperedges, q, max_passes=5, max_imbalance=0.05,
                 repair_balance=True, verbose=False, node_weights=None,
                 hyperedge_weights=None):
    """Simple FM (greedy incremental) refinement.

    Parameters
    ----------
    repair_balance : bool
        If True, actively repair balance via ``_repair_balance_fast``
        after FM refinement finishes. A checked capacitated projection is
        attempted if the legacy repair does not produce a feasible result.
    node_weights : np.ndarray or None
        Per-vertex weights for weighted balance computation.

    hyperedge_weights : np.ndarray or None
        Per-hyperedge objective weights, including at coarse levels.

    A failed search/repair raises rather than returning an infeasible
    assignment. Failure is not an infeasibility certificate. A feasible input
    uses native weighted improving moves that preserve the upper capacity.
    """
    from src.partition.hyper_utils import greedy_refine_hypergraph_incremental
    from src.partition.hyper_refine_contract import (
        capacity_state, normalized_hyperedges, validated_weights,
    )
    assignment, node_weights, _, _, _ = capacity_state(
        assignment, q, node_weights, max_imbalance,
    )
    hyperedges = normalized_hyperedges(hyperedges, len(assignment))
    edge_weights = validated_weights(hyperedge_weights, len(hyperedges), 'hyperedge_weights')
    if verbose:
        print(f"[refine:flow] start max_passes={max_passes} max_imbalance={max_imbalance}")
    result = greedy_refine_hypergraph_incremental(
        assignment, hyperedges,
        hyperedge_weights=edge_weights,
        q=q, max_passes=max_passes, max_imbalance=max_imbalance, node_weights=node_weights,
    )
    _, _, loads, capacity, tolerance = capacity_state(result, q, node_weights, max_imbalance)
    if repair_balance and np.any(loads > capacity + tolerance):
        result = _repair_balance_fast(
            result, hyperedges, max_imbalance=max_imbalance, q=q,
            node_weights=node_weights,
        )
        _, _, loads, capacity, tolerance = capacity_state(result, q, node_weights, max_imbalance)
        if np.any(loads > capacity + tolerance):
            from src.partition.hyper_objective import round_balanced_probabilities
            result = round_balanced_probabilities(
                np.eye(q, dtype=np.float64)[result], node_weights, q,
                epsilon=max_imbalance,
            )
            _, _, loads, capacity, tolerance = capacity_state(result, q, node_weights, max_imbalance)
        if verbose:
            print(f"[refine:flow] balance repair applied, loads={loads.tolist()}, cap={capacity:.6g}")
    if np.any(loads > capacity + tolerance):
        raise RuntimeError(
            'flow refinement failed to return a capacity-feasible partition; '
            'this search/repair failure does not prove the instance infeasible'
        )
    return result


def _target_counts(n, q):
    if q <= 0:
        return np.zeros(0, dtype=int)
    base = n // q
    remainder = n % q
    return np.array([base + (1 if i < remainder else 0) for i in range(q)], dtype=int)


def _balance_limits(assignment, max_imbalance, q=None, node_weights=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if q is None:
        q = int(assignment.max()) + 1 if assignment.size else 2
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
        counts = np.zeros(q, dtype=np.float64)
        np.add.at(counts, assignment, node_weights)
        total = float(node_weights.sum())
    else:
        counts = np.bincount(assignment, minlength=q).astype(np.float64)
        total = float(assignment.size)
    ideal = total / float(q) if q > 0 else 0.0
    max_size = ideal * (1.0 + float(max_imbalance)) if total > 0 else 0.0
    min_size = ideal * (1.0 - float(max_imbalance)) if total > 0 else 0.0
    return q, counts, min_size, max_size


def _repair_balance_fast(assignment, hyperedges, max_imbalance=0.05, seed=None, q=None,
                         node_weights=None, max_iterations_mult=1):
    """Fast balance repair without cut evaluation.

    If ``node_weights`` is provided, balance is computed and repaired
    using weighted block sums instead of raw vertex counts.

    Parameters
    ----------
    max_iterations_mult : int
        Multiplier on the default iteration count (``assignment.size * 2``).
        Use >1 when coarse vertices have heavy weights that make single
        moves larger, requiring more passes to converge.
    """
    rng = np.random.default_rng(seed)
    assignment = np.asarray(assignment, dtype=np.int64).copy()
    if assignment.size == 0:
        return assignment
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
    q, counts, min_size, max_size = _balance_limits(
        assignment, max_imbalance, q=q, node_weights=node_weights,
    )
    node_degree = np.zeros(assignment.size, dtype=float)
    for he in hyperedges:
        for v in he:
            if 0 <= v < assignment.size:
                node_degree[v] += 1.0
    w = (lambda v: node_weights[v]) if node_weights is not None else (lambda v: 1.0)
    max_iters = max(1, assignment.size * 2 * max_iterations_mult)
    for _ in range(max_iters):
        over = np.where(counts > max_size)[0]
        if len(over) == 0:
            break
        under = np.where(counts < min_size)[0]
        if len(under) == 0:
            under = np.array([int(np.argmin(counts))], dtype=int)
        donor = int(over[np.argmax(counts[over] - max_size)])
        donor_vertices = np.where(assignment == donor)[0]
        if donor_vertices.size == 0:
            break
        rng.shuffle(donor_vertices)
        donor_vertices = donor_vertices[np.argsort(node_degree[donor_vertices], kind='mergesort')]
        moved = False
        for v in donor_vertices:
            vw = w(v)
            for g in under:
                g = int(g)
                if g == donor:
                    continue
                if counts[g] + vw > max_size:
                    continue
                assignment[v] = g
                counts[donor] -= vw
                counts[g] += vw
                moved = True
                break
            if moved:
                break
        if not moved:
            # Fallback: find the lightest vertex that doesn't overshoot,
            # or if none exists, the lightest vertex overall.
            g = int(np.argmin(counts))
            best_v = int(donor_vertices[0])
            best_vw = w(best_v)
            for v in donor_vertices:
                vw = w(v)
                if counts[g] + vw <= max_size:
                    best_v = int(v)
                    best_vw = vw
                    break
                if vw < best_vw:
                    best_v = int(v)
                    best_vw = vw
            assignment[best_v] = g
            counts[donor] -= best_vw
            counts[g] += best_vw
    return assignment


def _repair_balance(assignment, hyperedges, max_imbalance=0.05, seed=None, q=None,
                    node_weights=None):
    """Cut-aware balance repair using FM-style O(deg(v)) delta evaluation.

    Pre-computes ``he_pins`` (shape ``[num_hyperedges, q]``) and ``node_to_he``,
    then evaluates candidate moves via ``move_gain`` instead of calling
    ``evaluate_kahypar_cut_value`` on a full copy of the assignment.
    """
    rng = np.random.default_rng(seed)
    assignment = np.asarray(assignment, dtype=np.int64).copy()
    num_nodes = len(assignment)
    if num_nodes == 0:
        return assignment
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
    q, counts, min_size, max_size = _balance_limits(
        assignment, max_imbalance, q=q, node_weights=node_weights,
    )
    if node_weights is not None:
        total_weight = float(node_weights.sum())
        targets = np.full(q, total_weight / float(q) if q > 0 else 0.0, dtype=np.float64)
    else:
        targets = _target_counts(len(assignment), q).astype(np.float64)
    w = (lambda v: node_weights[v]) if node_weights is not None else (lambda v: 1.0)

    # ── Build tracking structures (FM-style) ─────────────────────────────
    he_pins = np.zeros((len(hyperedges), q), dtype=np.int32)
    node_to_he = [[] for _ in range(num_nodes)]
    for e_idx, he in enumerate(hyperedges):
        for v in he:
            if v < num_nodes:
                he_pins[e_idx][assignment[v]] += 1
                node_to_he[v].append(e_idx)

    # ── Compute base cut ─────────────────────────────────────────────────
    hyperedge_weights = [1.0] * len(hyperedges)
    current_cut = 0.0
    for e_idx in range(len(hyperedges)):
        num_groups = int(np.count_nonzero(he_pins[e_idx]))
        if num_groups > 1:
            current_cut += (num_groups - 1) * hyperedge_weights[e_idx]

    for _ in range(max(1, assignment.size)):
        over = np.where(counts > targets)[0]
        under = np.where(counts < targets)[0]
        if len(over) == 0 or len(under) == 0:
            break
        moved = False
        candidates = np.where(np.isin(assignment, over))[0]
        rng.shuffle(candidates)
        for v in candidates:
            vw = w(v)
            old = int(assignment[v])
            best_g = None
            best_gain = -float('inf')
            for g in under:
                g = int(g)
                if g == old:
                    continue
                if counts[g] + vw > targets[g]:
                    continue
                # FM-style delta evaluation — O(deg(v)), no copy needed
                gain = 0.0
                for e_idx in node_to_he[v]:
                    pins = he_pins[e_idx]
                    wgt = hyperedge_weights[e_idx]
                    if pins[old] == 1:
                        gain += wgt
                    if pins[g] == 0:
                        gain -= wgt
                if gain > best_gain:
                    best_gain = gain
                    best_g = g
            if best_g is not None:
                # ── Apply move ──
                assignment[v] = best_g
                counts[old] -= vw
                counts[best_g] += vw
                current_cut -= best_gain
                # ── Update tracking structures ──
                for e_idx in node_to_he[v]:
                    he_pins[e_idx][old] -= 1
                    he_pins[e_idx][best_g] += 1
                moved = True
                break
        if not moved:
            break
    return assignment


def _partition_summary(assignment, q=None, node_weights=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.size == 0:
        return 0, np.zeros(0, dtype=int), 0.0
    if q is None:
        q = int(assignment.max()) + 1
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
        counts = np.zeros(q, dtype=np.float64)
        np.add.at(counts, assignment, node_weights)
        total = float(node_weights.sum())
    else:
        counts = np.bincount(assignment, minlength=q).astype(np.float64)
        total = float(assignment.size)
    ideal = total / float(q) if q > 0 else 0.0
    imb = float(np.max(np.abs(counts - ideal) / ideal)) if ideal > 0 else 0.0
    return q, counts, imb


def _assignment_cache_key(assignment):
    assignment = np.asarray(assignment, dtype=np.int64)
    return assignment.shape, assignment.tobytes()


def _cached_cut_and_imbalance(assignment, hyperedges, cache=None, node_weights=None):
    from src.partition.hyper_utils import evaluate_kahypar_cut_value
    if cache is None:
        cache = {}
    key = _assignment_cache_key(assignment)
    if key not in cache:
        assignment_arr = np.asarray(assignment, dtype=np.int64)
        cut = evaluate_kahypar_cut_value(assignment_arr, hyperedges, [1.0] * len(hyperedges))[0]
        _, _, imb = _partition_summary(assignment_arr, node_weights=node_weights)
        cache[key] = (float(cut), float(imb))
    return cache[key]


def _refine_mcts(assignment, hyperedges, q, num_rollouts=16, depth=3, seed=None,
                 max_imbalance=0.05, verbose=False, metrics_cache=None,
                 node_weights=None):
    """Monte-Carlo style refinement via randomized move simulations."""
    rng = np.random.default_rng(seed)
    best = np.asarray(assignment, dtype=np.int64).copy()
    base = best.copy()
    if metrics_cache is None:
        metrics_cache = {}
    best_score, _ = _cached_cut_and_imbalance(best, hyperedges, metrics_cache, node_weights=node_weights)
    q = int(q) if q is not None else (int(best.max()) + 1 if best.size else 2)
    if best.size:
        node_to_he = [[] for _ in range(best.size)]
        for e_idx, he in enumerate(hyperedges):
            for v in he:
                if 0 <= v < best.size:
                    node_to_he[v].append(e_idx)
    if verbose:
        _, _, imb = _partition_summary(best, q=q, node_weights=node_weights)
        print(f"[refine:mcts] start rollouts={num_rollouts} depth={depth} cut={best_score} imb={imb:.4f}")

    # Calculate boundary vertices ONCE before rollouts begin
    boundary_vertices = []
    for v in range(best.size):
        for e_idx in node_to_he[v]:
            if len(set(best[u] for u in hyperedges[e_idx] if 0 <= u < best.size)) > 1:
                boundary_vertices.append(v)
                break

    if not boundary_vertices:
        return best

    boundary_vertices = np.asarray(boundary_vertices, dtype=np.int64)

    for _ in range(max(1, int(num_rollouts))):
        cand = best.copy()
        for _step in range(max(1, int(depth))):
            v = int(boundary_vertices[int(rng.integers(0, boundary_vertices.size))])
            old = int(cand[v])
            new_g = int(rng.integers(0, q - 1))
            if new_g >= old:
                new_g += 1
            if new_g != old:
                cand[v] = new_g
        score, _ = _cached_cut_and_imbalance(cand, hyperedges, metrics_cache, node_weights=node_weights)
        if score < best_score:
            best_score = score
            best = cand
    if _partition_summary(best, q=q, node_weights=node_weights)[2] > max_imbalance:
        best = _repair_balance_fast(best, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                     node_weights=node_weights)
    if verbose:
        _, _, imb = _partition_summary(best, q=q, node_weights=node_weights)
        print(f"[refine:mcts] done cut={best_score} imb={imb:.4f}")
    return best


def _refine_evolution(assignment, hyperedges, q, population_size=8, generations=5,
                      mutation_rate=0.1, seed=None, max_imbalance=0.05,
                      verbose=False, metrics_cache=None, node_weights=None):
    """Small evolutionary search over discrete assignments."""
    rng = np.random.default_rng(seed)
    base = np.asarray(assignment, dtype=np.int64)
    if metrics_cache is None:
        metrics_cache = {}
    q = int(q) if q is not None else (int(base.max()) + 1 if base.size else 2)
    base_score, _ = _cached_cut_and_imbalance(base, hyperedges, metrics_cache, node_weights=node_weights)
    _, _, base_imb = _partition_summary(base, q=q, node_weights=node_weights)
    low_cut_mode = base_score < 200
    if low_cut_mode:
        mutation_rate = min(float(mutation_rate), 0.01)
        generations = min(int(generations), 3)
    if verbose:
        print(f"[refine:evolution] start pop={population_size} gens={generations} cut={base_score} imb={base_imb:.4f}")
    population = [base.copy() for _ in range(max(1, int(population_size)))]
    mutant_count = max(1, len(population) // 4)
    for idx in range(1, min(len(population), mutant_count + 1)):
        cand = base.copy()
        mask = rng.random(cand.shape[0]) < float(mutation_rate)
        if mask.any():
            cand[mask] = rng.integers(0, q, size=int(mask.sum()))
            if _partition_summary(cand, q=q, node_weights=node_weights)[2] > max_imbalance:
                cand = _repair_balance_fast(cand, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                             node_weights=node_weights)
        population[idx] = cand
    for _gen in range(max(1, int(generations))):
        scored = []
        for cand in population:
            score, _ = _cached_cut_and_imbalance(cand, hyperedges, metrics_cache, node_weights=node_weights)
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] > base_score:
            scored = [(base_score, base.copy())] + scored
        elites = [base.copy()]
        elites.extend(cand.copy() for _, cand in scored[: max(1, len(scored) // 3)])
        next_population = elites[:]
        while len(next_population) < len(population):
            p1 = elites[int(rng.integers(0, len(elites)))]
            p2 = elites[int(rng.integers(0, len(elites)))]
            child = np.where(rng.random(base.shape[0]) < 0.5, p1, p2).copy()
            mut_mask = rng.random(child.shape[0]) < float(mutation_rate)
            if mut_mask.any():
                child[mut_mask] = rng.integers(0, q, size=int(mut_mask.sum()))
                if _partition_summary(child, q=q, node_weights=node_weights)[2] > max_imbalance:
                    child = _repair_balance_fast(child, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                                 node_weights=node_weights)
            next_population.append(child)
        population = next_population
    scored = [(_cached_cut_and_imbalance(cand, hyperedges, metrics_cache, node_weights=node_weights)[0], cand) for cand in population]
    scored.sort(key=lambda x: x[0])
    best_score, best = scored[0]
    if best_score > base_score:
        best = base.copy()
        best_score = base_score
    if _partition_summary(best, q=q, node_weights=node_weights)[2] > max_imbalance:
        best = _repair_balance_fast(best, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                     node_weights=node_weights)
    if verbose:
        _, _, imb = _partition_summary(best, q=q, node_weights=node_weights)
        print(f"[refine:evolution] done cut={best_score} imb={imb:.4f}")
    return best


def _refine_hybrid(
    assignment, hyperedges, q,
    mode_cycle=('mcts', 'evolution', 'flow'),
    rounds=3, seed=None, max_imbalance=0.05,
    flow_passes=3, mcts_rollouts=16, mcts_depth=3,
    evolution_population=8, evolution_generations=5, evolution_mutation=0.1,
    skip_exploration_if_good=True, verbose=False,
    node_weights=None,
):
    """Hybrid refinement pipeline (MCTS / evolution / flow).

    Parameters
    ----------
    node_weights : np.ndarray or None
        Per-vertex weights for weighted balance computation.
    """
    refined = np.asarray(assignment, dtype=np.int64).copy()
    if q is None:
        q = int(refined.max()) + 1 if refined.size else 2

    metrics_cache = {}

    def evaluate(candidate):
        return _cached_cut_and_imbalance(candidate, hyperedges, metrics_cache,
                                          node_weights=node_weights)

    def ensure_balanced(candidate):
        candidate = np.asarray(candidate, dtype=np.int64).copy()
        if _partition_summary(candidate, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return candidate

        # ── Attempt 1: cut-aware repair ──
        repaired = _repair_balance(candidate, hyperedges, max_imbalance=max_imbalance,
                                    seed=seed, q=q, node_weights=node_weights)
        if _partition_summary(repaired, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return repaired

        # ── Attempt 2: fast repair (more passes) + cut-aware repair ──
        repaired = _repair_balance_fast(
            repaired, hyperedges, max_imbalance=max_imbalance,
            seed=seed, q=q, node_weights=node_weights,
            max_iterations_mult=10,  # more aggressive with heavy coarse vertices
        )
        repaired = _repair_balance(repaired, hyperedges, max_imbalance=max_imbalance,
                                    seed=seed, q=q, node_weights=node_weights)
        if _partition_summary(repaired, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return repaired

        # ── Attempt 3: fast repair only (most aggressive) ──
        repaired = _repair_balance_fast(
            repaired, hyperedges, max_imbalance=max_imbalance,
            seed=seed, q=q, node_weights=node_weights,
            max_iterations_mult=50,
        )
        if _partition_summary(repaired, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return repaired

        # ── Fallback: best effort — warn but do not crash.
        # Intermediate V-cycle levels will be re-refined at the next finer
        # level anyway, so a minor balance violation is tolerable.
        if verbose:
            _, _, imb = _partition_summary(repaired, q=q, node_weights=node_weights)
            print(f"  [warn] ensure_balanced: best imb={imb:.4f} > {max_imbalance}, continuing")
        return repaired

    if verbose:
        cut, _ = evaluate(refined)
        _, counts, imb = _partition_summary(refined, q=q, node_weights=node_weights)
        print(f"[refine:hybrid] start q={q} cut={cut} counts={counts.tolist()} imb={imb:.4f}")

    refined = ensure_balanced(refined)
    best = refined.copy()
    best_cut, best_imb = evaluate(best)

    def maybe_repair_and_accept(candidate):
        nonlocal best, best_cut, best_imb
        cand = np.asarray(candidate, dtype=np.int64).copy()
        cand = ensure_balanced(cand)
        cand_cut, cand_imb = evaluate(cand)
        if cand_cut < best_cut or (cand_cut == best_cut and cand_imb <= best_imb):
            best = cand.copy()
            best_cut = float(cand_cut)
            best_imb = float(cand_imb)
            return cand
        return best.copy()

    dynamic_good_cut_threshold = max(1.0, 0.1 * float(best_cut))
    good_initial = skip_exploration_if_good and best_cut <= dynamic_good_cut_threshold and best_imb <= float(max_imbalance)
    effective_mode_cycle = ('flow',) if good_initial else tuple(mode_cycle)
    effective_rounds = 1 if good_initial else max(1, int(rounds))
    effective_flow_passes = 1 if good_initial else int(flow_passes)

    for round_idx in range(effective_rounds):
        if verbose:
            print(f"[refine:hybrid] round {round_idx + 1}/{int(effective_rounds)}")
        if 'mcts' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=MCTS")
            candidate = _refine_mcts(
                refined, hyperedges, q,
                num_rollouts=mcts_rollouts, depth=mcts_depth, seed=seed,
                max_imbalance=max_imbalance, verbose=verbose,
                metrics_cache=metrics_cache, node_weights=node_weights,
            )
            refined = maybe_repair_and_accept(candidate)
        if 'evolution' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=Evolution")
            candidate = _refine_evolution(
                refined, hyperedges, q,
                population_size=evolution_population,
                generations=evolution_generations,
                mutation_rate=evolution_mutation, seed=seed,
                max_imbalance=max_imbalance, verbose=verbose,
                metrics_cache=metrics_cache, node_weights=node_weights,
            )
            refined = maybe_repair_and_accept(candidate)
        if 'flow' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=Flow")
            candidate = _refine_flow(
                refined, hyperedges, q,
                max_passes=effective_flow_passes,
                max_imbalance=max_imbalance, verbose=verbose,
                node_weights=node_weights,
            )
            refined = maybe_repair_and_accept(candidate)
        if _partition_summary(refined, q=q, node_weights=node_weights)[2] > max_imbalance:
            refined = _repair_balance(refined, hyperedges, max_imbalance=max_imbalance,
                                       seed=seed, q=q, node_weights=node_weights)
        refined = maybe_repair_and_accept(refined)
        if verbose:
            print(f"[refine:hybrid] round_done cut={best_cut} counts={_partition_summary(best, q=q, node_weights=node_weights)[1].tolist()} imb={best_imb:.4f}")

    refined = best.copy()
    if _partition_summary(refined, q=q, node_weights=node_weights)[2] > max_imbalance:
        refined = _repair_balance(refined, hyperedges, max_imbalance=max_imbalance,
                                   seed=seed, q=q, node_weights=node_weights)
    refined = maybe_repair_and_accept(refined)
    if verbose:
        cut, _ = evaluate(refined)
        _, counts, imb = _partition_summary(refined, q=q, node_weights=node_weights)
        print(f"[refine:hybrid] done cut={cut} counts={counts.tolist()} imb={imb:.4f}")
    return refined


# ═════════════════════════════════════════════════════════════════════════
#  Internal helpers (moved from hyper_coarsen.py)
# ═════════════════════════════════════════════════════════════════════════


def _build_incidence(hyperedge_list, vertex_count, hyperedge_weights=None):
    vertex_to_edges = [set() for _ in range(vertex_count)]
    edge_vertices = []
    edge_weights = []
    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedge_list)
    for he, weight in zip(hyperedge_list, hyperedge_weights):
        verts = []
        seen = set()
        for v in he:
            if 0 <= v < vertex_count and v not in seen:
                verts.append(int(v))
                seen.add(int(v))
        if len(verts) > 1:
            edge_vertices.append(verts)
            edge_weights.append(float(weight))
            for v in verts:
                vertex_to_edges[v].add(len(edge_vertices) - 1)
    return vertex_to_edges, edge_vertices, edge_weights


def _rebuild_hyperedges_from_groups(hyperedges, original_to_bucket, bucket_count):
    coarse_hyperedges = []
    for he in hyperedges:
        mapped = []
        seen = set()
        for v in he:
            if 0 <= v < len(original_to_bucket):
                c = int(original_to_bucket[v])
                if c not in seen:
                    mapped.append(c)
                    seen.add(c)
        if len(mapped) > 1:
            coarse_hyperedges.append(mapped)
    return coarse_hyperedges


def _lsh_bucketize_vertices(
    hyperedges, num_nodes, target_buckets=None,
    num_planes=4, num_tables=32, seed=None,
    jaccard_threshold=0.1, num_hashes=128, verbose=False,
):
    """Pre-coarsen vertices using MinHash/LSH over incident hyperedge sets."""
    if num_nodes == 0:
        return np.arange(0, dtype=np.int64), []

    incident_edge_sets = _vertex_incident_edge_sets(hyperedges, num_nodes)

    from src.partition.hyper_coarsen import _lsh_groups_from_incident_sets
    groups = _lsh_groups_from_incident_sets(
        incident_edge_sets,
        num_hashes=num_hashes,
        num_bands=max(1, int(num_tables)),
        rows_per_band=max(1, int(num_planes)),
        target_buckets=max(1, int(target_buckets)) if target_buckets is not None else max(1, int(num_nodes // 4)),
        threshold=float(jaccard_threshold),
        min_threshold=0.02,
        seed=seed,
        verbose=verbose,
    )

    original_to_bucket = np.empty(num_nodes, dtype=np.int64)
    for idx, verts in enumerate(groups):
        for v in verts:
            original_to_bucket[v] = idx
    return original_to_bucket, groups


def _vertex_incident_edge_sets(hyperedges, num_nodes):
    incident = [set() for _ in range(num_nodes)]
    for eid, he in enumerate(hyperedges):
        for v in he:
            if 0 <= v < num_nodes:
                incident[v].add(eid)
    return incident


# ═════════════════════════════════════════════════════════════════════════
#  V-Cycle uncoarsening helper
# ═════════════════════════════════════════════════════════════════════════


def vcycle_uncoarsen(
    coarse_assignment,
    hierarchy_stack,
    original_hyperedges,
    q,
    refine_solver: HyperRefineSolver,
    verbose: bool = True,
    *,
    node_weights=None,
    hyperedge_weights=None,
) -> np.ndarray:
    """Multilevel V-Cycle: iteratively project and refine through the hierarchy.

    Takes an initial partition on the coarsest level, then walks back up
    through the saved hierarchy levels (finest-first stack).  At each step:

        1. Project the current assignment to the next finer level via ``remap``.
        2. Run ``refine_solver.refine()`` on that finer-level hypergraph.

    After all hierarchy levels are processed, a final refinement pass is
    run on the **original** hypergraph.

    Parameters
    ----------
    coarse_assignment : np.ndarray
        Partition assignment at the coarsest level (output of
        ``FemCoarsenSolver.initial_partition`` or ``initial_partition_greedy``).
        Length must equal the number of coarse nodes.
    hierarchy_stack : list of dict
        The ``hierarchy_stack`` returned by ``KahyparLikeSolver.coarsen()``.
        Each entry has keys ``hyperedges``, ``remap``, ``num_nodes``, ``groups``.
        Ordered from finest (first contraction) to coarsest (last contraction).
    original_hyperedges : list of list of int
        The original (finest-level) hyperedges — used for the final
        refinement step after all hierarchy levels are processed.
    q : int
        Number of blocks (partitions).
    refine_solver : HyperRefineSolver
        Refinement solver (FM / hybrid) to apply at each level.
    verbose : bool
        If True, print cut / imbalance after each level.
    node_weights, hyperedge_weights : array-like or None
        Weights on the original hypergraph. Supply these explicitly for an
        empty hierarchy. Otherwise stored level weights are used when they
        can be aligned safely; inconsistent metadata raises. Edge weights
        remain attached to individual hyperedges, including parallel nets.

    Returns
    -------
    np.ndarray
        Final assignment on the original hypergraph.
    """
    from src.partition.hyper_quotient import connectivity_cost
    from src.partition.hyper_refine_contract import (
        normalized_hyperedges, validated_weights,
    )
    assignment = np.asarray(coarse_assignment, dtype=np.int64).copy()
    n_levels = len(hierarchy_stack)

    # The incidence list need not mention isolated vertices.
    num_original_nodes = hierarchy_stack[0]['num_nodes'] if hierarchy_stack else len(assignment)

    original_hyperedges = normalized_hyperedges(original_hyperedges, num_original_nodes)

    def aligned_edge_weights(target_edges, source_edges, source_weights):
        # Coarsening can discard singleton nets; their cut contribution is
        # identically zero. Preserve the order/multiplicity of all other nets.
        target = [(index, tuple(sorted(edge))) for index, edge in enumerate(target_edges)
                  if len(edge) > 1]
        source = [(tuple(sorted(set(edge))), float(weight))
                  for edge, weight in zip(source_edges, source_weights) if len(set(edge)) > 1]
        if len(target) != len(source) or any(a[1] != b[0] for a, b in zip(target, source)):
            raise ValueError(
                'cannot safely align stored hyperedge_weights with original hyperedges; '
                'pass original hyperedge_weights explicitly'
            )
        result = np.ones(len(target_edges), dtype=np.float64)
        for (index, _), (_, weight) in zip(target, source):
            result[index] = weight
        return result

    first_level = hierarchy_stack[0] if hierarchy_stack else None
    if node_weights is None and first_level is not None:
        node_weights = first_level.get('node_weights', [len(g) for g in first_level['groups']])
    orig_weights = validated_weights(node_weights, num_original_nodes, 'original node_weights')
    if first_level is not None and 'node_weights' in first_level:
        stored_nodes = validated_weights(first_level['node_weights'], num_original_nodes,
                                         'first-level node_weights')
        if not np.allclose(orig_weights, stored_nodes, rtol=1e-12, atol=1e-12):
            raise ValueError('original node_weights disagree with first hierarchy level')
    if hyperedge_weights is None and first_level is not None and 'hyperedge_weights' in first_level:
        stored_edges = validated_weights(first_level['hyperedge_weights'],
                                         len(first_level['hyperedges']),
                                         'first-level hyperedge_weights')
        hyperedge_weights = aligned_edge_weights(original_hyperedges,
                                                 first_level['hyperedges'], stored_edges)
    elif hyperedge_weights is None and any(
            'hyperedge_weights' in level and np.any(np.asarray(level['hyperedge_weights']) != 1.0)
            for level in hierarchy_stack):
        raise ValueError('nonunit hierarchy weights require original hyperedge_weights or first-level metadata')
    orig_edge_weights = validated_weights(hyperedge_weights, len(original_hyperedges),
                                         'original hyperedge_weights')

    # Walk back up: coarsest → finest
    for level_idx, level in enumerate(reversed(hierarchy_stack)):
        fine_hyperedges = level['hyperedges']
        remap = level['remap']
        fine_n = level['num_nodes']

        # ── Extract node weights for this level (cluster sizes) ──
        fine_weights = validated_weights(
            level.get('node_weights', [len(g) for g in level['groups']]), fine_n,
            'hierarchy node_weights',
        )
        fine_hyperedges = normalized_hyperedges(fine_hyperedges, fine_n)
        if 'hyperedge_weights' in level:
            fine_edge_weights = validated_weights(level['hyperedge_weights'], len(fine_hyperedges),
                                                  'hierarchy hyperedge_weights')
        elif np.any(orig_edge_weights != 1.0):
            if fine_n != num_original_nodes:
                raise ValueError('weighted V-cycle requires hyperedge_weights at each hierarchy level')
            fine_edge_weights = aligned_edge_weights(fine_hyperedges, original_hyperedges,
                                                      orig_edge_weights)
        else:
            fine_edge_weights = np.ones(len(fine_hyperedges), dtype=np.float64)

        # ── Project: fine_assignment[v] = coarse_assignment[remap[v]] ──
        projected = np.array([assignment[remap[v]] for v in range(fine_n)], dtype=np.int64)

        if verbose:
            cut = connectivity_cost(projected, fine_hyperedges, fine_edge_weights)
            loads = np.bincount(projected, weights=fine_weights, minlength=q)
            lvl_label = n_levels - level_idx
            print(f'  [V-cycle] level {lvl_label}/{n_levels}: projected  cut={cut}, loads={loads.tolist()}')

        # ── Refine at this level (with weighted balance) ──
        assignment = refine_solver.refine(projected, fine_hyperedges, q,
                                          node_weights=fine_weights,
                                          hyperedge_weights=fine_edge_weights)

        if verbose:
            cut = connectivity_cost(assignment, fine_hyperedges, fine_edge_weights)
            loads = np.bincount(assignment, weights=fine_weights, minlength=q)
            lvl_label = n_levels - level_idx
            print(f'  [V-cycle] level {lvl_label}/{n_levels}: refined   cut={cut}, loads={loads.tolist()}')

    # ── Final refinement on the original hypergraph ──
    if verbose:
        cut = connectivity_cost(assignment, original_hyperedges, orig_edge_weights)
        loads = np.bincount(assignment, weights=orig_weights, minlength=q)
        print(f'  [V-cycle] original (pre-refine): cut={cut}, loads={loads.tolist()}')

    assignment = refine_solver.refine(assignment, original_hyperedges, q,
                                      node_weights=orig_weights,
                                      hyperedge_weights=orig_edge_weights)

    if verbose:
        cut = connectivity_cost(assignment, original_hyperedges, orig_edge_weights)
        loads = np.bincount(assignment, weights=orig_weights, minlength=q)
        print(f'  [V-cycle] original (post-refine): cut={cut}, loads={loads.tolist()}')

    return assignment
