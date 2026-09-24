"""Probe the independent mean-field reference on exact finite-gap instances.

Consumes saved geometry instances without regenerating them. This is a small
cross-check, not an OGP hardness theorem or a legacy HIP/FEM benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hypergraph.ogp_objective_diagnostic import (
    balanced_assignments, costs, optimize_reference, swap_descent, topk_round,
)


def masks(states):
    """Canonicalize global label flips by fixing vertex 0 to label 0."""
    states = np.asarray(states, dtype=np.int8)
    canonical = np.where(states[:, :1] == 1, 1 - states, states)
    powers = 1 << np.arange(states.shape[1], dtype=np.int64)
    return (canonical @ powers).tolist()


def stage_metrics(states, edges, optimum, optimal_masks):
    assert np.all(states.sum(-1) == states.shape[-1] // 2)
    values = costs(states, edges)[0]
    labels = masks(states)
    hits = values == optimum
    found = Counter(mask for mask, hit in zip(labels, hits) if hit)
    assert set(found).issubset(optimal_masks)
    return {
        "best_gap": float(values.min() - optimum),
        "mean_gap": float(values.mean() - optimum),
        "optimal_hits": int(hits.sum()),
        "trials": len(states),
        "optimal_hit_fraction": float(hits.mean()),
        "optimal_classes_covered": len(found),
        "total_optimal_classes": len(optimal_masks),
        "optimal_class_hit_counts": {str(mask): found.get(mask, 0) for mask in sorted(optimal_masks)},
        "canonical_masks": labels,
        "native_costs": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=ROOT / "benchmarks/hypergraph/results/ogp-diagnostic-20260924/geometry")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/hypergraph/results/ogp-diagnostic-20260924/crosscheck")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--balance-penalty", type=float, default=5.0)
    args = parser.parse_args()
    if min(args.trials, args.steps) < 1:
        parser.error("trials and steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "summary.json"
    if target.exists():
        parser.error("summary.json exists; choose a new output directory")
    torch.set_num_threads(1)
    instance_ids = ["random-n08-i003", "random-n08-i009", "random-n12-i008"]
    source_files = [Path(__file__).resolve(), ROOT / "benchmarks/hypergraph/ogp_objective_diagnostic.py"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    results = []
    for instance_id in instance_ids:
        path = args.geometry / "instances" / (instance_id + ".json")
        data = json.loads(path.read_text())
        n, seed = data["n"], data["seed"]
        edges = np.asarray(data["hyperedges"], dtype=np.int64)
        assert edges.shape[1] == 4
        assert all(w == 1 for w in data["node_weights"])
        assert all(w == 1 for w in data["hyperedge_weights"])
        exact = balanced_assignments(n)
        energies = costs(exact, edges)[0]
        oracle = dict(zip(masks(exact), energies.tolist()))
        saved = dict(zip(data["feasible_class_masks"], data["native_energy_by_feasible_mask"]))
        assert oracle == saved, "independent enumeration disagrees with saved geometry"
        optimum = float(energies.min())
        assert optimum == data["exact_optimum"]
        optimal_masks = {mask for mask, energy in oracle.items() if energy == optimum}
        assert len(optimal_masks) == 2
        threshold = next(x for x in data["thresholds"] if x["epsilon_absolute"] == 0)
        assert threshold["has_nontrivial_including_self_interior_gap"]
        record = {
            "instance_id": instance_id, "n": n, "seed": seed,
            "source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "exact_optimum": optimum, "optimal_class_masks": sorted(optimal_masks),
            "exact_geometry_revalidated": True,
            "including_self_overlap_support_at_optimum": threshold["including_self_support"],
            "finite_gap_at_optimum": threshold["interior_gaps_including_self"],
            "methods": {},
        }
        paired_initial = None
        for method in ("native", "clique", "detached_cut"):
            p, initial, elapsed = optimize_reference(edges, n, method, seed,
                args.trials, args.steps, args.learning_rate, args.balance_penalty)
            if paired_initial is None:
                paired_initial = initial
            else:
                assert np.array_equal(paired_initial, initial)
            rounded = topk_round(p)
            refined = np.stack([swap_descent(state, edges) for state in rounded])
            record["methods"][method] = {
                "topk": stage_metrics(rounded, edges, optimum, optimal_masks),
                "refined": stage_metrics(refined, edges, optimum, optimal_masks),
                "optimize_seconds": elapsed,
            }
            r = record["methods"][method]
            print(f"{instance_id} {method}: optimum={optimum:g} "
                  f"topk={r['topk']['optimal_hits']}/{args.trials} "
                  f"classes={r['topk']['optimal_classes_covered']}/2 gap={r['topk']['best_gap']:g}; "
                  f"refined={r['refined']['optimal_hits']}/{args.trials} "
                  f"classes={r['refined']['optimal_classes_covered']}/2 gap={r['refined']['best_gap']:g}", flush=True)
        record["paired_initialization_verified"] = True
        results.append(record)
    current_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    report = {
        "scope": "Independent autograd mean-field reference on three saved finite-gap instances; not legacy HIP/FEM end-to-end and not an OGP barrier test.",
        "config": {"instance_ids": instance_ids, "trials": args.trials, "steps": args.steps,
            "learning_rate": args.learning_rate, "balance_penalty": args.balance_penalty,
            "optimizer_seed": "saved geometry instance seed, identical across methods",
            "methods": ["native", "clique", "detached_cut"], "clique_pair_weight": 1 / 3,
            "rounding": "top n/2 marginals", "refinement": "strict best-improving native-cost 1-for-1 swaps"},
        "environment": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "platform": platform.platform()},
        "invocation": shlex.join([sys.executable, *sys.argv]),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256_before": source_hashes, "source_sha256_after": current_hashes,
        "sources_unchanged_during_run": source_hashes == current_hashes,
        "results": results,
        "interpretation_limits": [
            "A finite overlap support gap among exact optima is not by itself an algorithmic impossibility result.",
            "Success here concerns only three tiny selected instances, one seed and 32 restarts per method.",
            "Marginal optimization and rounding may pass through or return non-optimal states; they need not follow a path confined to optimal configurations.",
            "Covering both optimal classes is a finite discovery observation, not evidence of correct Gibbs weights or fast mixing.",
        ],
    }
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
