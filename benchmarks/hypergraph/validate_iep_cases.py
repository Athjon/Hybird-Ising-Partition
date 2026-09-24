"""Close and independently audit the complete currently usable IEP case set.

The suite audits saved production assignments for the 40 exact objective
instances, four weighted-capacity instances, and IBM01/IBM02 seeds 30--39.
It also runs the previously unmeasured ``bad_for_ec`` fixture through direct
FEM and HEM/boundary-coarsened FEM.  Geometry and coarsening ensembles are
diagnostic inputs for other stages and are deliberately not counted as IEP
performance cases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hypergraph.time_budget_hgr import read_unweighted_hgr
from src.hyper_solver import FemCoarsenSolver, KahyparLikeSolver
from src.partition.hyper_objective import capacity_limits
from src.partition.hyper_quotient import connectivity_cost


EXPECTED_INPUT_HASHES = {
    "ibm01.hgr": "40f7f7c4dfd96c06b0570f696e67b9c667ac2cbdf4d0858da5e690f4f4d5ac72",
    "ibm02.hgr": "ff09f3be9ed84a8c13257f1655555938072cdf01fae40f1548795763981eae05",
    "bad_for_ec.hgr": "33ddff8bb4c3f9ccbb36a2df1deb8184c5157d2d340a9e39a952e237caf56416",
}


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(assignment, edges, node_weights, edge_weights, q, epsilon):
    assignment = np.asarray(assignment)
    node_weights = np.asarray(node_weights, dtype=np.float64)
    edge_weights = np.asarray(edge_weights, dtype=np.float64)
    if assignment.shape != (len(node_weights),) or not np.issubdtype(assignment.dtype, np.integer):
        raise AssertionError("invalid assignment shape or dtype")
    if np.any(assignment < 0) or np.any(assignment >= q):
        raise AssertionError("assignment labels outside [0,q)")
    native = float(connectivity_cost(assignment, edges, edge_weights))
    independent = sum(
        float(weight) * max(0, len({int(assignment[vertex]) for vertex in set(edge)}) - 1)
        for edge, weight in zip(edges, edge_weights)
    )
    if not np.isclose(native, independent, atol=1e-10, rtol=1e-12):
        raise AssertionError("native km1 disagrees with independent evaluation")
    loads = np.bincount(assignment, weights=node_weights, minlength=q)
    capacity, tolerance = capacity_limits(node_weights, q, epsilon)
    return {
        "native_cut": native,
        "block_loads": loads,
        "capacity": capacity,
        "capacity_tolerance": tolerance,
        "feasible": bool(np.all(loads <= capacity + tolerance)),
        "independent_native_cut_verified": True,
    }


def exact_optimum(edges, node_weights, edge_weights, q, epsilon, *, allow_infeasible=False):
    node_weights = np.asarray(node_weights, dtype=np.float64)
    capacity, tolerance = capacity_limits(node_weights, q, epsilon)
    best = float("inf")
    feasible_states = 0
    for labels in itertools.product(range(q), repeat=len(node_weights)):
        loads = np.bincount(labels, weights=node_weights, minlength=q)
        if np.any(loads > capacity + tolerance):
            continue
        feasible_states += 1
        best = min(best, float(connectivity_cost(labels, edges, edge_weights)))
    if not np.isfinite(best) and not allow_infeasible:
        raise AssertionError("exact enumeration found no capacity-feasible assignment")
    return best, feasible_states


def audit_objective(saved_instances, repair_directory):
    inputs = [json.loads(line) for line in saved_instances.read_text().splitlines() if line.strip()]
    results = json.loads((repair_directory / "saved_instances.json").read_text())
    rows = results["rows"]
    if len(inputs) != 40 or len(rows) != 40:
        raise AssertionError("objective set must contain exactly 40 inputs and results")
    by_key = {(row["n"], row["family"], row["seed"]): row for row in rows}
    audited = []
    for item in inputs:
        key = (item["n"], item["family"], item["seed"])
        row = by_key[key]
        if row["status"] != "success":
            raise AssertionError(f"saved objective run failed: {key}")
        checked = measure(
            np.asarray(row["assignment"], dtype=np.int64), item["edges"],
            np.ones(item["n"]), np.ones(len(item["edges"])), 2, 0.0,
        )
        if not checked["feasible"] or checked["native_cut"] != float(item["optimum"]):
            raise AssertionError(f"objective assignment does not hit exact optimum: {key}")
        audited.append({"n": item["n"], "family": item["family"], "seed": item["seed"], **checked})
    return {
        "case_count": len(audited), "successful": len(audited), "feasible": len(audited),
        "exact_hits": len(audited), "input_sha256": digest(saved_instances), "rows": audited,
    }


def audit_weighted(repair_directory):
    rows = []
    for n, q in itertools.product((8, 12), (2, 3)):
        name = f"weighted-n{n:02d}-q{q}"
        item = json.loads((repair_directory / f"{name}-input.json").read_text())
        arrays = np.load(repair_directory / f"{name}-assignments.npz")
        edge_weights = np.asarray(item["hyperedge_weights"], dtype=np.float64)
        node_weights = np.asarray(item["node_weights"], dtype=np.float64)
        optimum, feasible_states = exact_optimum(item["edges"], node_weights, edge_weights, q, 0.0)
        checked = measure(arrays["fem_final_assignment"], item["edges"], node_weights, edge_weights, q, 0.0)
        if not checked["feasible"]:
            raise AssertionError(f"weighted FEM result is infeasible: {name}")
        rows.append({
            "name": name, "n": n, "q": q, "exact_optimum": optimum,
            "exact_feasible_states": feasible_states,
            "fem_gap": checked["native_cut"] - optimum,
            "input_sha256": digest(repair_directory / f"{name}-input.json"), **checked,
        })
    return {
        "case_count": len(rows), "successful": len(rows),
        "feasible": sum(row["feasible"] for row in rows),
        "exact_hits": sum(np.isclose(row["fem_gap"], 0.0) for row in rows), "rows": rows,
    }


def audit_ibm(input_directory, multiseed_directory):
    summary = json.loads((multiseed_directory / "summary.json").read_text())
    units = summary["units"]
    rows = []
    for name in ("ibm01", "ibm02"):
        path = input_directory / f"{name}.hgr"
        if digest(path) != EXPECTED_INPUT_HASHES[path.name]:
            raise AssertionError(f"unexpected input hash for {path.name}")
        n, edges = read_unweighted_hgr(path)
        node_weights, edge_weights = np.ones(n), np.ones(len(edges))
        selected = sorted((unit for unit in units if unit["instance"] == name), key=lambda unit: unit["seed"])
        if [unit["seed"] for unit in selected] != list(range(30, 40)):
            raise AssertionError(f"{name} does not contain exactly seeds 30--39")
        for unit in selected:
            run = unit["single_runs"]["fem"]
            if run["status"] != "success":
                raise AssertionError(f"{name} seed {unit['seed']} FEM failed")
            arrays = np.load(multiseed_directory / f"{name}-seed{unit['seed']}" / "assignments.npz")
            final = measure(arrays["single_fem_final_assignment"], edges, node_weights, edge_weights, 4, 0.03)
            lifted = arrays["single_fem_coarse_assignment"][arrays["original_to_coarse"]]
            coarse_lifted = measure(lifted, edges, node_weights, edge_weights, 4, 0.03)
            if not final["feasible"] or not coarse_lifted["feasible"]:
                raise AssertionError(f"{name} seed {unit['seed']} contains an infeasible FEM assignment")
            if not np.isclose(final["native_cut"], run["final"]["native_cut"]):
                raise AssertionError(f"{name} seed {unit['seed']} recorded cut disagrees")
            rows.append({
                "instance": name, "seed": unit["seed"], "coarse_lifted": coarse_lifted,
                "final": final, "input_sha256": digest(path),
            })
    return {
        "case_count": len(rows), "successful": len(rows),
        "feasible_coarse_and_final": len(rows), "rows": rows,
    }


def run_bad_for_ec(path):
    if digest(path) != EXPECTED_INPUT_HASHES[path.name]:
        raise AssertionError("unexpected bad_for_ec input hash")
    n, edges = read_unweighted_hgr(path)
    node_weights, edge_weights = np.ones(n), np.ones(len(edges))
    exact, exact_states = exact_optimum(edges, node_weights, edge_weights, 2, 0.03)
    rows = []
    for seed in range(30, 40):
        direct_solver = FemCoarsenSolver()
        direct = direct_solver.initial_partition(
            edges, node_weights, 2, hyperedge_weights=edge_weights, epsilon=0.03,
            seed=seed, num_trials=32, num_steps=300, dev="cpu",
        )
        direct_measure = measure(direct, edges, node_weights, edge_weights, 2, 0.03)
        record = {"seed": seed, "direct_fem": {**direct_measure, "gap": direct_measure["native_cut"] - exact}}
        for mode in ("hem", "boundary"):
            for guarded in (False, True):
                key = f"{mode}_{'guarded' if guarded else 'local'}"
                coarse = KahyparLikeSolver().coarsen(
                    edges, n, 2, node_weights=node_weights, hyperedge_weights=edge_weights,
                    coarsen_to=4, seed=seed, epsilon=0.03, score_mode=mode,
                    enforce_balance_cap=True, enforce_global_feasibility=guarded,
                    global_feasibility_max_nodes=20,
                    num_pilots=4, pilot_refine_passes=2,
                )
                coarse_nodes = coarse["coarse_node_weights"].numpy()
                coarse_edges = coarse["coarse_hyperedges"]
                coarse_edge_weights = np.asarray(coarse["coarse_hyperedge_weights"], dtype=np.float64)
                coarse_exact, coarse_states = exact_optimum(
                    coarse_edges, coarse_nodes, coarse_edge_weights, 2, 0.03,
                    allow_infeasible=True,
                )
                if not np.isfinite(coarse_exact):
                    record[key] = {
                        "status": "coarse_infeasible", "coarse_nodes": len(coarse_nodes),
                        "coarse_exact_feasible_states": 0,
                    }
                    continue
                solver = FemCoarsenSolver()
                assignment = solver.initial_partition(
                    coarse_edges, coarse_nodes, 2, hyperedge_weights=coarse_edge_weights,
                    epsilon=0.03, seed=seed, num_trials=32, num_steps=300, dev="cpu",
                )
                coarse_measure = measure(
                    assignment, coarse_edges, coarse_nodes, coarse_edge_weights, 2, 0.03,
                )
                lifted_measure = measure(
                    assignment[coarse["original_to_coarse"]], edges,
                    node_weights, edge_weights, 2, 0.03,
                )
                if not np.isclose(coarse_measure["native_cut"], lifted_measure["native_cut"]):
                    raise AssertionError("quotient and lifted bad_for_ec costs disagree")
                record[key] = {
                    "status": "success", "coarse_nodes": len(coarse_nodes),
                    "coarse_exact_optimum": coarse_exact,
                    "coarse_exact_feasible_states": coarse_states,
                    "coarse_fem_gap": coarse_measure["native_cut"] - coarse_exact,
                    "original_exact_gap_after_lift": lifted_measure["native_cut"] - exact,
                    "coarse": coarse_measure, "lifted": lifted_measure,
                }
        rows.append(record)
    for row in rows:
        if not row["direct_fem"]["feasible"]:
            raise AssertionError(f"bad_for_ec seed {row['seed']} direct FEM is infeasible")
        for method in ("hem_guarded", "boundary_guarded"):
            if row[method]["status"] != "success" or not row[method]["lifted"]["feasible"]:
                raise AssertionError(f"bad_for_ec seed {row['seed']} {method} did not restore feasibility")
    method_summary = {}
    for method in ("hem_local", "hem_guarded", "boundary_local", "boundary_guarded"):
        successful = [row[method] for row in rows if row[method]["status"] == "success"]
        method_summary[method] = {
            "successful": len(successful),
            "coarse_infeasible": len(rows) - len(successful),
            "coarse_fem_exact_hits": sum(np.isclose(item["coarse_fem_gap"], 0.0) for item in successful),
            "original_exact_hits_after_lift": sum(
                np.isclose(item["original_exact_gap_after_lift"], 0.0) for item in successful
            ),
            "mean_lifted_cut_successes_only": (
                float(np.mean([item["lifted"]["native_cut"] for item in successful]))
                if successful else None
            ),
        }
    return {
        "case_count": len(rows), "seed_range": [30, 39], "input_sha256": digest(path),
        "n": n, "m": len(edges), "q": 2, "epsilon": 0.03,
        "exact_optimum": exact, "exact_feasible_states": exact_states,
        "direct_and_guarded_methods_all_seeds_feasible": True,
        "direct_exact_hits": sum(np.isclose(row["direct_fem"]["gap"], 0.0) for row in rows),
        "local_coarse_infeasible": {
            method: sum(row[method]["status"] == "coarse_infeasible" for row in rows)
            for method in ("hem_local", "boundary_local")
        },
        "guarded_coarse_exact_hits": {
            method: sum(np.isclose(row[method]["coarse_fem_gap"], 0.0) for row in rows)
            for method in ("hem_guarded", "boundary_guarded")
        },
        "method_summary": method_summary,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=ROOT / "benchmarks/hypergraph/data/hmetis")
    parser.add_argument("--repair-directory", type=Path, default=ROOT / "benchmarks/hypergraph/results/fem-repair-20260924")
    parser.add_argument("--multiseed-directory", type=Path, default=ROOT / "benchmarks/hypergraph/results/fem-multiseed-20260924-v2")
    parser.add_argument("--saved-instances", type=Path, default=ROOT / "benchmarks/hypergraph/results/ogp-diagnostic-20260924/objective/instances.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": {
            "included": ["objective exact set (40)", "weighted-capacity set (4)",
                         "IBM01/IBM02 seeds 30--39 (20)", "bad_for_ec seeds 30--39"],
            "excluded": {
                "geometry 60": "OGP/overlap diagnostic, not an IEP performance suite",
                "coarsening 400": "coarsening representational/feasibility diagnostic, not an IEP solver suite",
                "finite-gap 3": "subset of geometry 60, not independent cases",
                "legacy named cases": "raw inputs are unavailable locally",
            },
        },
        "input_hashes": {name: digest(args.input_directory / name) for name in EXPECTED_INPUT_HASHES},
    }
    save(args.output / "progress.json", report)
    try:
        report["objective"] = audit_objective(args.saved_instances, args.repair_directory)
        save(args.output / "progress.json", report)
        report["weighted"] = audit_weighted(args.repair_directory)
        save(args.output / "progress.json", report)
        report["ibm"] = audit_ibm(args.input_directory, args.multiseed_directory)
        save(args.output / "progress.json", report)
        report["bad_for_ec"] = run_bad_for_ec(args.input_directory / "bad_for_ec.hgr")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        save(args.output / "summary.json", report)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_seconds"] = time.perf_counter() - started
        save(args.output / "summary.json", report)
    print(json.dumps({
        "status": report["status"], "objective": report["objective"]["case_count"],
        "weighted": report["weighted"]["case_count"], "ibm": report["ibm"]["case_count"],
        "bad_for_ec": report["bad_for_ec"]["case_count"], "seconds": report["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
