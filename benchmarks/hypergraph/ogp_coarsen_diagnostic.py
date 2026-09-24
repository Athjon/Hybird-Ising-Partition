"""Archive existing exact small-instance coarsening baselines reproducibly.

This does not measure OGP. It isolates contraction loss and coarse balance
feasibility using the existing eight-vertex enumeration benchmarks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_hashes():
    paths = [
        "benchmarks/hypergraph/ogp_coarsen_diagnostic.py",
        "benchmarks/hypergraph/coarsen_quotient.py",
        "benchmarks/hypergraph/coarsen_feasibility.py",
        "src/hyper_solver.py", "src/partition/hyper_quotient.py",
        "src/partition/hyper_utils.py", "tests/test_hyper_quotient.py",
    ]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def parse_records(stdout):
    records = []
    for line in stdout.splitlines():
        if not line.startswith("family="):
            continue
        record = {}
        for field in line.split():
            key, value = field.split("=", 1)
            try:
                parsed = float(value)
                record[key] = (int(parsed) if parsed.is_integer() else parsed) if math.isfinite(parsed) else None
            except ValueError:
                record[key] = value
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/hypergraph/results/ogp-diagnostic-20260924/coarsen")
    args = parser.parse_args()
    if args.instances < 1:
        parser.error("--instances must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        parser.error("summary.json already exists; select a new output directory")

    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    start_hashes = source_hashes()
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "python_version": sys.version,
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_at_start": git("status", "--short"),
        "source_sha256_before": start_hashes,
        "thread_environment": {k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "working_directory": str(ROOT),
        "invocation": shlex.join([sys.executable, *sys.argv]),
    }
    write_json(out / "environment.json", metadata)
    jobs = [
        ("quotient", ["benchmarks/hypergraph/coarsen_quotient.py", "--instances", str(args.instances), "--seed", str(args.seed), "--num-pilots", "4", "--boundary-weight", "0.4", "--pilot-refine-passes", "2", "--planted-probability", "0.85"]),
        ("feasibility", ["benchmarks/hypergraph/coarsen_feasibility.py", "--instances", str(args.instances), "--seed", str(args.seed)]),
        ("tests", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_hyper_quotient.py"]),
    ]
    commands = []
    summaries = {}
    for name, argv in jobs:
        command = [sys.executable, *argv]
        print(f"Running {name}: {shlex.join(command)}", flush=True)
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        (out / f"{name}.log").write_text(result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""))
        commands.append({"name": name, "argv": command, "shell_display": shlex.join(command), "returncode": result.returncode, "elapsed_seconds": elapsed})
        write_json(out / "commands.json", commands)
        summaries[name] = parse_records(result.stdout)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise SystemExit(f"{name} failed with exit code {result.returncode}; see {out / (name + '.log')}")

    summary = {
        "config": {
            "instances_per_family": args.instances, "instance_seed_start": args.seed,
            "instance_seed_end": args.seed + args.instances - 1,
            "coarsener_seed": "trial index, 0 through instances-1, as in existing baselines",
            "families": ["random", "planted"], "vertices": 8,
            "hyperedges": 14, "edge_size_range_inclusive": [2, 4],
            "hyperedge_weights": "unit", "q": 2, "coarsen_to": 4,
            "num_pilots": 4, "boundary_weight": 0.4, "pilot_refine_passes": 2,
            "planted_within_half_probability": 0.85,
            "quotient_vertex_weights": "unit", "quotient_epsilon": 0.25,
            "feasibility_vertex_weights": "iid integers 1..4, rejection-sampled to exact balance feasibility",
            "feasibility_weight_seed": "instance seed + 100000",
            "feasibility_epsilon": 0.0,
            "optimum_method": "complete enumeration of binary assignments, evaluated on the original native km1 objective after lifting",
        },
        "quotient": summaries["quotient"], "feasibility": summaries["feasibility"],
        "tests": {"returncode": 0, "log": "tests.log"},
        "interpretation_limits": [
            "Synthetic eight-vertex instances; not a reproduction of the manuscript's full benchmark.",
            "Original and each fixed quotient are solved exactly, so finite contraction loss is not solver error.",
            "Quotient benchmark cut/loss means are on instances feasible for both compared coarseners.",
            "Feasibility benchmark cut/loss means are conditional on feasibility; compare paired statistics or rescued cases.",
            "Exact global packing guard is a small-instance oracle, not a claimed scalable algorithm.",
            "No overlap measurement, OGP theorem test, MCMC run, or FEM/SBM run is performed.",
            "Timing is informational and may be affected by concurrent local workloads.",
        ],
    }
    write_json(out / "summary.json", summary)
    metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(), source_sha256_after=source_hashes())
    metadata["sources_unchanged_during_run"] = metadata["source_sha256_after"] == start_hashes
    write_json(out / "environment.json", metadata)
    print(f"Saved {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
