# fem-partition

A Python library for graph and hypergraph partitioning, with multi-level pipelines, QUBO/Ising backends, and native higher-order FEM objectives.

## Problem Types

| Type | Description |
|------|-------------|
| **Balanced minimum cut** (normal graph) | Partition graph vertices into `k` equal-weight blocks while minimizing the cut edges |
| **Balanced minimum cut** (hypergraph) | Minimize weighted connectivity-minus-one (km1), with a maximum weighted load per block; km1 equals cut-net only for two blocks |
| **Max-cut** | Partition graph vertices into two blocks maximizing the cut edges |
| **Max-SAT** | Approximate maximum satisfiability via QUBO encoding |

## Solvers

QUBO/Ising solvers (FEM, SBM) are provided by the external
**[qubo-solver](https://github.com/yao-baijian/qubo-solver)** submodule,
cloned at ``lib/qubo-solver/``.  Import directly:

```python
from fem import FemSolver       # mean-field annealing
from sbm import SbmSolver       # simulated bifurcation
from sbm import BaseSolver, BSBStrategy, GSBMixin  # composable API
```

### Hypergraph Solvers (built-in — ``src/hyper_solver.py``)

| Solver | Role | Description |
|--------|------|-------------|
| **KahyparLikeSolver** | Coarsening | HEM (heavy-edge matching) coarsening with optional LSH pre-coarsening; saves a `hierarchy_stack` for V-Cycle |
| **FemCoarsenSolver** | Initial partition | Differentiable native km1 expectation, capacity-checked rounding, and native-cost selection across trials |
| **HyperRefineSolver** | Refinement | FM (greedy incremental), opt-in native FEM-IER, MCTS rollouts, or evolutionary search |

These three solvers compose into a complete hypergraph pipeline.

### KaFFPa / KaHIP, METIS

External partitioner wrappers in ``src/partition/``.

## Pipelines

### Normal Graph Pipeline

Multi-level partitioning combining coarsening, initial partitioning, and refinement via solver composition:

| Method | Family | Init Solver | Refine Solver |
|--------|--------|-------------|---------------|
| `direct_fem` | DI | FEM (on full graph) | — |
| `direct_sbm` | DI | SBM (on full graph) | — |
| `kaffpa` | DML | KaFFPa (native multi-level) | KaFFPa |
| `init_fem_refine_kaffpa` | IECM | FEM (on coarse) | KaFFPa |
| `init_sbm_refine_kaffpa` | IECM | SBM (on coarse) | KaFFPa |
| `init_kaffpa_refine_fem` | MIER | KaFFPa (on coarse) | Cyclic Expansion FEM |
| `coarse_fem_refine_kaffpa` | IECM | FEM (on coarse) | KaFFPa |
| `coarse_kaffpa_refine_fem` | MIER | KaFFPa (on coarse) | Cyclic Expansion FEM |

Pipeline families:
| Family | Meaning |
|--------|---------|
| **DI** | Direct solver on full graph (no coarsening) |
| **DML** | Native tool manages its own coarsening + refinement |
| **IECM** | Coarsen → FEM/SBM init → External refine |
| **MIER** | External init on coarse → Cyclic Expansion FEM refine |

### Hypergraph Pipeline

| Stage | Method | Description |
|-------|--------|-------------|
| **Coarsening** | HEM / LSH | Heavy-edge matching directly on hyperedges; optional MinHash/LSH pre-coarsening. Intermediate levels saved in a `hierarchy_stack`. |
| **Initial partition** | Greedy / FEM / PUBO | Initial assignment on the coarsest level. |
| **V-Cycle uncoarsening** | Iterative projection + refinement | Pop levels off the `hierarchy_stack` one-by-one: project the current assignment to the next finer level, then refine immediately. |
| **Refinement** | FM / FEM-IER / MCTS / Evolution | Native FEM-IER can compose with FM; MCTS/evolution use the separate legacy hybrid path. |

```python
# Usage:
res = kahypar_solver.coarsen(
    hyperedges, num_nodes, q, node_weights=node_weights,
    hyperedge_weights=edge_weights, epsilon=0.03,
)
fem_part = fem_solver.initial_partition(
    res['coarse_hyperedges'], res['coarse_node_weights'], q,
    hyperedge_weights=res['coarse_hyperedge_weights'],
    map_type='native', epsilon=0.03, num_trials=16, num_steps=300,
)
refine_solver.update_params(mode_cycle=('flow',), max_imbalance=0.03)
final = vcycle_uncoarsen(
    fem_part, res['hierarchy_stack'], hyperedges, q, refine_solver,
    node_weights=node_weights, hyperedge_weights=edge_weights,
)
```

### Native hypergraph FEM

`FemCoarsenSolver` imports the actual `fem` submodule package. Its default
`map_type='native'` optimizes the exact product-distribution expectation of
weighted km1. `method='pubo'` is a compatible alias for this native objective
for any `q >= 2`. Explicit FEM `map_type='clique'` or `'star'` options remain
surrogate experiments; star auxiliary nodes carry zero resource weight.
Every returned candidate is scored on the original weighted km1 objective.

The FEM core now optimizes categorical logits using free-energy gradients.
It supports Adam, SGD and RMSprop, categorical entropy for multiway problems,
and `lin`, `exp`, `inverse`/`inv` schedules. Both inverse aliases interpolate
beta harmonically from `betamin` to `betamax`. `FEM.solve()` returns discrete
configurations and their discrete objective values; final marginals and
free-energy history are available on `case.solver.probabilities` and
`case.solver.free_energy_history`. Manual callbacks must return `dE/dp` with
the same `(batch, nodes, q)` shape as the probabilities. The standalone FEM
package implements `qubo` and `customize`; legacy named problem types without
an implementation raise rather than silently treating every problem as QUBO.
`use_compile=True` currently warns and uses the verified eager path.

Balance is an **upper capacity** constraint:
`max(block_load) <= (1 + epsilon) * total_node_weight / q`.
The smooth penalty is scaled by the mean logical-node weight, so changing
resource units does not change it. It is not a discrete-feasibility guarantee:
each trial is rounded and independently checked using the physical weights.
Equal-weight bisections use a capacity sort. General weights use greedy
packing, followed by bounded exact packing for at most 20 nodes by default.
`exact_balance_max_nodes` and `balance_search_budget` control that fallback.
`BalanceInfeasibleError` indicates a certificate of infeasibility;
`BalanceSearchError` means the search failed or exhausted its budget, and does
not prove the instance infeasible. Rounding cannot restore a partition that
the coarse hierarchy has made impossible.

Native flow, FEM-IER and V-cycle preserve node and hyperedge weights at
every level, including an empty hierarchy. Legacy hybrid/MCTS/evolution refinement
currently rejects nonunit hyperedge weights explicitly; use
`mode_cycle=('flow',)` or opt-in `('fem_ier', 'flow')` for weighted edges. Pass original weights explicitly
when calling V-cycle if level metadata cannot identify them unambiguously.

Repair regressions can be run from the project root:

```bash
python -m pytest -q tests/test_hyper_fem_native.py tests/test_hyper_weighted_refine.py
```

Run the core tests separately from `lib/qubo-solver`:

```bash
python -m pytest -q tests/test_fem_gradients.py tests/test_adaptive_annealing.py
```

The [repair validation report](benchmarks/hypergraph/results/fem-repair-20260924/REPORT.md)
records 93 passing tests, 40 saved instances, four weighted pipelines, and
IBM01/IBM02 runs through the actual production API. It includes reproduction
commands, input/source hashes, timings, and the limits of these comparisons.

The [complete IEP case report](benchmarks/hypergraph/results/iep-complete-20260924/REPORT.md)
independently re-evaluates every saved assignment in the current IEP suite,
checks exact optima where enumeration is possible, and adds a ten-seed
`bad_for_ec` coarsening-pathology run. Reproduce the audit with:

```bash
python benchmarks/hypergraph/validate_iep_cases.py \
  --output benchmarks/hypergraph/results/iep-complete-rerun
```

The [search follow-up](benchmarks/hypergraph/results/ogp-search-followup-20260924/REPORT.md)
adds paired IBM runs across ten seeds, greedy restarts within each FEM run's
measured time budget, and exact small-instance reachability/energy-barrier
checks. The standalone `benchmarks/hypergraph/native_mcmc.py` implements native
weighted-km1 Metropolis moves, swaps, and symmetric block proposals. Fixed-beta
detailed balance is tested; annealing trajectories are optimization records.
These research baselines preserve the production pipeline's current defaults.

### Native FEM cyclic refinement (IER)

An opt-in IER stage uses the same FEM core to select combinations of balanced
move atoms. Each atom exchanges equal total resource between two blocks;
atoms have disjoint vertices, so any selected combination preserves capacity.
The numeric checks also bound accumulated positive load residuals across atoms.
Pairs and 2-for-1 exchanges are supported. The native km1 expectation treats
all pins in one atom as correlated and retains interactions between atoms;
it does not truncate the selector objective to a QUBO. The FEM variational
distribution still factorizes across atom selectors: nodes within an atom
share one selector, while different selectors are independent Bernoulli variables.
IER generates candidates, solves their joint selection, accepts a result and
refreshes candidates; FEM, random and exact are interchangeable selection backends.

```python
refine_solver.update_params(
    mode_cycle=('fem_ier', 'flow'), max_imbalance=0.03, seed=30,
    ier_rounds=2, ier_max_moves=24, ier_boundary_pool=96,
    ier_num_trials=8, ier_num_steps=100, ier_pool_strategy='local',
)
```

This refiner can be passed directly to `vcycle_uncoarsen`. The local candidate
pool grows from a cut net; `'global'` selects from the full boundary instead.
`ier_boundary_pool` limits total pooled vertices, and `ier_max_moves` limits
binary move variables. Every round regenerates candidates at its current
partition and accepts only a strict native-cut improvement after independent
capacity checks. The starting partition must already be feasible.

All-off, all-on and individual-atom candidates are shared with the diagnostic
`ier_backend='random'` control; its default sample count is
`ier_num_trials * ier_num_steps`. Per-round `last_result['stages']` records
whether the backend beats these shared deterministic candidates. This alone
does not establish superiority over other combination searches. A bounded `'exact'` backend certifies
only the generated move subproblem. These controls have different runtimes.
The default refinement mode remains `('flow',)`.

Two additional production controls support complete-pipeline comparisons:
`ier_backend='deterministic'` uses only the shared candidates, while `'pairs'`
also evaluates every two-atom subset. Both use the same native scoring and
capacity acceptance as FEM. The deterministic control includes all-on and is
therefore broader than choosing a single atom. Selector counts and unique
counts are saved per round. See the
[equal-time protocol](benchmarks/hypergraph/IER_EQUAL_TIME_PROTOCOL.md) for the
fixed inputs, budgets, restart policy, fallback and completion rules.

The [FEM-IER validation report](benchmarks/hypergraph/results/fem-ier-20260924/REPORT.md)
records 138 passing regression tests, four exact weighted cases, ten fixed IBM
starts with random-selection and strict-time additional-FM controls, and two
full V-cycle integration checks. Final polishing improves five of ten starts;
three gains exceed the shared deterministic candidates. Offline pair enumeration
matches every saved IBM FEM round, so these results do not establish FEM's
quality advantage over sparse combination search. Per-level integration
improves both tested seed-30 outputs but adds runtime; it is not an equal-time
or KaHyPar comparison.

The [complete V-cycle equal-time report](benchmarks/hypergraph/results/fem-ier-equal-time-20260924/REPORT.md)
compares five paired starts on each IBM input at preset 5/10/20-second deadlines.
At 20 seconds, mean km1 for FEM/pairs/shared-candidates/flow is
775.6/757.8/750.6/805.6 on IBM01 and 1127.2/1126.4/1140.6/1170.2 on IBM02.
FEM wins/ties/loses 1/4/5 against pairs: these CPU settings do not establish
an equal-time FEM advantage. Coarsening and coarse initialization are common
given inputs outside this refinement budget. The report separates initial-only
fallbacks from completed V-cycles, retains excluded overshoots, and includes
171 passing regressions and an independent audit of all 2,466 IER rounds.

### Experimental quotient coarsening

`KahyparLikeSolver.coarsen()` also supports an opt-in boundary-aware matching
score. It estimates how often a candidate pair is separated by several cheap
greedy partitions, and discounts pairs with high disagreement. A balance cap
prevents a supervertex from becoming too heavy for one block. This mode uses
`use_lsh=False`, since LSH pre-groups vertices before the pair score runs:

```python
result = kahypar_solver.coarsen(
    hyperedges, num_nodes, q,
    score_mode='boundary', coarsen_to=200, seed=1,
    num_pilots=4, pilot_refine_passes=0,
    enforce_balance_cap=True, epsilon=0.03,
)
```

The default `score_mode='hem'` keeps the original heavy-edge score. Every
contraction preserves the weighted hypergraph connectivity objective for
partitions lifted from the coarse graph; it still restricts which partitions
can be represented. `coarse_hyperedge_weights` is returned alongside
`coarse_hyperedges`. Pass those weights explicitly to the coarse initializer;
native FEM and flow V-cycle now preserve them as well.

Run `python benchmarks/hypergraph/coarsen_quotient.py --instances 80` to
compare both scores on small synthetic hypergraphs. It solves each coarse
instance by exhaustive enumeration, isolating the effect of coarsening from
FEM/SBM. The output reports the exact original optimum, each method's
contraction loss, and a paired approximate 95% interval for the mean
boundary gain (HEM cut minus boundary cut). For example, use
`--instances 2000 --seed 4000 --planted-probability 0.85` for a larger run.
On that fixed synthetic sample, boundary scoring reduced the planted-family
mean cut from 1.885 to 1.799, but did not improve the random-family mean
cut (8.4885 versus 8.5085). More aggressive `boundary_weight=0.8` can
make both families worse. These results concern eight-vertex exact coarse
solutions, not end-to-end FEM/SBM or large real hypergraphs.

The balance cap alone is only a local necessary condition: each supervertex
fits within one block, but the set of supervertices may still be impossible
to pack into all blocks. For example, weights `[4, 4, 2, 2]` admit two blocks
of capacity 6, whereas merging the two weight-2 vertices gives `[4, 4, 4]`,
which does not. For small inputs, opt into an **exact global feasibility
guard** to reject such contractions:

```python
result = kahypar_solver.coarsen(
    hyperedges, num_nodes, q,
    coarsen_to=4, node_weights=vertex_weights,
    enforce_global_feasibility=True, epsilon=0.0,
)
```

The guard solves an exponential-time bin-packing decision problem after
candidate merges, so it defaults to at most 20 original vertices and raises
on larger inputs or `use_lsh=True`. It also rejects an input whose original
vertices are already infeasible under the requested balance cap. It may stop
coarsening above `coarsen_to` when no safe merge remains. The default remains
off. Run the weighted synthetic ablation with:

```bash
python benchmarks/hypergraph/coarsen_feasibility.py --instances 1000 --seed 4000
```

Locally capped HEM produced
422/1000 infeasible random-family coarse graphs, versus 0 with the guard;
the planted family was 439/1000 versus 0. Conditional mean cuts from only
the feasible local runs are selection-biased and should not be compared to
the guarded all-instance mean. This guard has not yet been evaluated on real
hypergraph instances or the complete FEM/V-Cycle pipeline.

### Equal-time checks on real hypergraphs

`benchmarks/hypergraph/time_budget_hgr.py` compares one boundary run against
HEM restarts that finish within the same measured wall-clock budget. Both
paths use the same greedy coarse initializer and two-pass FM V-Cycle. The
script also reports the best HEM result when one extra run is allowed to
cross the deadline, since whole-run timing can leave unused budget.

Two public, unweighted hMETIS cases are pinned under
`benchmarks/hypergraph/data/hmetis`: [ibm01.hgr](https://github.com/kahypar/mt-kahypar/blob/eee7b7a03dbbd565a39f6cb2679b083e484c905e/lib/examples/ibm01.hgr)
(12,752 vertices, 14,111 hyperedges) and
[ibm02.hgr](https://github.com/TILOS-AI-Institute/HypergraphPartitioning/blob/ff614601a9b8f7853e21019e1fd320f44e445f3c/benchmark/ISPD_benchmark/ibm02.hgr)
(19,601 vertices, 19,584 hyperedges). Pinned download URLs and SHA-256
checksums are in the benchmark script's docstring and the local data README.
For example:

```bash
python benchmarks/hypergraph/time_budget_hgr.py benchmarks/hypergraph/data/hmetis/ibm01.hgr --trials 8 --seed 30 --pilot-refine-passes 0
python benchmarks/hypergraph/time_budget_hgr.py benchmarks/hypergraph/data/hmetis/ibm01.hgr --trials 8 --seed 38 --pilot-refine-passes 0
```

The ibm01 rows combine those two eight-trial batches, with the indicated
pilot-refinement setting; ibm02 uses four trials starting at seed 30. The
following are exploratory means, not claims of general superiority.
Lower cut is better; `HEM best` is the best completed restart per trial:

| Case | Pilot refinement passes | Trials | Boundary mean cut | Equal-time HEM best mean cut | Boundary wins | Mean boundary time |
|------|-------------------------:|-------:|------------------:|-----------------------------:|--------------:|-------------------:|
| ibm01 | 0 | 16 | 1,149 | 1,270 | 13/16 | 3.47 s |
| ibm01 | 2 | 16 | 1,353 | 1,201 | 3/16 | 4.87 s |
| ibm02 | 0 | 4 | 2,937 | 2,635 | 0/4 | 7.29 s |
| ibm02 | 2 | 4 | 2,974 | 2,124 | 1/4 | 11.51 s |

On these cases, refining the pilot partitions added cost without a stable
cut benefit. The unrefined boundary score helped on ibm01 but hurt on ibm02;
it is not enabled by default. The exact global feasibility guard remains
limited to small instances and was not used for these real-hypergraph runs.
The pilots themselves *did* improve with refinement (for seeds 30–33, mean
pilot cut fell from 5,860 to 5,109 on ibm01 and from 10,448 to 8,164 on
ibm02). Better pilot cut therefore does not imply a better quotient: the
pilot is used to score safe contractions, not as the final partition.

## Acceleration

- SBM can use opt-in `torch.compile`. FEM currently uses eager autograd and
  warns if `use_compile=True` is requested.

```python
compile_fem = False   # FEM currently uses eager autograd
compile_sbm = True    # compile SBM bsb_torch_batch step function
```

## Latest Updates

- **Repo cleanup**: Solver code (FEM, SBM, QIS3, DIGCIM) extracted to external
  **[qubo-solver](https://github.com/yao-baijian/qubo-solver)** submodule.
  ``src/fem/``, ``src/sbm/``, ``src/qis3/``, ``src/digcim/`` removed.
  Import via ``from fem import FemSolver`` (automatically resolves via submodule).
- **Unified SB**: strategy pattern + GSB/GGSB/Quantization mixins.
- **Benchmark suite**: grid-search benchmark for all SB method combinations.
- **Backward compatible**: ``sys.path`` setup in ``src/__init__.py`` handles
  submodule discovery.

## Project Structure

```
src/
├── __init__.py          # Adds lib/qubo-solver/src to sys.path
├── hyper_solver.py      # Hypergraph: KahyparLikeSolver, FemCoarsenSolver,
│                        #   HyperRefineSolver, vcycle_uncoarsen
└── partition/           # Multi-level partitioning (coarsen, refine, hyper_utils)
    ├── coarsen.py, hyper_coarsen.py, hyper_refine.py
    ├── hyper_utils.py, kaffpa_multiway.py, refine.py, utils.py
    └── script/test_kahypar.py
lib/                     # Git submodules
└── qubo-solver/         # https://github.com/yao-baijian/qubo-solver
tests/                   # Test suite and benchmarks
├── test_hyper_bmincut_coarsen.py
├── test_hyper_bmincut.py
├── test_bmincut_coarsen.py
├── test_bmincut.py
├── test_bmincut_base.py
├── test_bmincut_gpu_boost.py
├── plot_results.py
└── utils.py
benchmarks/
├── bmincut/             # Balanced min-cut benchmarks
├── maxcut/              # Max-cut benchmarks (Gset, WK2000)
└── maxsat/              # Max-SAT benchmarks
doc/                    # Module documentation
config/                 # Working solver configs (copied from src/configs/)
build/                  # Benchmark result CSVs
```

## Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate fem

# 2. Install PyTorch (see https://pytorch.org/)
pip3 install torch torchvision torchaudio

# 3. Optional: external partition tools
pip install pymetis             # METIS wrapper
pip install kahypar             # KaHyPar
pip install kahip               # KaFFPa/KaHIP
```

## Configuration

Each solver has a default JSON config under `src/configs/`. At runtime these are copied to `config/` (gitignored) where you can override them:

```
config/
├── fem.json
├── sbm.json
├── kaffpa.json
├── metis.json
└── cyclic.json
```

Use `method_registry.ensure_configs()` to populate the working directory, or manually edit the JSON files in `config/`.

## Running Tests

Run from project root:

```powershell
python -u tests/test_hyper_bmincut_coarsen.py   # Hypergraph V-Cycle (best-of-N trials)
python -u tests/test_hyper_bmincut.py           # Hypergraph bmincut tests
python -u tests/test_bmincut_coarsen.py         # Multi-level coarsening benchmarks
python -u tests/test_bmincut.py                 # Graph bmincut tests
python -u tests/test_bmincut_gpu_boost.py       # GPU acceleration tests
python -u tests/plot_results.py                 # Plot results (5 plot types)
```

### Hypergraph V-Cycle Test

`test_hyper_bmincut_coarsen.py` runs a configurable benchmark:
- Coarsens once with HEM matching
- Runs `num_runs` outer trials (different seeds) for greedy and FEM initial partitions
- Each trial performs a full V-Cycle uncoarsening with refinement at every hierarchy level
- Reports the best result in pipe-delimited table format:
  ```
  |powersim|4|88|0.047859578229574|140|0.046344235383255|15|
  ```

### Select Methods

Set `partition_methods` in any test file to pick which pipeline to run (see [Normal Graph Pipeline](#normal-graph-pipeline) table above).

## Documentation

See `doc/` for detailed module documentation:
- `doc/fem.md` — FEM solver details
- `doc/partition.md` — Partition pipeline details
- `doc/sbm.md` — Simulated Bifurcation details
- `doc/qis3.md` — Quantum-Inspired Solver v3 details

## References

- FEM framework: mean-field entropy minimization with annealing
- Simulated Bifurcation: Goto et al., Science Advances (2019)
- Cyclic Expansion: arXiv 2312.15467v1
- KaHyPar: Schlag et al., SEA (2016)
- KaFFPa/KaHIP: Sanders & Schulz, ALENEX (2011)
- KaHIP — Karlsruhe High Quality Partitioning (kahip.github.io)
