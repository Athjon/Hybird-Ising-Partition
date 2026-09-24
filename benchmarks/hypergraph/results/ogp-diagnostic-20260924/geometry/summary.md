# Exact finite-size native hypergraph overlap diagnostic

Synthetic control ensembles; these are not the manuscript benchmark instances. This experiment does not establish or refute asymptotic OGP.

Configuration: n=[8, 12, 16]; 10 instances per family and size; 2n independently drawn four-pin unit-weight hyperedges; planted within-block probability 0.75; seed base 20260924.

All nodes have unit weight. Each block contains exactly n/2 vertices. Native cut-net equals km1 for this binary setting. The complete feasible set is enumerated with bit 0 fixed to zero to remove global label flips. q = |n - 2d| / n. Every unordered pair of different equivalence classes is counted. Two gap diagnostics are reported: one conservatively requires both boundaries to come from different-class pairs, while the other also admits self-pairs q=1 as in the standard all-pairs definition. Singleton sets are excluded from nontrivial including-self gap evidence.

| Family | n | Absolute epsilon | Mean optimum | Mean near-optimal classes | Singleton sets | Missing any feasible q | Distinct-bounded gaps | Including-self gaps (nontrivial) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 8 | 0 | 13.70 | 2.20 | 3/10 | 6/10 | 0/10 | 2/10 |
| random | 8 | 1 | 13.70 | 11.30 | 1/10 | 1/10 | 0/10 | 0/10 |
| random | 8 | 2 | 13.70 | 29.40 | 0/10 | 0/10 | 0/10 | 0/10 |
| planted | 8 | 0 | 4.40 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| planted | 8 | 1 | 4.40 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| planted | 8 | 2 | 4.40 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| random | 12 | 0 | 18.50 | 3.50 | 1/10 | 5/10 | 0/10 | 1/10 |
| random | 12 | 1 | 18.50 | 16.50 | 0/10 | 0/10 | 0/10 | 0/10 |
| random | 12 | 2 | 18.50 | 57.30 | 0/10 | 0/10 | 0/10 | 0/10 |
| planted | 12 | 0 | 6.40 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| planted | 12 | 1 | 6.40 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| planted | 12 | 2 | 6.40 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| random | 16 | 0 | 22.90 | 4.50 | 4/10 | 8/10 | 0/10 | 0/10 |
| random | 16 | 1 | 22.90 | 31.10 | 0/10 | 4/10 | 0/10 | 0/10 |
| random | 16 | 2 | 22.90 | 128.50 | 0/10 | 1/10 | 0/10 | 0/10 |
| planted | 16 | 0 | 7.80 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| planted | 16 | 1 | 7.80 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |
| planted | 16 | 2 | 7.80 | 1.00 | 10/10 | 10/10 | 0/10 | 0/10 |

“Missing any feasible q” includes empty or narrow support and is not evidence of separated clusters. “Distinct-bounded gaps” requires two actually attained distinct-pair overlap values bounding an absent full-feasible lattice point. This stricter condition is NOT necessary for OGP. “Including-self gaps” also permits q=1 as the upper boundary and requires at least two solution classes with a q<1 pair; a singleton is not counted.

Across 180 instance/threshold combinations, distinct-bounded gaps: 0; nontrivial including-self gaps: 3; gaps with q=1 self-overlap upper boundary: 3. A self-bounded gap may simply reflect a sparse low-energy window, and does not establish an algorithmic barrier.

Processed 60 instances and 237,596 near-optimal distinct pairs (summed over thresholds, so nested sets repeat pairs). All are exhaustive.

Validation compares exact native costs with an independent scalar evaluator, and complete pair counts for a real n=8 instance with explicit spin-dot-product enumeration. A separate hand-built balanced solution-set fixture checks a distinct-bounded gap with witnesses on both sides, and a two-class fixture verifies a gap bounded above by q=1. These fixtures are not included in ensemble results.

Files: `instances/*.json` contain hyperedges, all feasible masks and energies, threshold supports/counts and gap witnesses; `instances.csv` has threshold rows; `feasible_baselines.json` has complete feasible-pair supports; `validation.json`, `summary.json`, and `manifest.json` record checks, aggregation and provenance.

Limitations:

- Synthetic ensembles are not manuscript benchmark instances.
- n<=16 and three absolute thresholds cannot establish asymptotic OGP.
- A finite-size gap does not imply an algorithmic lower bound.
- The distinct-bounded diagnostic is stricter than the standard all-pairs overlap-gap definition; its absence is not absence of OGP.
- An including-self gap can reflect only a sparse low-energy window; it does not establish an algorithmic barrier.
- Singleton near-optimal sets are excluded from nontrivial gap evidence.
- Neither finite-size diagnostic establishes or excludes ensemble OGP at other sizes or thresholds.
- No HIP, FEM, SBM, MCMC or KaHyPar performance is evaluated here.
- Planted n=8 has only one internal 4-subset per block; retained repeated nets make this especially simple.
- Fixed absolute epsilon changes relative accuracy as n and m change.

Reproduce from repository root:

```sh
python benchmarks/hypergraph/ogp_exact_geometry.py
```
