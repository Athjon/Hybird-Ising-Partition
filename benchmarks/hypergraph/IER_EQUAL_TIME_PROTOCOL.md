# Native IER equal-time protocol

Fixed before measured runs on 2026-09-24. Primary endpoint: native weighted
km1 available at **20 seconds**; secondary checkpoints: **5 and 10 seconds**.
All three checkpoints come from one execution of each arm, not independent runs.

## Question and inputs

Does native FEM joint selection improve the time–quality tradeoff of a complete
V-cycle, compared with cheaper selection methods that can run more cycles?

- IBM01 and IBM02, seeds 30–34, q=4, capacity epsilon=0.03.
- Reconstruct one HEM hierarchy shared by all arms for an input/seed. Verify
  that its final coarse mapping and node weights match the saved baseline;
  this check does not certify equality with every historical intermediate level.
- Reuse the corresponding saved coarse FEM assignment from
  `results/fem-multiseed-20260924-v2` as a common input for all arms.
- The shared hierarchy/coarse initialization is given input. Its construction
  is excluded from these refinement budgets and cannot be called complete HIP
  end-to-end latency.

## Arms

1. **FEM + flow:** two IER rounds at each refinement call, followed by two
   FM-style flow passes. FEM: 8 trials × 100 steps.
2. **Pairs + flow:** same pipeline, with all two-atom subsets instead of FEM
   selection.
3. **Deterministic candidates + flow:** same pipeline, using only all-off,
   all-on and each individual atom. This is not restricted to single moves,
   because all-on is also available.
4. **Flow reference:** a flow-only V-cycle, then additional flow calls on its
   own final graph until the assignment is unchanged or the budget is exhausted.
   This path is deterministic on the feasible inputs used here, so a fixed
   point permits early stopping. It is not a random-restart baseline.

All IER arms use the same local pool policy, at most 96 pooled vertices and
24 disjoint balanced atoms; they share all-off/all-on/single candidates.
Each new full V-cycle starts from the same given coarse assignment, with
candidate/FEM seed `input_seed + 1000003 * attempt_index`. This outer stride
differs from the inner IER round stride of 104729, avoiding adjacent-restart
reuse of the same round seed. Different accepted
states can produce different later candidate pools despite paired seeds.
No arm may borrow another arm's results. Tied selections may also produce
different future trajectories.

## Timing and deadlines

- CPU single thread, with BLAS/OMP/MKL thread limits set to 1. No concurrent
  benchmark or other agent CPU workload. Warm up all paths on a separate small
  instance before starting measured runs.
- Rotate arm execution order by instance/seed; keep order fixed before results
  are observed. Run each arm once up to the maximum budget.
- At the start of each arm, lift the common coarse assignment, score it and
  check capacity inside the timer. Record when this fallback becomes available.
- Count configuration, copying, projection, candidate generation, optimization,
  native scoring, capacity acceptance, incumbent selection and runtime
  diagnostics. Candidate availability is timestamped only after the work
  needed to produce and select the output completes.
- External independent duplicate verification and artifact I/O occur after
  the entire timed arm, uniformly for every method.
- A final indivisible call may finish beyond 20 seconds. Save it as an
  overshoot; it is ineligible at all earlier checkpoints. Do not use its
  partial progress as a completed output.
- At each checkpoint report the number of complete V-cycles, best eligible
  complete result and best available result including the common fallback.
  If no complete V-cycle has finished, label that explicitly as `no_result`
  / `initial_only`; never substitute an externally computed flow result.

## Evidence and interpretation

Save the hierarchy, original/coarse inputs, per-stage initial/final assignments,
move pools, selector scores, all attempts, timing events, statuses, source/input
hashes and environment. Verify original final cuts and weighted capacities
independently. Preserve failed attempts and overshoots.

Report each input separately, all five paired cases, checkpoint completion
counts and means of the explicitly defined best-available objective. Compare
methods on the same input/seed/budget. Do not hide missing completions in a
conditional mean or treat checkpoint measurements as independent samples.

This tests the complete refinement strategies on two fixed inputs. It does
not by itself establish FEM-specific causal contributions, KaHyPar superiority,
SOTA, asymptotic scaling or OGP barriers/breakthroughs. Do not retune parameters
after looking at results; follow-ups belong to separately identified runs.
