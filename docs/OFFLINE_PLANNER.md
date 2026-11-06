# Offline planner: implementation and mathematical boundary

## What is computed, and what must be measured

`radplan` computes workload features and ranks an explicit finite set of candidates.
It does not generate HIP/ISA, emulate a GPU on a CPU, infer arbitrary code behavior,
change production dispatch, or certify correctness. Dense transposed-B GEMM is the
current canonical operation (`gemm_nt`); the W4A8 launcher may be `gemm_nt_q`.

The model in this implementation is narrower than a general multi-phase simulator.
It has a main GEMM resource balance, optional split-K reduction, and explicit serial
stage composition. Whole-CFG dependency chains, cache simulation, bank conflicts,
barrier latency, arbitrary fusion, expert scheduling and P2P are not inferred.
These are reasons to calibrate and retain uncertainty, not evidence of zero cost.

## Work and tile efficiency

For M x K times N x K, useful work is `2*M*N*K` multiply/add operations. For a
candidate declaring full-tile work:

```
gM = ceil(M/BM)
gN = ceil(N/BN)
gK = ceil((K/splitK)/BK)
executed_ops = 2 * (gM*BM) * (gN*BN) * (gK*BK*splitK)
tile_efficiency = useful_ops / executed_ops
groups = gM*gN*splitK
```

An exact-tail candidate declares `work_policy=exact` and uses useful operation
count instead. This is a contract, not a proof from arbitrary binary code. Since
padding is already included in executed work it is NOT charged a second time.

Shapes must satisfy explicit launcher multiples, tail constraints, quantization
group boundaries, numerical/layout contracts, and workgroup/LDS limits. Unknown
known-issue tags are rejected. The reviewed source adapter excludes scalar-scale
M tails because its source expression reads the scale before the final store guard.
The planner does not claim that an emitted GPU read or wedge was reproduced.

## Memory and quantization

Let bytes(A), bytes(B), bytes(C) include packed element storage. Weight scales are
`N*ceil(K/group)*weight_scale_bytes`; activation row scales are
`M*activation_scale_bytes`. An unpacked LDS representation is not assumed equal to
packed VRAM storage. Candidate-specific static LDS and per-K-block LDS traffic are
separate fields.

The implementation distinguishes:

- Unique operand bytes: input/scale/output logical footprint.
- Tile-request bytes: a declared per-workgroup load-reuse model, not measured DRAM.
- Scenario bytes: unique main-phase bytes plus a user-declared fraction of repeated
  tile requests. Cache reuse is an explicit assumption, not inferred cache hits.
- Split-K partial stores and subsequent reduction reads/final output: counted once
  in their respective phases. A prior implementation double-count risk was fixed.

Warm or unknown cache state does not get a universal DRAM lower bound. Even cold
figures are conditional on declared work/traffic and an actual cold start. Scratch
allocation size is recorded but is **not** converted into invented scratch traffic.

## R9700 scenario equation

All times are microseconds. GOPS and GB/s use decimal SI units.

```
Tcompute = executed_ops / (peak_GOPS * 1000)
Tmemory  = scenario_bytes / (bandwidth_GBs * 1000)
Tunpack  = declared_unpack_ops / (assumed_ALU_GOPS * 1000)
TLDS     = declared_LDS_bytes / (assumed_LDS_GBs * 1000)
```

The supplied example applies explicitly illustrative compute/memory efficiencies,
a simple grid-size correction, and a register-capacity latency-hiding assumption.
If matrix and unpack operations are declared to share issue resources their costs
are summed; otherwise their maximum is used. Then:

```
Tscenario = assumed_launch_us
          + max(matrix_and_unpack_cost, memory_cost, LDS_cost)
          + optional_split_reduction_cost
```

These efficiency factors, launch cost, ALU/LDS rates and reuse fractions are NOT
measured R9700 values. They are exposed in `r9700-unmeasured.json`. No confidence
interval or automatic performance selection is derived from them. The manufacturer
compute/VRAM peaks and assumed efficiencies are kept in different profile sections.

For cold AQL work, the report also gives a conditional ideal resource floor:
`max(executed_ops/peak, unique_main_bytes/peak_bandwidth)`. It omits other costs and
is not a predicted runtime. A calibrated prediction below that floor is flagged,
not silently trusted. The warm-state floor is null.

The reviewed W4A8 prefill actually executes INT8 WMMA after INT4 unpacking. Thus its
profile path is `wmma_i8`, not `wmma_fp8` and not `wmma_i4`. A different source/kernel
requires a different reviewed compute-path description.

## Register resources

For static gfx1201 allocation, use the maximum of logical VGPR requirement and
encoded descriptor requirement, then round to the physical granule:

```
Wave32: allocated = ceil(R/24)*24; slots <= min(16, floor(1536/allocated))
Wave64: allocated = ceil(R/12)*12; slots <= min(16, floor( 768/allocated))
```

Logical VGPR values and descriptor encoding are not the same quantity. Missing
compiler resources stay unknown. The optional compiled import independently
cross-checks actual ELF metadata and descriptors against the existing strict ISA
report. Estimated resources retain their weaker provenance.

`ceil(BM*BN/(wave_size*waves_per_group))` is only a simple FP32 accumulator count
estimate. It is not the total compiler register requirement. An optional loose
resident-workgroup bound accounts for registers and LDS across a WGP; placement,
fragmentation, CU/WGP mode details and achieved occupancy are NOT established.
Higher occupancy is not equated to proportional speedup.

## Empirical latency surrogate

The calibration path uses a nonnegative linear surrogate, **separately per exact
candidate identity**:

```
T_hat = sum(beta_j * feature_j), beta_j >= 0
features = [1, executed_Gops, logical_operand_MB, tile_requested_MB,
            millions_of_group_K_blocks, millions_of_output_elements]
```

Coefficients are fitted with normalized cyclic coordinate descent to relative
squared error. They are not physical bandwidth, latency or instruction-throughput
measurements. The model is a CPU OR GPU latency interpolator; it is not a way of
scaling CPU microseconds into GPU microseconds.

All candidates for a whole shape/format/cache group remain together. Repetitions
are retained and summarized by their median; they are never scattered across
training and test splits. The candidate panel must be complete, with at least eight
training shape groups per candidate. The demo uses 20 train + 10 calibration + 10
test groups. The held-out test is not reused to tune coefficients or error radius.

The calibration score for shape group g is:

```
score_g = max_over_candidates(abs(log(measured_time/predicted_time)))
k = ceil((number_of_calibration_groups + 1)*(1-alpha))
q = kth ordered calibration score, or unbounded when k is too large
interval(candidate) = prediction * [exp(-q), exp(q)]
```

This has split-conformal form for candidate-simultaneous group coverage **under an
exchangeability assumption**. A deterministic benchmark grid on a noisy shared
host does not establish that assumption. The label is an empirical residual
envelope, not a certified production-confidence interval. Fit convergence,
held-out errors, top-k selection, regret and envelope coverage remain in the JSON,
including poor results. No fallback replaces disappointing measurements.

New/changed binaries, candidates, resource mappings, profiles, backend, device,
worker or measurement mode cannot silently reuse the old fit. Outside the training
M/N/K bounding box or contract/cache-state set, the point estimate is diagnostic
only and has no accepted interval. Being inside the box itself is not an accuracy
proof; a different access-pattern distribution can still invalidate a fit.

## Selection and requested measurements

Candidates with invalid geometry/contracts are listed as rejected with reasons.
When **every valid competitor** has an eligible calibrated envelope, candidates
are ranked by upper bound. Selection is separated only when the winner's upper
bound is strictly below every rival's lower bound. Otherwise the measurement queue
contains candidates with overlapping potential wins.

If even one candidate is uncalibrated, changed or out of domain, all candidates
are ranked by analytic scenarios instead. Measured-fit costs are not mixed with
uncalibrated costs. These rankings request measurements; they are not validated
winners. A one-candidate panel is not treated as proof of optimality.

A plan never marks a runtime/ISA test passed and never changes a production
launcher. Runtime/numerical tests remain necessary for changed code even if its
speed can be estimated from other evidence.

## Serial path composition

`path` performs dynamic programming over user-declared serial stages and explicit
layout-transition costs. It can prefer a slightly slower kernel when the alternative
requires a costly conversion. Missing conversion costs mean no valid path, not zero
cost. No concurrent scheduling, memory-lifetime proof, tensor-shape connectivity,
TP2 communication or fusion generation is implied. Stage/data compatibility must
already have been reviewed. Sum of upper envelopes is a scenario bound, not a joint
statistical guarantee for a whole model. Output is `SERIAL_PATH_SCENARIO`, never a
claim of production tokens/s.
