# Retained CPU experiment — measured, not projected

## Execution

The existing v0.4 standalone `rad-kbench-replay` was built against the actual
supplied Radiance registry, host device backend, encoding and kBench-comparison
sources. `libref.so` was built from genuine supplied-source code. The three other
candidates, `cpu_panel1`, `cpu_panel4` and `cpu_panel16`, are newly written CPU-only
GEMM implementations with fixed column panels. They are not GPU kernels, not
emulations, and not represented as preexisting Radiance optimizations.

The full engine / monolithic rad-kbench binary was not built. The supplied tree
has a missing `core/kvstream/bind.h`; the standalone host target does not use that
engine module and does not fake its implementation.

| Property | Actual retained run |
|---|---|
| Host | AMD EPYC 9V74 80-Core Processor, authoring container |
| Compiler | GNU C++ 14.2.0 |
| Candidate count | 4 |
| Distinct mathematical shapes | 40 |
| Checked replay executions | 160 |
| Warmups / timed repetitions per execution | 2 / 15 |
| Timed samples retained | 2400 |
| Shapes: M | 1, 2, 5, 8, 16, 31, 32, 33, 63, 64 |
| Shapes: N | 16, 32, 64, 96 |
| Shapes: K | Cycled through 32, 64, 96, 128 |
| Group split | 20 training, 10 calibration, 10 test |
| Scheduling | Deterministically randomized shape and candidate order, seed 7 |
| Correctness | Independent Python `math.fsum` oracle, all warmups and timed runs checked |
| Memory diagnostics | Three operation-contract passes before timing |
| Instrumentation in timing | No ASan; no per-operation diagnostic checkpoints in timing |
| Threading | Single-threaded candidate kernels, libref built without OpenMP |
| Host power/frequency/noise control | Not controlled; affinity mask recorded |
| AMD GPU execution | None |

Restored inputs precede measured launcher windows. The cache label is warm CPU
work; it is not a measured GPU hit-rate. Time is host ordered launcher-window
`steady_clock` time, not isolated CPU instruction latency.

## Held-out result

All forty candidate measurements from the ten held-out shapes were unused by
training and calibration. Different repetitions of one shape were never split
across groups.

| Measure | Operations-only baseline | Feature-based calibrated surrogate |
|---|---:|---:|
| Median relative latency error | 8.10% | 5.60% |
| 90th percentile relative latency error | Not recorded for this baseline | 25.51% |
| Worst selection regret | 81.78% | 0.00% |
| Top-1 choice on held-out shapes | Not used as the baseline headline | 10 / 10 |

Selection regret is actual chosen latency divided by the measured fastest latency,
minus one. A zero observed regret is not a claim of a globally optimal algorithm.
The operations-only baseline uses one fitted time-per-useful-operation coefficient
per candidate, learned on the same training groups; it is not a manufacturer peak.

The residual envelope covered all four candidates simultaneously on **9/10** test
shapes. Only **2/10** test decisions had separated envelopes; both selected the
measured winner. The planner requests measurements for the other eight despite the
retrospective top-1 result. The median timing error is not a confidence guarantee;
the 25.51% tail error and limited independent test set remain visible.

These results are evidence for the implementation and this particular CPU workload
family, not a guarantee of repeatable CPU superiority, R9700 timing accuracy,
Qwen throughput, or prediction for arbitrarily modified kernels. Shared-host noise
was not controlled. The dataset was not selected from repeated experiments for the
best metrics; the original raw run and failures policy are retained.

## Inspect or reproduce

`evidence/cpu-comparison/` contains the original reports, portable input bytes,
expected outputs, run index, all samples, fitted coefficients, group assignments,
held-out predictions, scenario/measurement decisions and environment identities.
Original executable paths are historical; binaries are not bundled.

Run `check-planner-cpu.sh RADIANCE_SOURCE NEW_DIRECTORY` to build on another host
and produce an independently bound experiment. Results need not match these
numbers. Fitting the retained `measurements.json` with its catalog/profile and
seed reproduces the stored calibration exactly; this was checked after the final
validation hardening. It does not repeat the hardware timings.
