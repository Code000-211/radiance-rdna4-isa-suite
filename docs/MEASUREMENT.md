# Measurement semantics and production fidelity

## What is actually repeated

Each iteration restores all declared initial allocations, reconstructs already
bound pointer relationships, resets scratch, runs the prefix and executes the
measured operation window. Expected bytes and input data are identical for checks
and timing. Correctness, guards and device-to-host checks are outside the measured
interval. Snapshot restoration changes cache contents; a representative prefix
can improve context, but this release does not reconstruct an exact production
cache state, NUMA state, eviction sequence or virtual-address placement.

State evolves normally **within** the operation window. Across independent
iterations it resets. A multi-step decode/verify trajectory can be described as
one longer fixture if all state remains explicit and supported. Hidden global
state and mutable pointer tables are not supported.

## Timing modes

* **Host:** steady-clock duration of the measured launcher sequence, excluding
  snapshot restore, prefix and verification. This is not a GPU estimate.
* **Single-stream gated AQL:** submit measured work behind the existing Radiance
  queue gate, then open it and wait for the stop event. Only implemented when the
  requested backend really is AQL and there is one stream. Gate/queue failures
  produce INCOMPLETE; there is no fallback that keeps the same measurement label.
* **Host-paced AQL:** prefix queues join at a common start event; measured streams
  fan out and join at the stop event. Event time includes genuine device waits,
  inter-queue dependencies and host submission gaps. The join/fanout instrumentation
  is extra work compared with an engine's pre-recorded tape.

The submission plan is pre-resolved and typed. JSON interpretation, plugin/row
resolution, allocation and disk logging do not occur inside the measured submit
loop. Existing launchers can themselves do CPU work; their behavior has not been
replaced by a custom HIP or fake AQL path.

**These are window durations, not per-dispatch latencies.** A launcher can issue
multiple actual kernels. A `--reps` sample covers the whole window. Median and
p90 are computed from those window samples, not averages mislabeled as individual
kernel tails. Summing timings across queues is not used as a makespan estimate.
No tokens/s are inferred. No CPU-only run is presented as R9700 performance.

## Profiling and comparisons

No hardware counter profiler is included or activated. Do not mix counter-enabled
serialized runs with uninstrumented overlap measurements without labeling them.
The runner records its hash, actual plugin/fixture identities, compiler/ABI,
backend/device information and best-effort raw sysfs readings. Missing telemetry
stays missing; a binary `gpu_metrics` file is not decoded using guessed offsets.

Candidate rebinding preserves inputs and oracles and forces new static evidence
for changed GPU binaries. This release does not automatically interleave A/B
runs or infer regression confidence from a few samples. Use matched devices,
clocks, power profiles, temperatures, driver/runtime versions and sufficiently
many samples. Keep independent engine-level calibrations for the operation
windows used to approximate your workloads.

## Fault isolation

Each fixture runs in a separate child process with an external wall-clock timeout.
A timeout kills that process group and is reported INCOMPLETE, never a valid
latency sample. The process can be gone while the GPU is still unhealthy; this
is not a reset mechanism or wedge-prevention guarantee. Stop the series and inspect
host kernel logs/device health after a fault. No automatic retries or resets occur.

Progress is logged at iteration/prefix/measurement/check boundaries. It does not
add a disk write to each timed dispatch or claim to identify the exact device
instruction that hung. The saved typed plan and stdout/stderr support subsequent
narrowing of a failing window.

## v0.4 diagnostic separation and paired measurements

Contract diagnostics are extra serial passes before timing, not timings of the
optimized production stream. Instrumented host ASan and `--trace-dispatches` are
diagnostic-only; they cannot supply benchmark samples. Normal iterations restore
the initial window state, preserve state across its nodes, and check final outputs
and guards on every repetition.

The `ab` command uses ABBA blocks with separately supervised workers. Inputs,
references/tolerances, target, rows and frozen tuning must match; explicit candidate
binary identities may differ. Each run checks its outputs. First failure or missing
measurement stops the experiment; all attempts remain. Summary ratios use paired
block means of per-run medians, with an exploratory whole-block bootstrap. These
are not individual-dispatch percentile estimates, nor proof of clock/thermal
stability. Environment values are recorded, not remotely attested.

No engine-window calibration was supplied or created in this release. Similar
standalone timing alone cannot establish identical cache, overlap or P2P behavior.
The recorded A/A demonstration is a functional control, not performance evidence.
