# Radiance kernel snapshot replay

**Status: RUNTIME_CHECKS_PASS**

This report covers this fixture only. It is not full ISA, memory-safety, or GPU-wedge certification.

## Input and execution evidence

```json
{
  "schema": "radiance-kernel-replay/1",
  "status": "FIXTURE_VALID",
  "manifest_sha256": "1dfd926242c4f9285ad40ca954cb0dac4a8d5dd56989c46388ac3548c84fbdbd",
  "id": "cpu-M32-N64-K32-panel16",
  "allocation_bytes_including_scratch": 20480,
  "blob_bytes": 28672,
  "kernel_nodes": [
    "gemm"
  ],
  "node_count": 1,
  "stream_count": 1,
  "kernel_rows": [
    "planner_cpu:cpu_panel16"
  ],
  "provenance": {
    "kind": "synthetic",
    "reference": "independent",
    "note": "CPU-only planner comparison; independent fsum oracle; not production activation snapshot."
  },
  "runtime_executed": false,
  "all_radiance_kernels_covered": false
}
```

## Runtime

```json
{
  "backend": "host",
  "checked_iterations": 17,
  "checks_per_iteration": 1,
  "device": {
    "arch": "host",
    "dma_lane": "",
    "n_cu": 5,
    "name": "AMD EPYC 9V74 80-Core Processor",
    "ordinal": 0,
    "pci_address": "",
    "runtime_version": "",
    "vram_bytes": 3118428160
  },
  "diagnostics": {
    "allocation_byte_bounds_not_scalar_element_bounds": true,
    "checked_nodes": 1,
    "completed_passes": 3,
    "explicit_checkpoints": 0,
    "intermediate_equality": "explicit checkpoint tolerances, then exact bytes for remaining writes against same-candidate baseline",
    "output_poison_scope": "packed exact-span output-only operands, excluding all readable aliases; other views not covered",
    "output_poison_variants": 2,
    "per_operation_write_contracts": true,
    "poisoned_output_bytes_total": 16384,
    "race_freedom_proved": false,
    "read_checks": "NOT_CHECKED",
    "scratch_contracts": true,
    "status": "PASS"
  },
  "dispatch_plan": "typed pre-resolved launcher plan; no JSON/disk I/O in measured submit loop",
  "fixture_sha256": "1dfd926242c4f9285ad40ca954cb0dac4a8d5dd56989c46388ac3548c84fbdbd",
  "measurement": "host ordered launcher-window steady_clock time",
  "physical_bytes_with_guards": 22784,
  "reference": "independent",
  "restore_policy": "restore every allocation and zero scratch before every checked iteration",
  "samples_us": [
    16.545,
    16.475,
    16.956,
    16.505,
    20.1,
    16.495,
    16.505,
    16.465,
    16.565,
    3053.231,
    18.148,
    16.525,
    16.566,
    16.545,
    16.475
  ],
  "schema": "radiance-replay-worker-result/1",
  "state_policy": "no hidden state; warmups and timed iterations use identical pristine bytes",
  "statistics": {
    "median": 16.545,
    "min": 16.465,
    "n": 15,
    "p90": 20.1,
    "p90_is_per_dispatch": false,
    "tokens_per_second_estimated": false,
    "unit": "us per complete window"
  },
  "status": "RUNTIME_CHECKS_PASS",
  "worker_build": {
    "compiler": "14.2.0",
    "cxx": 202002,
    "host_asan": false,
    "radiance_abi": 30
  }
}
```

## Limits

- Ordered launcher-level replay; not arbitrary raw AQL packets or device-side chains.
- Single device/rank only. Captured baseline is a regression oracle, not independent correctness.
- Guard bytes alone do not detect reads. Host ASan checks only instrumented host paths.
- Diagnostic checkpoints serialize operations; they do not prove race freedom or production timing.
- Output variation excludes readable aliases, padded/strided views and custom encodings.
- Same-operation writes restored before a checkpoint can still be missed.
- Queue-window time is not per-dispatch latency and is never converted to tokens/s.
