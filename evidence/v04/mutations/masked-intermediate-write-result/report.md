# Radiance kernel snapshot replay

**Status: FAIL**

This report covers this fixture only. It is not full ISA, memory-safety, or GPU-wedge certification.

## Input and execution evidence

```json
{
  "schema": "radiance-kernel-replay/1",
  "status": "FIXTURE_VALID",
  "manifest_sha256": "34afac3bda864c0fa87cdf7569779f0d0f92c30deb283013c5f75e7901fa7fc3",
  "id": "masked-intermediate-write",
  "allocation_bytes_including_scratch": 12,
  "blob_bytes": 20,
  "kernel_nodes": [
    "tested",
    "later"
  ],
  "node_count": 2,
  "stream_count": 1,
  "kernel_rows": [
    "replay_test:copy",
    "replay_test:mutate"
  ],
  "provenance": {
    "kind": "synthetic",
    "reference": "independent",
    "note": "Deliberately faulty HOST mutation/control; never GPU code."
  },
  "runtime_executed": false,
  "all_radiance_kernels_covered": false
}
```

## Runtime

```json
{
  "backend": "host",
  "checked_iterations": 0,
  "checks_per_iteration": 2,
  "device": {
    "arch": "host",
    "dma_lane": "",
    "n_cu": 5,
    "name": "AMD EPYC 9V74 80-Core Processor",
    "ordinal": 0,
    "pci_address": "",
    "runtime_version": "",
    "vram_bytes": 3124326400
  },
  "diagnostics": {
    "allocation_byte_bounds_not_scalar_element_bounds": true,
    "intermediate_equality": "explicit checkpoint tolerances, then exact bytes for remaining writes against same-candidate baseline",
    "output_poison_variants": 0,
    "per_operation_write_contracts": false,
    "race_freedom_proved": false,
    "read_checks": "NOT_CHECKED",
    "scratch_contracts": false,
    "status": "RUNNING"
  },
  "error": "operation write contract violated: tested changed read-only/undeclared bytes in x",
  "fixture_sha256": "34afac3bda864c0fa87cdf7569779f0d0f92c30deb283013c5f75e7901fa7fc3",
  "samples_us": [],
  "schema": "radiance-replay-worker-result/1",
  "status": "FAIL",
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
