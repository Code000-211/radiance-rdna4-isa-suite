# Radiance kernel snapshot replay

**Status: FAIL**

This report covers this fixture only. It is not full ISA, memory-safety, or GPU-wedge certification.

## Input and execution evidence

```json
{
  "schema": "radiance-kernel-replay/1",
  "status": "FIXTURE_VALID",
  "manifest_sha256": "1a5f276929c29f2e4f2e36acb6d0574c89dc7d70878ce5c6b0eb2508f6224788",
  "id": "missing-store",
  "allocation_bytes_including_scratch": 8,
  "blob_bytes": 12,
  "kernel_nodes": [
    "tested"
  ],
  "node_count": 1,
  "stream_count": 1,
  "kernel_rows": [
    "replay_test:audit_no_store"
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
  "checks_per_iteration": 1,
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
    "completed_passes": 1,
    "intermediate_equality": "explicit checkpoint tolerances, then exact bytes for remaining writes against same-candidate baseline",
    "output_poison_variants": 0,
    "per_operation_write_contracts": false,
    "race_freedom_proved": false,
    "read_checks": "NOT_CHECKED",
    "scratch_contracts": false,
    "status": "RUNNING"
  },
  "error": "diagnostic output/scratch invariance failed at tested; exact intermediate bytes differ (not a performance or race-freedom claim)",
  "fixture_sha256": "1a5f276929c29f2e4f2e36acb6d0574c89dc7d70878ce5c6b0eb2508f6224788",
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
