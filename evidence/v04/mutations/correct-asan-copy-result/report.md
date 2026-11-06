# Radiance kernel snapshot replay

**Status: RUNTIME_CHECKS_PASS**

This report covers this fixture only. It is not full ISA, memory-safety, or GPU-wedge certification.

## Input and execution evidence

```json
{
  "schema": "radiance-kernel-replay/1",
  "status": "FIXTURE_VALID",
  "manifest_sha256": "514b08aa6ab8ea93c08a55c1dc216ed74fba3026bd1e0a237f0357a699814da0",
  "id": "correct-asan-copy",
  "allocation_bytes_including_scratch": 8,
  "blob_bytes": 12,
  "kernel_nodes": [
    "tested"
  ],
  "node_count": 1,
  "stream_count": 1,
  "kernel_rows": [
    "replay_test:copy"
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
    "checked_nodes": 1,
    "completed_passes": 3,
    "explicit_checkpoints": 0,
    "intermediate_equality": "explicit checkpoint tolerances, then exact bytes for remaining writes against same-candidate baseline",
    "output_poison_scope": "packed exact-span output-only operands, excluding all readable aliases; other views not covered",
    "output_poison_variants": 2,
    "per_operation_write_contracts": true,
    "poisoned_output_bytes_total": 8,
    "race_freedom_proved": false,
    "read_checks": "HOST_ASAN_INSTRUMENTED_PATHS",
    "scratch_contracts": true,
    "status": "PASS"
  },
  "fixture_sha256": "514b08aa6ab8ea93c08a55c1dc216ed74fba3026bd1e0a237f0357a699814da0",
  "measurement": "NOT_MEASURED_DIAGNOSTIC_ONLY",
  "reference": "independent",
  "samples_us": [],
  "schema": "radiance-replay-worker-result/1",
  "statistics": null,
  "status": "RUNTIME_CHECKS_PASS",
  "worker_build": {
    "compiler": "Clang 17.0.0 (https://github.com/swiftlang/llvm-project.git 10999b6d034fe318f3d56c83bddb6572593a8bb0)",
    "cxx": 202002,
    "host_asan": true,
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
