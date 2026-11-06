# Radiance kernel snapshot replay

**Status: FAIL**

This report covers this fixture only. It is not full ISA, memory-safety, or GPU-wedge certification.

## Input and execution evidence

```json
{
  "schema": "radiance-kernel-replay/1",
  "status": "FIXTURE_VALID",
  "manifest_sha256": "09126e11ac2aa67c019e087615b8524380b0682d2db07a906055d407215bea70",
  "id": "overread-asan",
  "allocation_bytes_including_scratch": 8,
  "blob_bytes": 12,
  "kernel_nodes": [
    "tested"
  ],
  "node_count": 1,
  "stream_count": 1,
  "kernel_rows": [
    "replay_test:audit_oob_read"
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
  "status": "FAIL",
  "backend": "host",
  "memory_check": "HOST_ASAN_DETECTED_ERROR",
  "fixture_sha256": "09126e11ac2aa67c019e087615b8524380b0682d2db07a906055d407215bea70"
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

## Failure

Host AddressSanitizer detected an invalid memory access; see worker.stderr
