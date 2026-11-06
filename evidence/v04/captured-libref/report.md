# Radiance kernel snapshot replay

**Status: RUNTIME_CHECKS_PASS**

This report covers this fixture only. It is not full ISA, memory-safety, or GPU-wedge certification.

## Input and execution evidence

```json
{
  "schema": "radiance-kernel-replay/1",
  "status": "FIXTURE_VALID",
  "manifest_sha256": "ef2ee42df925f436576f4280e42416d28d1d26877900ba2e94f7153a64789985",
  "id": "case-00000001",
  "allocation_bytes_including_scratch": 336,
  "blob_bytes": 352,
  "kernel_nodes": [
    "kernel0"
  ],
  "node_count": 1,
  "stream_count": 1,
  "kernel_rows": [
    "libref:ref_add"
  ],
  "provenance": {
    "kind": "kbench-seeded",
    "note": "Exact kBench invocation bytes; output is a regression baseline, not an independent oracle. Original kBench verdict is separate.",
    "reference": "captured-baseline"
  },
  "runtime_executed": false,
  "all_radiance_kernels_covered": false
}
```

## Runtime

```json
{
  "backend": "host",
  "checked_iterations": 4,
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
  "dispatch_plan": "typed pre-resolved launcher plan; no JSON/disk I/O in measured submit loop",
  "fixture_sha256": "ef2ee42df925f436576f4280e42416d28d1d26877900ba2e94f7153a64789985",
  "measurement": "host ordered launcher-window steady_clock time",
  "physical_bytes_with_guards": 2640,
  "reference": "captured-baseline",
  "restore_policy": "restore every allocation and zero scratch before every checked iteration",
  "samples_us": [
    0.821,
    0.521,
    0.391
  ],
  "schema": "radiance-replay-worker-result/1",
  "state_policy": "no hidden state; warmups and timed iterations use identical pristine bytes",
  "statistics": {
    "median": 0.521,
    "min": 0.391,
    "n": 3,
    "p90": 0.821,
    "p90_is_per_dispatch": false,
    "tokens_per_second_estimated": false,
    "unit": "us per complete window"
  },
  "status": "RUNTIME_CHECKS_PASS",
  "worker_build": {
    "compiler": "14.2.0",
    "cxx": 202002,
    "radiance_abi": 30
  }
}
```

## Limits

- Ordered launcher-level replay; not arbitrary raw AQL packets or device-side chains.
- Single device/rank only. Captured baseline is a regression oracle, not independent correctness.
- Guard bytes detect observed writes, not invalid reads or transient writes restored later.
- Queue-window time is not per-dispatch latency and is never converted to tokens/s.
