# v0.6 graph scenario

**Not measured; not a production throughput forecast.**

Selected scenario: 185.999 µs; serialized model: 369.378 µs.
Search: exhaustive fixed-policy scenarios. Evaluations: 96.

| Node | Candidate | Start µs | Finish µs |
|---|---|---:|---:|
| rank0 | moe.w4a8.m16.n128 | 0 | 178.548 |
| rank1 | moe.w4a8.m16.n128 | 0 | 177.439 |
| reduce | tp2.oneshot.push.c65536 | 178.548 | 185.999 |

## Limits

- Fixed manifest/queue admission, equal resource sharing, no preemption or register-placement simulation.
- Resource floor is conditional on model and profile, not a hardware latency bound.
- Allocation sizes/lifetimes must be complete; opaque allocator or engine buffers are not inferred.
- Concurrent timings must be validated as graph windows, not summed isolated kernels.
