# v0.6 operator scenarios

No GPU was executed. Every new operator ranking needs target measurement.

| Workload | Candidate | Scenario µs | Decision |
|---|---|---:|---|
| qsa.r0 | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| qsa.r0 | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| qsa.r0 | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| attn.r0 | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| attn.r0 | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| attn.r0 | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| attn.r0 | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| attn.r0 | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| attn.r0 | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `attn.r0` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `attn.r0` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `attn.r0` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `attn.r0` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `attn.r0` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `attn.r0` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `attn.r0` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `attn.r0` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `attn.r0` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `attn.r0` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `attn.r0` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `attn.r0` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `attn.r0` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `attn.r0` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `attn.r0` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `attn.r0` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `attn.r0` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `attn.r0` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| moe.r0 | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| moe.r0 | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| moe.r0 | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| moe.r0 | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| qsa.r1 | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| qsa.r1 | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| qsa.r1 | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| attn.r1 | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| attn.r1 | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| attn.r1 | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| attn.r1 | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| attn.r1 | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| attn.r1 | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `attn.r1` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `attn.r1` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `attn.r1` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `attn.r1` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `attn.r1` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `attn.r1` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `attn.r1` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `attn.r1` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `attn.r1` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `attn.r1` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `attn.r1` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `attn.r1` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `attn.r1` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `attn.r1` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `attn.r1` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `attn.r1` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `attn.r1` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `attn.r1` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| moe.r1 | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| moe.r1 | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| moe.r1 | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| moe.r1 | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| ar | tp2.oneshot.push.c65536 | 7.45143 | NEEDS_MEASUREMENT |
| ar | tp2.ring2.push.c1073741824 | 10.3657 | NEEDS_MEASUREMENT |
| ar | tp2.oneshot.pull.c65536 | 16.4114 | NEEDS_MEASUREMENT |
| ar | tp2.oneshot.push.c4096 | 19.2869 | NEEDS_MEASUREMENT |
| ar | tp2.ring2.pull.c1073741824 | 19.3257 | NEEDS_MEASUREMENT |
| ar | tp2.oneshot.pull.c4096 | 28.2469 | NEEDS_MEASUREMENT |
