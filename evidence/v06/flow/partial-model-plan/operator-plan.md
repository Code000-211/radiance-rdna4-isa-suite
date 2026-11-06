# v0.6 operator scenarios

No GPU was executed. Every new operator ranking needs target measurement.

| Workload | Candidate | Scenario µs | Decision |
|---|---|---:|---|
| L0.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L0.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L0.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L0.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L0.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L0.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L0.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L0.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L1.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L1.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L1.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L1.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L1.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L1.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L1.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L1.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L2.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L2.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L2.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L2.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L2.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L2.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L2.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L2.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L3.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L3.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L3.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L3.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L3.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L3.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L3.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L3.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L3.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L3.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L3.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L3.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L3.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L3.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L3.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L3.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L3.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L3.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L3.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L3.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L3.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L3.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L3.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L3.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L3.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L3.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L3.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L3.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L3.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L3.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L3.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L3.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L3.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L3.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L3.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L3.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L3.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L3.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L3.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L3.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L3.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L3.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L4.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L4.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L4.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L4.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L4.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L4.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L4.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L4.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L5.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L5.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L5.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L5.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L5.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L5.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L5.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L5.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L6.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L6.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L6.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L6.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L6.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L6.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L6.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L6.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L7.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L7.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L7.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L7.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L7.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L7.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L7.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L7.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L7.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L7.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L7.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L7.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L7.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L7.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L7.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L7.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L7.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L7.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L7.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L7.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L7.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L7.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L7.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L7.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L7.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L7.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L7.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L7.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L7.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L7.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L7.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L7.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L7.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L7.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L7.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L7.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L7.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L7.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L7.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L7.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L7.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L7.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L8.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L8.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L8.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L8.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L8.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L8.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L8.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L8.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L9.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L9.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L9.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L9.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L9.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L9.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L9.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L9.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L10.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L10.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L10.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L10.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L10.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L10.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L10.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L10.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L11.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L11.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L11.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L11.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L11.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L11.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L11.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L11.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L11.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L11.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L11.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L11.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L11.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L11.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L11.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L11.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L11.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L11.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L11.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L11.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L11.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L11.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L11.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L11.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L11.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L11.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L11.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L11.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L11.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L11.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L11.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L11.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L11.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L11.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L11.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L11.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L11.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L11.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L11.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L11.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L11.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L11.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L12.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L12.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L12.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L12.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L12.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L12.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L12.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L12.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L13.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L13.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L13.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L13.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L13.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L13.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L13.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L13.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L14.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L14.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L14.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L14.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L14.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L14.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L14.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L14.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L15.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L15.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L15.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L15.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L15.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L15.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L15.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L15.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L15.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L15.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L15.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L15.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L15.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L15.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L15.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L15.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L15.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L15.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L15.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L15.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L15.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L15.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L15.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L15.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L15.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L15.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L15.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L15.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L15.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L15.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L15.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L15.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L15.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L15.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L15.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L15.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L15.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L15.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L15.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L15.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L15.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L15.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L16.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L16.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L16.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L16.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L16.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L16.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L16.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L16.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L17.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L17.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L17.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L17.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L17.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L17.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L17.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L17.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L18.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L18.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L18.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L18.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L18.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L18.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L18.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L18.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L19.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L19.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L19.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L19.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L19.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L19.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L19.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L19.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L19.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L19.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L19.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L19.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L19.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L19.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L19.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L19.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L19.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L19.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L19.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L19.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L19.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L19.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L19.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L19.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L19.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L19.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L19.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L19.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L19.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L19.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L19.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L19.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L19.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L19.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L19.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L19.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L19.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L19.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L19.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L19.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L19.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L19.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L20.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L20.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L20.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L20.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L20.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L20.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L20.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L20.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L21.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L21.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L21.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L21.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L21.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L21.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L21.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L21.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L22.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L22.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L22.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L22.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L22.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L22.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L22.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L22.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L23.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L23.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L23.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L23.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L23.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L23.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L23.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L23.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L23.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L23.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L23.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L23.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L23.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L23.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L23.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L23.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L23.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L23.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L23.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L23.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L23.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L23.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L23.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L23.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L23.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L23.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L23.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L23.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L23.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L23.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L23.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L23.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L23.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L23.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L23.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L23.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L23.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L23.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L23.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L23.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L23.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L23.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L24.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L24.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L24.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L24.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L24.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L24.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L24.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L24.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L25.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L25.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L25.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L25.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L25.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L25.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L25.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L25.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L26.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L26.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L26.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L26.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L26.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L26.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L26.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L26.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L27.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L27.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L27.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L27.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L27.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L27.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L27.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L27.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L27.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L27.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L27.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L27.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L27.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L27.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L27.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L27.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L27.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L27.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L27.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L27.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L27.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L27.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L27.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L27.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L27.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L27.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L27.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L27.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L27.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L27.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L27.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L27.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L27.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L27.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L27.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L27.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L27.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L27.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L27.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L27.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L27.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L27.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L28.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L28.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L28.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L28.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L28.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L28.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L28.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L28.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L29.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L29.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L29.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L29.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L29.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L29.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L29.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L29.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L30.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L30.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L30.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L30.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L30.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L30.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L30.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L30.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L31.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L31.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L31.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L31.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L31.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L31.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L31.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L31.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L31.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L31.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L31.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L31.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L31.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L31.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L31.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L31.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L31.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L31.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L31.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L31.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L31.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L31.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L31.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L31.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L31.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L31.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L31.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L31.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L31.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L31.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L31.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L31.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L31.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L31.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L31.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L31.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L31.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L31.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L31.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L31.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L31.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L31.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L32.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L32.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L32.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L32.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L32.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L32.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L32.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L32.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L33.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L33.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L33.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L33.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L33.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L33.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L33.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L33.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L34.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L34.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L34.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L34.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L34.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L34.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L34.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L34.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L35.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L35.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L35.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L35.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L35.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L35.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L35.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L35.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L35.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L35.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L35.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L35.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L35.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L35.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L35.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L35.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L35.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L35.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L35.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L35.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L35.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L35.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L35.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L35.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L35.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L35.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L35.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L35.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L35.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L35.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L35.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L35.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L35.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L35.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L35.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L35.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L35.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L35.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L35.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L35.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L35.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L35.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L36.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L36.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L36.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L36.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L36.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L36.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L36.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L36.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L37.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L37.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L37.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L37.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L37.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L37.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L37.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L37.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L38.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L38.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L38.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L38.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L38.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L38.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L38.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L38.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L39.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L39.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L39.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L39.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L39.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L39.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L39.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L39.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L39.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L39.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L39.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L39.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L39.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L39.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L39.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L39.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L39.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L39.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L39.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L39.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L39.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L39.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L39.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L39.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L39.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L39.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L39.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L39.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L39.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L39.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L39.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L39.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L39.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L39.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L39.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L39.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L39.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L39.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L39.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L39.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L39.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L39.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L40.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L40.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L40.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L40.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L40.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L40.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L40.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L40.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L41.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L41.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L41.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L41.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L41.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L41.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L41.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L41.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L42.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L42.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L42.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L42.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L42.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L42.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L42.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L42.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L43.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L43.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L43.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L43.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L43.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L43.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L43.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L43.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L43.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L43.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L43.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L43.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L43.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L43.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L43.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L43.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L43.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L43.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L43.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L43.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L43.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L43.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L43.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L43.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L43.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L43.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L43.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L43.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L43.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L43.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L43.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L43.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L43.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L43.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L43.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L43.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L43.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L43.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L43.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L43.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L43.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L43.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L44.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L44.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L44.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L44.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L44.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L44.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L44.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L44.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L45.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L45.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L45.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L45.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L45.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L45.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L45.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L45.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L46.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L46.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L46.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L46.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L46.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L46.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L46.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L46.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
| L47.r0.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L47.r0.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L47.r0.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L47.r0.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L47.r0.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L47.r0.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L47.r0.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L47.r0.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L47.r0.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L47.r0.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L47.r0.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L47.r0.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L47.r0.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L47.r0.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L47.r0.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L47.r0.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L47.r0.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L47.r0.moe | moe.w4a8.m16.n128 | 178.548 | NEEDS_MEASUREMENT |
| L47.r0.moe | moe.w4a8.m16.n64 | 178.548 | NEEDS_MEASUREMENT |
| L47.r0.moe | moe.w4a8.m32.n128 | 178.548 | NEEDS_MEASUREMENT |
| L47.r0.moe | moe.w4a8.m32.n64 | 178.548 | NEEDS_MEASUREMENT |
| L47.r1.qsa | qsa.radix.t128 | 210.912 | NEEDS_MEASUREMENT |
| L47.r1.qsa | qsa.radix.t32 | 210.912 | NEEDS_MEASUREMENT |
| L47.r1.qsa | qsa.radix.t64 | 210.912 | NEEDS_MEASUREMENT |
| L47.r1.attn | attn.g12.8.prefill.s1 | 15.2952 | NEEDS_MEASUREMENT |
| L47.r1.attn | attn.g12.8.decode.s1 | 18.4334 | NEEDS_MEASUREMENT |
| L47.r1.attn | attn.g12.8.decode.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L47.r1.attn | attn.g12.8.prefill.s4 | 18.6423 | NEEDS_MEASUREMENT |
| L47.r1.attn | attn.g12.8.decode.s16 | 20.0983 | NEEDS_MEASUREMENT |
| L47.r1.attn | attn.g12.8.prefill.s16 | 20.0983 | NEEDS_MEASUREMENT |

Rejected `L47.r1.attn` / `attn.g12.16.decode.s1`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g12.16.decode.s4`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g12.16.decode.s16`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g12.16.prefill.s1`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g12.16.prefill.s4`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g12.16.prefill.s16`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g6.8.decode.s1`: candidate attention geometry differs

Rejected `L47.r1.attn` / `attn.g6.8.decode.s4`: candidate attention geometry differs

Rejected `L47.r1.attn` / `attn.g6.8.decode.s16`: candidate attention geometry differs

Rejected `L47.r1.attn` / `attn.g6.8.prefill.s1`: candidate attention geometry differs

Rejected `L47.r1.attn` / `attn.g6.8.prefill.s4`: candidate attention geometry differs

Rejected `L47.r1.attn` / `attn.g6.8.prefill.s16`: candidate attention geometry differs

Rejected `L47.r1.attn` / `attn.g6.16.decode.s1`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g6.16.decode.s4`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g6.16.decode.s16`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g6.16.prefill.s1`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g6.16.prefill.s4`: different numerical/layout contract

Rejected `L47.r1.attn` / `attn.g6.16.prefill.s16`: different numerical/layout contract
| L47.r1.moe | moe.w4a8.m16.n128 | 177.439 | NEEDS_MEASUREMENT |
| L47.r1.moe | moe.w4a8.m16.n64 | 177.439 | NEEDS_MEASUREMENT |
| L47.r1.moe | moe.w4a8.m32.n128 | 177.439 | NEEDS_MEASUREMENT |
| L47.r1.moe | moe.w4a8.m32.n64 | 177.439 | NEEDS_MEASUREMENT |
