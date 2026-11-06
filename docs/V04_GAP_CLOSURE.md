# Gap closure and remaining proof obligations

This table separates implemented detectors from evidence that requires target
hardware or a further execution backend. “Blocked” is not “implemented.”

| Previously identified gap | v0.4 response | Remaining boundary |
|---|---|---|
| OOB read leaves guard bytes unchanged | Host ASan worker/plugin mode + exact expressible per-operation poisoned ranges; prefix/suffix and intra-allocation view-overread mutations | **No certified gfx1201/AQL GPU read instrumentation.** Plain guards still miss reads. |
| No store when output initially matches | Two complementary pure-output patterns; no-store and partial-store mutations | Packed exact-span output-only views; excludes input aliases, padding/custom layouts. Not exhaustive store coverage. |
| Scratch read-before-write masked by zero | Explicit `zeroed` vs `kernel-initialized`; nonzero perturbation passes | Variation is a dependency test, not MemorySanitizer; hidden persistent state remains unsupported. |
| Earlier illegal write hidden by later legal overwrite | Per-operation write snapshots/guards and optional intermediate references | Diagnostics serialize operations. Same-operation write-and-restore and races can remain invisible. |
| Missing GPU validation | Executable target-host verification recipe and explicit gate | **No GPU or HIP compiler was available.** Not converted into a pass. |
| No boundary corpus | 46 generated finite-f32 GEMM cases, tile-adjacent axes and data distributions, independent Python `fsum` oracle | One host row; not all GPU shapes or quantized formats. Requires per-operation generators/oracles. |
| Reference provenance/tolerance ambiguity | Independent-vs-baseline provenance plus versioned op/dtype numeric-policy qualification | Provenance is engineering evidence, not authenticated proof of how someone generated values. |
| No continued-state testing | Existing within-window state retained; a 32-step in-place sequence has independent per-step checkpoints | Each repeat still restores the window's initial state. No opaque instance/device-global lifecycle support. |
| No actual dispatch mapping | Optional diagnostic AQL hook + pre-submission whitelist and post-run binding including observed helpers | Full target integration uncompiled/unrun. Does not cover tape replay, forwarded HIP paths, SDMA or asynchronous background submissions. |
| Corpus is not all-kernel coverage | Native plugin row inventory as independent denominator; missing rows stay NOT_TESTED | Registrations do not enumerate every shape, tuning combination or binary specialization. |
| Benchmark drift/cherry-picking | ABBA blocks, identical workloads, environment/mode check, stop-on-first-failure, retained raw attempts | No guaranteed clock/temperature stability. No actual engine-window calibration supplied. |
| TP2/P2P and production replay absent | Strict `production-tp2` qualification always lists the missing backend/protocol/calibration | **Not implemented.** Do not execute a waiting collective on one rank and call it tested. |

## Why a single “all fixed” flag would be wrong

A runtime pass concerns its declared fixture checks. A static pass concerns its
submitted binaries and implemented rules. A metadata row count concerns declared
registrations. None implies complete pointer validity, race freedom, all
specializations or production performance. v0.4 makes those dimensions visible and
refuses broader claims when their evidence is missing.

## Required target-host work (not fabricated here)

Start with one supported single-GPU row, a reviewed snapshot and its independent
reference. Build the complete intended HIP/HSA toolchain, apply the optional hooks
separately, strictly scan final binaries and helpers, run diagnostic dispatch
tracing, then measure without diagnostic instrumentation. Preserve first-fault
logs. Do not bundle any speculative kernel/cache/wait/undervolt changes into the
validation step. TP2 requires a genuine multi-rank protocol runner before launch.
