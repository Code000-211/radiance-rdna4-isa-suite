# feat(test): add portable kernel snapshots and standalone launcher replay

## Summary

Add v0.3 replay infrastructure alongside the unchanged v0.2 RDNA4 binary gate.
Snapshots preserve full operand bytes, aliasing, typed geometry, frozen tuning
choices and explicit relocations. An engine-free worker uses the real Radiance
registry/device ABI and existing launchers, verifies every warm-up/timed iteration,
and reports correctly labeled operation-window timings.

This is a **single-rank ordered-launcher replay stage**, not complete raw AQL tape
or TP2 production equivalence. Unsupported cases remain INCOMPLETE. No kernel
arithmetic, WMMA layout, wait instruction, cache hint or communication protocol
is rewritten by this PR.

## Changes

- Strict content-addressed fixture validator and Python construction SDK.
- Native worker with alias/relocation restore, schema/row/tuning checks, prefix
  and measured windows, explicit event dependencies and bounded scratch.
- Exact/numeric expected output, nonfinite checks, allocation/scratch redzones,
  immutable and undeclared-write checks outside the timing interval.
- Child-process supervision, timeout/error evidence, JSON/Markdown/JUnit reports.
- Hash-bound static plugin evidence, explicit candidate plugin rebinding and
  independently approved corpus coverage with no silent skips.
- Optional kBench pre/post byte-export patch and CMake target; disabled by default.
- Real-source host test build, real libref chain/capture tests and host-only
  negative test plugin. No intentionally invalid code is executed on a GPU.

## Evidence and limits

Read `docs/V03_REVIEW.md`, `docs/REPLAY_FORMAT.md`, `docs/MEASUREMENT.md` and
`evidence/RELEASE_VALIDATION.json`. Native host tests and Python/static tests were
executed; GPU/HIP/HSA integration and production snapshots were not. Full build
of the supplied repository is blocked by its missing `core/kvstream/bind.h`;
this PR does not invent a replacement. The patched kBench translation unit was
syntax-checked, not represented as a completed kBench executable test.

## Recommended commits

1. Portable format/SDK, validation tests and explicit scope documentation.
2. Native standalone worker, reports and supervisor tests.
3. Optional kBench capture/CMake integration and component tests.
4. Target-host fixtures and GPU results in a separate evidence commit.

## Target-host merge checklist

- [ ] Review source/version hashes and run `check.sh`.
- [ ] Build real libref and run `check-replay-host.sh` against the intended checkout.
- [ ] Apply the optional patch only after checking its context against that checkout.
- [ ] Restore a complete upstream source tree; build full kBench/worker with HIP/HSA.
- [ ] Verify capture disabled leaves ordinary kBench behavior unchanged.
- [ ] Capture one small supported single-GPU case and review every buffer/parameter.
- [ ] Strictly scan relevant final GPU code; reconcile runtime support kernels too.
- [ ] Run the snapshot with numerical/guard checks on the actual gfx1201 device.
- [ ] Retain original independent kBench reference results, not just saved output.
- [ ] Compare gated and host-paced timing labels; calibrate against engine windows.
- [ ] Review an explicit required-case corpus; do not count omitted/capped cases.
- [ ] Keep raw-tape, TP2 and hidden-state support separate until implemented/tested.

`RUNTIME_CHECKS_PASS` certifies only the declared fixture checks that actually ran.
It is not complete ISA, memory-safety, numerical or GPU-wedge certification.
