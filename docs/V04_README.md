# Radiance RDNA4 ISA + standalone replay testsuite — v0.4.0

Repository overlay: `tools/rdna4-isa-suite/`. No inference-engine startup and no
kernel arithmetic changes. The v0.2 static ISA checker remains unchanged. This
release hardens the v0.3 launcher replay and adds explicit, scoped acceptance.

**This is not complete GPU memory-safety, TP2, or production certification.**
The four supplied host counterexamples are now detected in the appropriate modes:
missing/partial stores, scratch-initialization dependencies, an illegal write
hidden by a later operation, and out-of-bounds reads with **instrumented host ASan**.
An ordinary uninstrumented run still does not observe a read that has no visible
side effect. No R9700 device sanitizer is claimed.

## Start here: one reproducible host verification command

Requirements: Python 3.10+, CMake 3.21+, a C++20 compiler, `clang++` with its host
AddressSanitizer runtime, and an LLVM decoder supporting gfx1201. Tests refuse
missing prerequisites rather than skipping them. Linux is the tested host.

From the Radiance checkout, after extracting this overlay:

```bash
bash tools/rdna4-isa-suite/check-all-host.sh "$PWD" "$PWD/replay-v04-validation"
```

Use a **new** work directory. This builds the worker, probe plugin, capture harness,
metadata inventory tool and real supplied-source `libref` without linking the
serving engine; runs static/Python/native/ASan tests; executes the generated 46-case
finite-f32 GEMM corpus; and performs scoped qualification. The deliberate faulty
plugin is **HOST-only**. Do not port those faulty probes to a GPU.

The current source snapshot lacks `core/kvstream/bind.h`; the standalone host build
does not require it and does not invent a stub. A whole HIP/HSA/engine build is
still a separate target-host prerequisite.

## Four different answers, not one misleading green badge

| Result | Meaning |
|---|---|
| `STATIC_CHECKS_PASS` | Implemented static rules passed for the explicit byte inventory. |
| `RUNTIME_CHECKS_PASS` | Declared checks for this fixture actually completed. Read/race/coverage scope stays explicit. |
| `SCOPED_QUALIFICATION_PASS` | All requirements of the named profile and exact approved corpus passed. |
| `INCOMPLETE` | Missing, unsupported, stale or partial evidence; not success. |

`host-contracts` can pass in this release. The stricter `single-gpu` profile
requires GPU read-instrumentation evidence, which this release **does not provide**;
`production-tp2` additionally blocks on missing TP2/raw-tape/production calibration.
A caller-supplied boolean cannot turn these missing implementations into a pass.

## New diagnostic modes

`replay.sh run` defaults to `--diagnostics contracts`: three serial passes precede
normal uninstrumented timing. They check per-operation write permissions, explicit
intermediate oracles, two complementary output-only initialization patterns and
scratch contracts. Final outputs and guards are also checked after each normal
warm-up/timed window. Diagnostics and their synchronization are not timed.

Pure outputs are perturbed only for packed, exact-span supported dtypes, excluding
all readable aliases. INOUT data, input aliases, custom encodings and padding are
not blindly overwritten. Reports state the actual number of perturbed bytes.

For scratch-using kernel nodes, declare one of:

```json
"scratch_policy": "zeroed"
```

or `"kernel-initialized"`. `zeroed` is an explicit caller contract, not a guessed
property of every kernel. The latter is checked with zero and two nonzero initial
patterns. Absent/`unspecified` remains parseable for old fixtures but **blocks
contract diagnostics** until its real contract is supplied.

Intermediate references use `Package.checkpoint(after, view, expected, ...)` or
the fixture's `checkpoints` array. Declared checkpoint tolerances are respected;
remaining written bytes must match the same-candidate baseline exactly. Without
an explicit approximate checkpoint, nondeterministic numerical output may require
review rather than loosening a tolerance after seeing failure.

```bash
bash tools/rdna4-isa-suite/replay.sh run path/to/manifest.json \
  --worker path/to/rad-kbench-replay --plugin p0=path/to/plugin.so \
  --out new-result --execute-trusted-code --diagnostic-only
```

`--diagnostics none` is explicitly exploratory legacy behavior. It cannot satisfy
v0.4 scoped contract qualification, even if final values match.

## Instrumented host reads

Build the worker **and the invoked plugin** with `RAD_REPLAY_HOST_ASAN=ON` (the
host-test CMake option); run:

```bash
bash tools/rdna4-isa-suite/replay.sh run path/to/host-fixture/manifest.json \
  --worker asan-build/rad-kbench-replay --plugin p0=asan-build/plugin.so \
  --out new-asan-result --execute-trusted-code --memory-check host-asan \
  --diagnostic-only
```

Rebind a fixture explicitly when instrumentation changes the plugin hash. The
supervisor verifies necessary ASan symbols; the worker poisons allocation guards
and undeclared per-operation ranges. Unrepresentable lower shadow boundaries fail
incomplete. Symbol presence is not proof that inline assembly or every linked
library was instrumented. ASan builds cannot generate benchmark evidence.

## Exact diagnostic AQL dispatches (optional target-host integration)

Apply `integration/aql-capture.patch` first, then
`integration/aql-dispatch-trace.patch`, after checking both against the intended
checkout. Neither is applied automatically. Rebuild an AQL worker. The second
patch observes the ordinary dispatcher **before queue reservation** and rejects
identities outside the strict static whitelist. It records object SHA-256,
descriptor symbol, node/pass, device, queue, dimensions and resource sizes.

```bash
bash tools/rdna4-isa-suite/check-target-gpu.sh \
  approved-gpu-corpus/expected.json path/to/aql-worker p0 path/to/libr4d.so \
  new-target-evidence
```

This collects/scans both plugin and worker helper binaries, then executes a
**diagnostic-only** corpus with `--trace-dispatches`. Every kernel node must be
observed in all three diagnostic passes. Unobserved nodes, unknown helpers,
background-thread submissions, missing hook, trace truncation and stale hashes
block acceptance. These are observed diagnostic dispatches, **not** timed-stream,
raw-tape, SDMA or TP2 equivalence. The full HIP/HSA integration was not compiled or
run in the authoring environment; the patch application and host collector were
tested. Do not interpret that as target-host evidence.

## Explicit corpus, row coverage and numeric policy

```bash
# Pin required fixtures BEFORE collecting their results.
bash tools/rdna4-isa-suite/replay.sh expect cases/*/manifest.json \
  --out cases/expected.json --scope 'approved single-device test shapes'

# Stop on the first failure/timeout/incomplete run. No resets or automatic retry.
bash tools/rdna4-isa-suite/replay.sh run-corpus --expect cases/expected.json \
  --worker build/rad-kbench-replay --plugin p0=build/plugin.so \
  --out corpus-results --execute-trusted-code

# Enumerate actual declared plugin rows; native initializers run, so trust the plugin.
build/rad-replay-inventory build/plugin.so new-rows.json
```

`row-coverage --inventory ... --manifest ... --report ... --out ...` uses that
independent row denominator. A registration is not the same as all its shapes,
tuning combinations or GPU specializations. Untested rows remain `NOT_TESTED`.

`qualify --expect ... --report ... --profile host-contracts --numeric-policy ...
--out ...` also requires independent-oracle provenance and a versioned explicit
op/dtype tolerance policy for numeric checks. See `examples/boundary_corpus.py`.
The generated corpus is **finite dense f32 GEMM only**, not QSA/MoE/quantization
quality coverage. The supplied `libref` has 210 declared rows; those 46 fixtures
exercise just `ref_gemm_nt`.

## Interleaved A/B, not two long sequential benchmark runs

```bash
bash tools/rdna4-isa-suite/replay.sh ab \
  --baseline baseline/manifest.json --candidate candidate/manifest.json \
  --baseline-worker build/rad-kbench-replay --candidate-worker build/rad-kbench-replay \
  --baseline-plugin p0=old/plugin.so --candidate-plugin p0=new/plugin.so \
  --out new-ab --execute-trusted-code --blocks 5 --warmups 2 --reps 20
```

A/B uses ABBA blocks, retains every attempt, stops on the first nonpass and requires
identical inputs, outputs, tolerances, row/tuning, device and measurement mode.
Only explicit candidate binary hashes/lineage may differ. Per-run medians are
compared in paired blocks, with an exploratory whole-block bootstrap interval.
No ASan timing, fabricated per-dispatch percentiles, automatic clock changes,
production calibration or tokens/s estimates are produced.

## Evidence and boundaries

See `docs/V04_REVIEW.md`, `docs/V04_GAP_CLOSURE.md`, `PR_DESCRIPTION.md` and
`evidence/RELEASE_VALIDATION.json`. Historical evidence is under `evidence/v03/`
and `evidence/v02/`; it is not a new execution of those versions.

No arbitrary native-plugin sandbox, formal access proof, complete memory-hazard
analysis, valid oracle for every operation, real production activation corpus,
TP2/P2P runner, opaque-instance lifecycle, device chains or raw-tape export is
claimed. A process timeout is not a GPU recovery mechanism. Stop target testing
after a device fault; never disable watchdogs to make the suite green.
