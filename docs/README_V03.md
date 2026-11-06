# Radiance RDNA4 ISA + kernel replay suite — v0.3

**New:** an engine-free, launcher-level snapshot runner using Radiance's actual
registry, ABI, device backend and kBench comparison code. The v0.2 binary ISA gate
is retained unchanged. This release is **host-tested; the AQL/GPU path is
implemented but not executed or HIP/HSA-built in the authoring environment**.

This is not yet complete production replay of all Radiance kernels. It supports
single-rank, stateless, explicitly described operations and short operation
windows. Raw AQL tape, device-side chaining, TP2/P2P, instance-managed state and
automatic production-engine snapshots remain unsupported. Unsupported cases do
not acquire a green result. See `docs/V03_REVIEW.md` before merging.

## Contents

| Component | Role |
|---|---|
| `radisa/`, `run.sh` | Existing CPU-only binary gate; documentation in `docs/ISA_V02.md`. |
| `radreplay/`, `replay.sh` | Strict fixture validation, process supervisor, hash-pinned evidence, explicit candidate rebinding and scoped corpus coverage. |
| `replay/worker.cpp` | Native executor calling existing Radiance plugin launchers. No model, tokenizer, architecture plugin, request scheduler or server is needed. |
| `replay/capture.h` | Opt-in exact kBench pre/post operand capture, including overlapping views. |
| `integration/kbench-replay.patch` | Optional kBench/CMake integration against the supplied source snapshot. |
| `examples/libref_chain.py` | Model-free synthetic add → in-place multiply window with independent exact expected values. |
| `check-replay-host.sh` | Compile and test real Radiance host code plus clearly labeled host-only negative fixtures. |

No original Radiance kernel, synchronization instruction or memory hint is
modified by extracting this overlay. Replace/back up the previous suite directory
rather than mixing old and new Python modules. No downloaded GPU code or model
weights are included. The pre-existing tiny ISA fixture is **static-test only**.

## 1. Test the analysis/validation tools

From the Radiance root:

```bash
bash tools/rdna4-isa-suite/check.sh
```

Requires Python 3.10+, a gfx1201-capable LLVM decoder, and the host C++ compiler
used by the retained ISA capture tests. This checks the ISA suite and Python replay
contracts. It **does not stand in for the compiled replay-worker tests** below.

## 2. Build the replay worker

### Full repository integration (complete Radiance checkout required)

```bash
git apply --check tools/rdna4-isa-suite/integration/kbench-replay.patch
git apply tools/rdna4-isa-suite/integration/kbench-replay.patch
cmake -S . -B build -DRAD_BUILD_REPLAY_SUITE=ON
cmake --build build --target rad-kbench rad-kbench-replay -j2
```

Keep your normal ROCm, GPU target and other CMake options. The worker uses the same
compiled device backend as this Radiance build; a host-only binary refuses an AQL
fixture. Full integration uses C++20 and the repository's vendored JSON header.

**Source-snapshot limitation:** the supplied `radiance-main(1).zip` lacks
`core/kvstream/bind.h`, which `core/runtime/ctx.cpp` includes. Its full host build
fails before completion. No invented stub was added. The patch applies and the
modified kBench translation unit passes a syntax check, but a full patched engine
build must be done against a complete checkout on the target host.

### Independently tested host path

Once the genuine `libref.so` target is built, the following compiles just the
necessary **real Radiance sources** and runs the native replay tests:

```bash
bash tools/rdna4-isa-suite/check-replay-host.sh \
  "$PWD" "$PWD/build/radiance_home/kernels/libref.so" \
  "$PWD/build-replay-host"
```

Adjust the libref path to your build. This route does not use an engine mock and
avoids the missing serving/runtime source above. It is a HOST diagnostic build,
not an alternative GPU implementation.

## 3. Run a complete, model-free example

```bash
python3 tools/rdna4-isa-suite/examples/libref_chain.py \
  --plugin "$PWD/build/radiance_home/kernels/libref.so" --out ./replay-demo

bash tools/rdna4-isa-suite/replay.sh validate ./replay-demo/manifest.json

bash tools/rdna4-isa-suite/replay.sh run ./replay-demo/manifest.json \
  --worker ./build-replay-host/rad-kbench-replay \
  --plugin p0="$PWD/build/radiance_home/kernels/libref.so" \
  --out ./replay-demo-results --warmups 2 --reps 20 \
  --execute-trusted-code
```

The fixture runs addition as an unmeasured prefix and multiplication on an
alias of its output as the measured window. Every warm-up and measured iteration
starts from the same captured allocations, runs the prefix and verifies the final
values. Its timings are CPU host-window timings, **not R9700 performance**.

Result directories must be fresh. Native plugins can execute arbitrary code:
`--execute-trusted-code` is deliberate consent, not a sandbox.

## 4. Capture exact existing kBench invocations

With the optional integration enabled, add `--snapshot-dir` to a normal kBench
correctness run, for example:

```bash
./build/bin/rad-kbench --fixture ./workload.rkb \
  --snapshot-dir ./snapshots-run1
```

Use the same home/plugin/options that make your existing kBench run valid.
`RAD_REPLAY_CAPTURE_DIR` is the equivalent environment variable.

Each complete package is under `case-XXXXXXXX/manifest.json`; `capture.jsonl`
records captures and explicit omissions. Defaults are at most 100 captured
cases and 256 MiB of declared operand/scratch payload per candidate. Override
with `RAD_REPLAY_CAPTURE_MAX_CASES` and `RAD_REPLAY_CAPTURE_MAX_BYTES` explicitly.
Do not treat a cap, unsupported WTAB or skipped stateful row as tested coverage.

**Exact bytes do not imply production provenance.** Seeded kBench recordings are
labeled `kbench-seeded`; model-backed kBench cases are `kbench-model-weights`.
Neither is automatically a serving-engine activation capture. Saved output is
labeled `captured-baseline`, not an independent mathematical oracle. Original
kBench reference verdicts must be retained separately. Capture performs extra
copies/synchronization; do not use capture-run timings as production measurements.

See `docs/REPLAY_FORMAT.md` to author larger operation windows with the Python SDK,
including explicit pointer relocations and independently computed expected data.

## 5. GPU replay on the R9700 (target validation required)

```bash
# Scan explicit final plugin binaries using the existing strict ISA gate.
bash tools/rdna4-isa-suite/run.sh ./selected-final-plugins . ./isa-results-v03

bash tools/rdna4-isa-suite/replay.sh run ./snapshots-run1/case-00000001/manifest.json \
  --worker ./build/bin/rad-kbench-replay \
  --plugin p0=/absolute/path/to/the-exact-captured-plugin.so \
  --isa-manifest ./isa-results-v03/evidence/manifest.json \
  --isa-report ./isa-results-v03/analysis/report.json \
  --out ./gpu-replay-run1 --device 0 --gate --warmups 3 --reps 30 \
  --timeout 120 --execute-trusted-code
```

Match **every** fixture plugin ID to the exact hashed binary. GPU execution
requires a strict static pass bound to those plugin binaries and their embedded
gfx1201 objects. This binds submitted plugin evidence; it does not prove the
launcher's actual dispatch sequence or automatically cover runtime support
kernels. Audit the executable/support objects separately for your deployment.
Generic/unsupported object paths remain explicit blockers in the retained gate.

`--gate` is single-stream AQL only. Without it the runner measures a host-paced
queue window, including submission gaps. See `docs/MEASUREMENT.md`; neither mode
is an exact replay of Radiance's optimized internal tape. An explicit
`--exploratory-without-isa` run may execute trusted GPU code without that gate but
**can never produce a clean overall pass**.

## 6. Test a rebuilt candidate without silently changing evidence

```bash
bash tools/rdna4-isa-suite/replay.sh candidate ./replay-demo/manifest.json \
  --plugin p0=/absolute/path/to/rebuilt/plugin.so --out ./replay-candidate
```

This preserves input/expected bytes, records the parent fixture and old/new
plugin hashes, and produces a new candidate identity. It does not bless the new
binary, change tolerances or update expected output to hide a regression. GPU
candidates need fresh corresponding ISA evidence.

## 7. Make missing cases visible

Place the expected-corpus JSON alongside/above its fixture directories:

```bash
bash tools/rdna4-isa-suite/replay.sh expect ./replay-demo/manifest.json \
  --scope 'Approved host add/mul fixture only' --out ./replay-expected.json

bash tools/rdna4-isa-suite/replay.sh coverage --expect ./replay-expected.json \
  --report ./replay-demo-results/report.json --out ./replay-coverage
```

Add all required manifests and reports explicitly. A missing case, stale fixture,
unexpected result or duplicate report blocks corpus success. No percentage over
this corpus is labeled "all Radiance kernels". Record an independently reviewed
workload inventory before making coverage claims.

## Results and exits

Each run writes `report.json`, `report.md`, `coverage.json`, `junit.xml`,
`worker-result.json`, `progress.jsonl`, stdout/stderr and a private staged plan.
JSON records fixture/worker/plugin hashes, provenance, backend, device/runtime
identity where available, measurement mode, every sample and checked iterations.

| Exit | Runtime meaning |
|---|---|
| 0 | `RUNTIME_CHECKS_PASS` for the declared fixture, or `CORPUS_CHECKS_PASS` for the explicit corpus. |
| 1 | Observed comparison/guard failure. |
| 2 | `INCOMPLETE`: malformed/unsupported input, absent evidence, backend mismatch, error or timeout. |

A killed process is not a GPU reset. Stop further GPU tests after a device fault
or timeout, preserve kernel logs and inspect device health. No retries, reset,
watchdog changes, overclock changes or deliberately invalid GPU launches are
performed automatically.

## Still not delivered

Exact raw AQL tape export/replay; TP2 collective/P2P protocols; lifecycle-managed
instances/global hidden state; automatic WTAB target discovery; automatic real
inference capture; state evolution across independent repetitions; instruction-
level trace equivalence; exhaustive template/kernel coverage; invalid-read
sanitization; hardware-counter profiling; calibrated engine latency prediction.
Do not describe this version as complete production equivalence or wedge safety.
