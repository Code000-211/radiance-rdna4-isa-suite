# Radiance RDNA4 ISA Suite — v0.2.0

A CPU-only **binary static-check gate** for `gfx1201` AMDHSA code objects. This
replaces the v0.1 text/regex prototype. It does not execute GPU kernels, change
GPU settings, patch kernel bodies, or promise freedom from GPU wedges.

**Read `docs/REVIEW.md` and `docs/RULE_COVERAGE.md` before merging.**

## What this stage delivers

* Independent ELF/AMDHSA kernel inventory, descriptor checks and byte coverage.
  Kernel identity is **code-object SHA-256 + descriptor symbol**, not name alone.
* Extraction from final GPU ELF, host ELF `.hip_fatbin`, uncompressed/concatenated
  clang offload bundles, and regular GNU/BSD archives. Identical objects are
  deduplicated while retaining their discovered origins.
* LLVM-backed decoding, with an **independent raw-bitfield VOPD decoder** and
  checks for the documented GFX12 opcode, wave-size, bank/port, source, register,
  scalar-bus and X-to-Y dependency constraints.
* Direct control-flow edges, reachable code-end checks, VGPR-release misuse,
  selected WMMA/PERMLANE hazards, and bounded memory/SGPR diagnostic patterns.
* JSON, Markdown and JUnit reports; explicit incomplete-evidence failures.
* An optional, separately reviewed Radiance **AQL resolution capture patch**.
  It captures code objects and resolved kernel names, not all dispatches.

## Installation

The ZIP is a repository overlay containing `tools/rdna4-isa-suite/`.
Back up an existing v0.1 directory before extracting; do not merge stale Python
modules or old test files into the new directory. No original Radiance files
are changed simply by extracting the archive. The AQL patch is opt-in.

Runtime requirements: Python **3.10+** and an AMDGPU-capable LLVM shared library
or `llvm-objdump` that supports `gfx1201`. Python packages are not required.
The full self-test also requires a Linux C++17 compiler and pthread support.
`llvmlite` is needed **only to regenerate** the already supplied fixture.

## First run: test the suite itself

From the Radiance root:

```bash
bash tools/rdna4-isa-suite/check.sh
```

This requires the decoder and host compiler instead of silently skipping their
tests. It does not require a GPU. The release evidence documents the exact tools
used and the test result. Plain `python3 -m unittest discover -s tests -v` inside
the suite may skip external-tool tests when those tools are unavailable; such a
run is **not equivalent** to the required release/CI run.

## Audit an existing build

Compatibility wrapper, run from the Radiance root:

```bash
bash tools/rdna4-isa-suite/run.sh ./build . ./isa-results-v02
```

The output path must not already exist. The source-root argument is retained for
v0.1 invocation compatibility; source text is **not** treated as proof of ISA
legality. Results are under:

```text
isa-results-v02/evidence/manifest.json
isa-results-v02/evidence/objects/<sha256>.hsaco
isa-results-v02/analysis/report.json
isa-results-v02/analysis/report.md
isa-results-v02/analysis/coverage.json
isa-results-v02/analysis/junit.xml
isa-results-v02/analysis/disassembly/<object-hash>-<symbol-hash>.s
```

For a controlled PR gate, select **final linked artifacts** or a directory of
final executables/plugins rather than every intermediate compiler output.
Relocatable device `.o` files are reported as incomplete because their entry
relocations and final executable mapping have not been resolved. Host `.hip.o`
files containing final embedded device code are supported.

Detailed interface:

```bash
export PYTHONPATH="$PWD/tools/rdna4-isa-suite${PYTHONPATH:+:$PYTHONPATH}"
python3 -m radisa collect ./build --out ./isa-evidence-v02
python3 -m radisa audit ./isa-evidence-v02/manifest.json \
  --out ./isa-analysis-v02 --strict-review
```

A collection exit code of 2 is a failure, not a reason to ignore that stage.
If a manifest was written, auditing it will preserve its collection errors and
can provide a fuller diagnostic report. The wrapper does this automatically.

To select a specific decoder, use **one** of:

```bash
python3 -m radisa audit ./isa-evidence-v02/manifest.json \
  --out ./isa-analysis-native --llvm-library /opt/rocm/llvm/lib/libLLVM.so

python3 -m radisa audit ./isa-evidence-v02/manifest.json \
  --out ./isa-analysis-objdump --objdump /opt/rocm/llvm/bin/llvm-objdump
```

Paths above are examples; supply the actual installed tool path. Auto-discovery
first checks ROCm locations, then LLVM 24 through 19 shared libraries, then
`llvm-objdump`. The wrapper also accepts `RAD_ISA_LLVM_LIBRARY` or
`RAD_ISA_OBJDUMP`. LLVM library decoding runs in a separate, timed process;
decoder failure, stderr diagnostics and timeouts do not become a successful scan.
The objdump backend is tested with output fixtures and negative fake tools;
the end-to-end real decoder exercised in this release is the LLVM shared library.

## Exit codes and meaning

| Code | Status | Meaning |
|---|---|---|
| 0 | `STATIC_CHECKS_PASS` | Evidence is complete for the submitted artifact inventory and no implemented blocking check found a problem. **Not full ISA certification.** |
| 1 | `FAIL` | An implemented check found an error, for example an invalid raw VOPD bank combination. Other incomplete findings may also be present. |
| 2 | `INCOMPLETE` | Missing/ambiguous evidence, unsupported control flow or container, decoder failure, or unresolved review finding. |

**Strict review is the default.** `--allow-review` is an exploratory option that
makes `review` diagnostics non-blocking; it cannot override missing evidence,
unknown decode, errors, or an unresolved indirect CFG. Performance observations
such as a possible waterfall are `info`, not an ISA failure.

Zero kernels, missing objects, changed hashes, omitted metadata records, unlisted
object files in the evidence directory and stale output directories are not
accepted. Symlink files are resolved; symlink directories are not silently
traversed or skipped. Select their final artifact files explicitly. `.git` and
`__pycache__` are outside recursive discovery scope.

## Freeze an independently reviewed inventory

```bash
python3 -m radisa baseline ./isa-evidence-v02/manifest.json \
  --out ./isa-approved-inventory.json
python3 -m radisa audit ./isa-evidence-v02/manifest.json \
  --expect ./isa-approved-inventory.json --out ./isa-approved-check
```

Review the inputs and expected kernel list **before approving the baseline**.
Generating a baseline from the same incomplete set of input paths does not prove
that every application kernel was included. The baseline pins binary hashes, so
an intentional rebuild changes it and requires a reviewed refresh. There is no
cryptographic attestation of an adversarially modified manifest and baseline.

## Runtime-observed AQL kernels

See `integration/README.md`. The patch is specific to the uploaded Radiance
snapshot and is not applied automatically. It snapshots each successfully loaded
AQL code image, then records each first resolved `Func`/device pair with the
capture-time SHA-256. This can close the gap between selected build artifacts
and the objects actually resolved by the tested workload.

It does not establish exhaustive JIT/template/configuration coverage. The
forwarded HIP backend, kernels never resolved by the workload, other processes,
other builds and indirect device-call semantics remain outside that claim.

## Boundaries that must remain explicit

This is not a complete interpreter or formal verifier of all 707 PDF pages.
General interprocedural memory hazards, pointer bounds, P2P visibility, barriers
across workgroups/devices, matrix layout correctness, numerical parity, resource
forward progress and driver/firmware reset behavior are **not proved**.

In particular, the earlier W4A8 `Ascale[m]` tail concern cannot be certified safe
by this ISA scanner alone. It needs argument-shape, optimized-code and runtime
memory-safety tests. No current Radiance GPU code objects were supplied here,
and no GPU was used for release testing. All supplied negative fixtures are
static-test inputs and **must never be launched on a GPU**.

The software is reviewable engineering tooling, not a guarantee that nothing
can go wrong. Keep numerical and GPU stress tests as independent merge gates.
