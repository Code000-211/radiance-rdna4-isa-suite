# Working on this repository

Instructions for coding agents. Humans start with [README.md](README.md).

## What this is

Offline tools for AMD `gfx1201` (RDNA4, Radeon R9700) kernels. The main one is `radisa check`,
a linter for kernel machine code. The repository ships as `tools/rdna4-isa-suite/` inside a
Radiance checkout and must never change a Radiance file.

| Path | Contents |
|---|---|
| `radisa/` | The linter and the build audit. Start with `check.py`. |
| `radisa/data/` | `rdna4_isa.json` (AMD's operand table, generated) and `llvm_pins.json`. |
| `tests/`, `fixtures/check/` | Linter tests and their input files. |
| `hwprobe/` | Test kernels that are run on the GPU. `cases/` is generated. |
| `radreplay/`, `replay_tests/`, `integration/` | Kernel replay on the host. |
| `radplan/`, `planner/`, `planner_tests/` | Offline cost planner. |
| `docs/CHECK.md` | The reference for `radisa check`. Read it before changing a rule. |
| `evidence/` | Recorded results of earlier runs. Do not edit. |

Where things are in `radisa/`:

| Module | Job |
|---|---|
| `check.py` | Entry point, report, status and exit code, edit hook. |
| `rules.py` | Control-flow graph (`cfg`), local checks. |
| `waits.py` | Memory counters (`MISSING_WAIT`, `LDS_STORE_BEFORE_BARRIER`). |
| `sgpr.py` | Scalar-register commit hazard (`MISSING_ALU_WAIT`). |
| `hazards.py` | WMMA, permlane, clause and dealloc rules over the whole graph. |
| `liveness.py`, `asmclobber.py`, `memorder.py` | Uninitialised reads, inline-asm clobbers, system-scope stores. |
| `functions.py` | Functions that kernels call. |
| `suppress.py` | The rule registry `RULES`, configuration, inline ignores, baselines. |
| `incremental.py`, `hipsrc.py` | Compiling HIP, the per-kernel cache, source lines. |
| `batch.py`, `formats.py` | `check-all`, SARIF and GitHub output. |
| `drift.py` | Compares LLVM sources with `llvm_pins.json`. |

## Commands

Run everything from the repository root.

```bash
export PYTHONPATH=$PWD

python3 -m unittest discover -s tests                     # the linter tests, about 90 s
python3 -m unittest discover -s tests -p 'test_hazards.py'  # one file
python3 -m radisa check FILE                              # .hip, .s or a code object
python3 -m radisa rules --explain RULE
sha256sum -c SHA256SUMS
```

- Run the full `tests/` suite before you say a change is done. It must end with `OK`.
- `bash check.sh` is the full check for the target toolchain. It sets `RAD_ISA_REQUIRE_LLVM=1`,
  which turns toolchain skips into failures. With LLVM older than 20 it fails on six tests. That
  is expected. Do not weaken it.
- `replay_tests/` and `planner_tests/` need `RADIANCE_SOURCE` to point at a Radiance checkout.
  Without one, say that you did not run them.

## Rules you must follow

**Python.** Standard library only, Python 3.10+. Add no dependency. Keep the existing style:
compact code, typed signatures, a docstring that states what a function is based on.

**Never invent hardware facts.** Every rule and every message cites its basis: a section of
AMD's RDNA4 manual (`isadb.cite(table, 'x.y')`), or an LLVM function and file. If you cannot cite
it, do not claim it. Say "not confirmed on hardware" where that is the case.

**Severity.** A new rule is an `error` only if AMD's manual documents the problem or LLVM's
AMDGPU backend fixes it for gfx1201. Everything else is `review` (printed as a warning) or `info`.
`LDS_STORE_BEFORE_BARRIER` is the one exception, because it was observed on the GPU. An error on
correct compiler output is the worst outcome: the linter blocks edits and commits.

**LLVM facts.** The rules follow ROCm llvm-project `d6f6cb69`, not upstream `main`. The two
differ. Check a claim against the sources of that commit, and run `python3 -m radisa drift
--llvm-src DIR` after changing a rule that restates LLVM code.

**Status.** `PASS` must never cover code that was not analysed. If something could not be
checked, the result is `INCOMPLETE` (exit 3).

## When you change something

| If you | Then also |
|---|---|
| add or rename a rule | add it to `RULES` in `radisa/suppress.py` (a test fails otherwise), document it in `docs/CHECK.md`, add one test that fires and one clean case that does not |
| change what a rule reports | update the rule table in `docs/CHECK.md` and `docs/RULE_COVERAGE.md` |
| change a cached kernel result | bump `SCHEMA` in `radisa/incremental.py` |
| change `hwprobe/generate.py` | run `python3 hwprobe/generate.py`; never edit `hwprobe/cases/` by hand |
| change any shipped file | update its line in `SHA256SUMS`, and add a line for a new file |
| parse numbers from disassembly | use `isadb.immediate`. LLVM 24 prints `lit(0x20)` where older versions print `0x20`. |

`SHA256SUMS` has no generator script. Replace the hash of each file you changed, keep the order,
and check with `sha256sum -c SHA256SUMS`.

Test fixtures in `fixtures/check/` named `clean_*` must produce no error and no warning. A fixture
that needs a newer compiler than LLVM 18 must skip there, not fail, unless
`RAD_ISA_REQUIRE_LLVM=1` is set.

## Measured numbers

`docs/CHECK.md`, `README.md` and `hwprobe/README.md` contain results measured on the R9700 with
AMD clang 24. You cannot reproduce them without that machine.

- Do not change a measured number unless you measured it again. Then give the date, the commit and
  the toolchain.
- Do not write that something was tested on the target if you ran it only with another compiler.
  Say which compiler you used.
- Keep "what was verified" and "what was not" apart, as the existing sections do.

## Things not to do

- Do not skip, delete or loosen a test to make the suite pass.
- Do not edit files under `evidence/`, `radisa/data/rdna4_isa.json` or `hwprobe/cases/` by hand.
- Do not remove a source-hash check in the planner or replay code when a hash changes. A changed
  hash means the adapter needs review.
- Do not put model names, tool names or session links in code, comments or documentation.

## Git

- Commit messages: a short subject that names the part, for example
  `radisa check: resolve long branches`, then what changed and why.
- Say in the commit which toolchain the tests ran with.
- Do not commit caches. `__pycache__/` and `.local/` are ignored.
