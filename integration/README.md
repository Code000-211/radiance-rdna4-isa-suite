# Optional Radiance AQL capture integration

**Status:** patch application checked against the uploaded snapshot; the standalone
host header compiled and tested. Full Radiance/ROCm compilation and GPU execution
are not yet validated. Do not enable by default in production.

## Why this hook

The reviewed snapshot has its own AQL/HSA loader in `core/device/aql.cpp`.
`module_load` receives the selected raw code object, and `resolve` obtains the
kernel symbol. The patch observes those existing operations. It does not replace
HSA APIs, inject an LD_PRELOAD library, alter device code, or change queue fences.

The two capture points are:

1. After successful executable load/freeze: persist the exact raw code object.
2. Before publishing a resolved `KernelRec`: record the kernel name, device ordinal,
   snapshot filename, size, process ID and capture-time SHA-256.

The directory must already exist, be owned by the process user, and not be group
or world writable. File creation is exclusive, with mode 0600. A fresh directory
avoids accidental reuse across runs. File writes and resolution records are
serialized. Hashing and fsync add startup/debug overhead and are not performance
benchmark settings. Do not change the environment variable while the process runs.

## Applying the patch

From the Radiance repository root, after reviewing the patch:

```bash
git apply --check tools/rdna4-isa-suite/integration/aql-capture.patch
git apply tools/rdna4-isa-suite/integration/aql-capture.patch
```

Rebuild using the repository's existing HIP/HSA build procedure. Do not assume a
successful standalone header test substitutes for that build. The original
`core/device/aql.cpp` hash is recorded in `SNAPSHOT_SHA256.txt`. If the patch no
longer applies cleanly, reconcile the loader and resolver integration manually;
do not force it onto a different source layout.

Without `RAD_ISA_CAPTURE_DIR`, the hook returns before allocating capture state
or opening files. Enabled failures propagate as explicit loader/resolver errors.
This can stop a debug workload if disk space, permissions or filesystem writes
fail; it is intentional, rather than pretending capture succeeded.

## Collecting a workload

Create a fresh directory, then run the normal workload with the environment set:

```bash
mkdir -m 700 ./isa-capture-run1
export RAD_ISA_CAPTURE_DIR="$PWD/isa-capture-run1"
# Run your usual Radiance executable/workload here with this environment.
# Do not launch deliberately invalid fixtures on the GPU.
unset RAD_ISA_CAPTURE_DIR
```

The capture directory contains `code-<pid>-<serial>.hsaco` files and
`resolved-<pid>.jsonl`. Filenames are not kernel-derived. The JSON log binds each
resolution to the exact snapshot hash. Duplicate code objects across ranks are
content-deduplicated by the offline collector.

From the repository root, collect the capture directory:

```bash
export PYTHONPATH="$PWD/tools/rdna4-isa-suite${PYTHONPATH:+:$PYTHONPATH}"
python3 -m radisa collect ./isa-capture-run1 --out ./isa-runtime-evidence
```

Pass the **actual** PID log path to audit (replace `12345`):

```bash
python3 -m radisa audit ./isa-runtime-evidence/manifest.json \
  --capture-log ./isa-capture-run1/resolved-12345.jsonl \
  --out ./isa-runtime-analysis-12345
```

For multiple processes, run an audit against each PID log with separate report
output directories. Every expected process/log must be accounted for by the test
harness. This interface accepts one log per audit; it does not silently treat one
rank's log as evidence for all ranks. All objects in the supplied manifest are
still audited on each run.

## Exact coverage limits

The log is a **first-resolution log**, not a dispatch trace. It cannot establish
that a resolved kernel ran, how often it ran, or that the process finished. There
is no completion seal; `process_completion_proved` stays false. A wedge can leave
an otherwise well-formed log containing only the resolutions reached before the
failure. Archive application exit status and host logs separately.

The scanner inventories **all metadata kernels in captured objects**, not only
the resolved symbols. But kernels from a different uninstrumented process,
forwarded HIP streams, different workloads/configurations, never-loaded objects
or later JIT activity are not proved covered. Fork after capture initialization
is rejected before acquiring an inherited mutex; do not use it as a supported
capture workflow. Start fresh worker processes instead.

## Rolling back

Disable capture by unsetting `RAD_ISA_CAPTURE_DIR`. To remove the optional source
integration, review and reverse the patch:

```bash
git apply --check -R tools/rdna4-isa-suite/integration/aql-capture.patch
git apply -R tools/rdna4-isa-suite/integration/aql-capture.patch
```

No kernel or GPU setting is modified by these offline audit tools.

## v0.4 ordinary-dispatch observation (optional, diagnostic-only)

Apply `aql-capture.patch` first, then `aql-dispatch-trace.patch`. Check both with
`git apply --check` against the intended checkout. The second patch adds a versioned
weak C callback at the ordinary AQL dispatcher and a hash to resolved kernel records.
With a diagnostic observer enabled, every identity is checked **before** queue
reservation. Unknown identities/refusal therefore cannot leave a reserved packet
unpublished. Hook state is per worker process; unscoped background submitter threads
are refused, not attributed to an arbitrary node.

The worker requests tracing only through `--trace-dispatches --diagnostic-only`.
Its strict ISA manifest must include the plugin AND runtime helper code objects.
A missing hook, missing kernel-node observation, stale/partial trace or nonwhitelisted
helper prevents a clean diagnostic result. The observer is not a general HIP
interposer, raw-tape replay exporter, SDMA tracer or multi-rank capture system.

Sequential patch applicability and the standalone host callback were tested.
Full HIP/HSA compilation, disabled-path overhead, enabled target execution and
runtime helper coverage remain target-host checks. Do not present these source and
host tests as successful AQL dispatches on an R9700.
