# Planner file contracts and commands

All JSON is UTF-8; duplicate keys, nonfinite numbers and unknown required-schema
fields are rejected. Output files/directories must be new. Inputs are trusted local
analysis data, not cryptographic remote attestation. Catalogs describe code but
ordinary planning never imports/dlopen-executes native plugins.

## Inputs

| Schema | Role |
|---|---|
| `radiance-planner-workloads/1` | Explicit mathematical shapes, format, numerical/layout contract and cache class |
| `radiance-planner-catalog/1` | Candidate algorithms, launch choice, tile/work/traffic model, identities and resources |
| `radiance-planner-profile/1` | Target peaks, separated assumptions and environment fingerprint |
| `radiance-planner-measurements/1` | Retained checked operation-window samples, exact candidate/profile and evidence hashes |
| `radiance-planner-calibration/1` | Exact feature version, models, train/calibration/test split, uncertainty and validation |
| `radiance-offline-plan/1` | Ranking, reasons, resource counts and requested measurements |
| `radiance-planner-path/1` | Explicit serial stages with accepted candidates and conversion costs |

Executable schema enforcement is in `radplan/model.py`, `calibrate.py` and
`planner.py`; runnable examples are under `planner/examples/`. A file ending in
`.json` does not automatically establish that it conforms to these contracts.

## Workloads and candidates

`M/N/K` describe the problem, whereas `tile=[BM,BN,BK]` describes one implementation.
Workload fields include `a_bits`, `b_bits`, `out_bits`, `weight_group`,
`weight_scale_bytes`, `activation_scale_bytes`, `contract` and `cache_state`.
All format fields must match the candidate. `frequency` is optional annotation;
this version does not pretend to predict an entire weighted model execution.

A candidate includes backend/architecture, compute instruction path, tile,
work-policy (`full_tiles` or `exact`), tail support, launcher dimension multiples,
wave/workgroup configuration, static LDS and explicit launch row/op/parameters.
`identity.kind` distinguishes source, plugin, code object and hypothesis; a
source-only candidate cannot produce a GPU calibration without compiled binding.
Optional compiler resources distinguish logical and encoded VGPR requirements,
scratch allocation and provenance. Optional `unmodeled` strings remain visible.

No automatic extraction of an arbitrary model's configuration or all `.rkb`
fixtures is claimed. Supply actual shapes from reviewed model/fixture metadata.
The existing v0.4 portable snapshots remain the runtime input format. The planner
import below reads those snapshots and their reports rather than inventing timing
records from source shapes alone.

## Source-bound Radiance candidate catalog

```bash
bash tools/rdna4-isa-suite/planner.sh catalog-radiance \
  --source "$PWD" --out ./my-source-catalog.json
```

Only the reviewed source hashes are accepted. The adapter extracts all 14 declared
W4A8-prefill arms and cross-checks the registered tunable table against the template
switch and tile map. Different source hashes require review/update or an explicit
catalog. A newer source is not falsely accepted by filename resemblance.

## Compiler-resource binding

First collect/audit final binaries with the existing v0.4 ISA tooling. Create a
mapping object with candidate IDs as keys and these exact fields as values:

```json
{
  "candidate-id": {
    "plugin_sha256": "<64 hexadecimal characters>",
    "object_sha256": "<64 hexadecimal characters>",
    "symbol": "<actual AMDHSA descriptor symbol>"
  }
}
```

The angle-bracket strings above are placeholders, not runnable evidence.

```bash
bash tools/rdna4-isa-suite/planner.sh bind-resources \
  --catalog ./my-source-catalog.json \
  --isa-manifest ./isa-results/evidence/manifest.json \
  --isa-report ./isa-results/analysis/report.json \
  --mapping ./reviewed-symbol-map.json --out ./compiled-catalog.json
```

The importer requires strict passing ISA evidence, verifies object bytes, metadata
and descriptor registers, confirms the mapped object belongs to the named plugin,
and checks wave size. It preserves all unmapped candidates as source-only. It does
not prove that a handwritten map corresponds to the launcher's actual specialization;
use the AQL observer/integration evidence for that association. The included `smoke`
import proof uses an LLVM fixture and is not a compiled Radiance GEMM result.

## Observed environment profile

After a checked uninstrumented replay on the target host:

```bash
bash tools/rdna4-isa-suite/planner.sh profile-from-replay \
  --base tools/rdna4-isa-suite/planner/profiles/r9700-unmeasured.json \
  --report ./target-run/report.json \
  --clock-power 'actual unchanged target settings, recorded by operator' \
  --compiler-id 'exact compiler/build description supplied by operator' \
  --out ./r9700-observed.json
```

This pins observed device/worker/mode identity. Clock/power and compiler descriptions
are explicit operator declarations, not automatically verified telemetry. The command
does not modify clock/power settings or calibrate throughput. Changes require a new
profile and fit; cross-device CPU/GPU conversion is rejected.

## Import and fit checked timings

Create an index array. Paths resolve relative to the index file:

```json
[
  {"workload_id":"actual-case", "candidate_id":"actual-candidate",
   "report":"runs/case/report.json", "fixture":"fixtures/case/manifest.json"}
]
```

```bash
bash tools/rdna4-isa-suite/planner.sh import-replay \
  --catalog ./compiled-catalog.json --profile ./r9700-observed.json \
  --workloads ./actual-workloads.json --index ./timing-index.json \
  --out ./checked-measurements.json
bash tools/rdna4-isa-suite/planner.sh fit \
  --catalog ./compiled-catalog.json --profile ./r9700-observed.json \
  --measurements ./checked-measurements.json \
  --calibration-groups 10 --test-groups 10 --seed 7 \
  --out ./r9700-calibration.json
```

Import verifies portable-fixture hashes, successful native execution, all checked
warmups/repetitions, non-ASan build, required correctness diagnostics, a single
measured operation, actual row/shape/tuning and plugin/worker/device identity.
Multi-operation window time is not assigned to one constituent kernel. GPU imports
additionally require compiled catalog binding and matching static report identity.
Counter-instrumented, serialized diagnostics cannot masquerade as normal timings.

Fit requires a complete candidate panel at each shape and at least eight distinct
training groups in addition to calibration/test groups. The example 10+10 split
therefore needs at least 28 complete shape groups; the CPU demo has 40. Repeated
samples of the same shape are NOT additional shape groups. Import is for genuine
single-operation cases; GPU replay itself has the unchanged v0.4 limitations.

## Planning with calibration

```bash
bash tools/rdna4-isa-suite/planner.sh plan \
  --catalog ./compiled-catalog.json --profile ./r9700-observed.json \
  --workloads ./new-workloads.json --calibration ./r9700-calibration.json \
  --out ./calibrated-plan
```

Changed candidate identity or out-of-range shapes retain diagnostic point estimates
where possible but lose accepted intervals. A missing fit for any rival forces a
scenario-only ranking for the whole comparison. `--require-calibrated` can be used
to make the command exit 2 unless every decision has scoped calibrated separation.
This remains independent of numerical/runtime approval.

## Explicit serial path

```bash
bash tools/rdna4-isa-suite/planner.sh path \
  --plan ./offline-plan-v05/plan.json \
  --spec tools/rdna4-isa-suite/planner/examples/serial-path.json \
  --out ./serial-path-result.json
```

The example is an abstract serial schedule, not a Qwen model graph. Its dimensions,
data connectivity and layout semantics are not automatically validated. Layout
conversions need explicit cost and `measured`/`assumed` provenance. No overlap or
TP2 inference is performed.
