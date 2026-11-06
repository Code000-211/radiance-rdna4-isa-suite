# Portable snapshot contract (`radiance-kernel-replay/1`)

This is a **declared** fixture, not a pointer-discovery engine or proof of plugin
behavior. All standard fields/variants are validated before native code is loaded.
Unknown fields, duplicate JSON keys, nonfinite JSON numbers, unresolved references
and unsupported execution modes are rejected. Python validation does not need a
GPU, native plugin loading or a model.

## Identity and provenance

A manifest identifies a case, target backend/architecture/rank count and the ABI the
plugin binaries report by SHA-256. The ABI itself is read from the checkout's
`abi/rad_abi.h` (`radreplay.format.radiance_abi`), never frozen in this suite. The exact manifest hash binds each result to its input.
A plugin is supplied on the command line by ID; no executable path or shell
command is accepted from the manifest.

Input provenance is one of `synthetic`, `kbench-seeded`, `kbench-model-weights`,
`production-snapshot`. This is author-declared metadata; the latter must only be
used when actual inference inputs were recorded. The optional kBench hook does
not produce that label. Expected-output provenance is `captured-baseline` or
`independent`; an old implementation's output is not a mathematical oracle.

A rebuilt plugin requires `replay.sh candidate`, recording parent manifest and
old/new hashes. Inputs, expected output and comparison tolerances are preserved.
This is explicit rebinding, not a claim of equivalent code or numeric approval.

## Allocations, tensor views and relocations

Each allocation has an ID, byte size, memory kind, alignment, mutable flag and
full initial blob. Content-addressed blobs have byte counts and SHA-256 hashes.
The runner restores every allocation before every iteration, including outputs;
per-node scratch starts at zero. There is no silently uninitialized input.

Views refer to allocation ID + offset, with dtype, shape, stride and an explicit
byte span. Aliasing is represented by multiple views into the same allocation.
Sizes use the supplied ABI's dtype widths and row-byte padding for sub-byte
formats. Plugin-private dtypes require an explicit opaque span; their semantic
layout is **not** validated. Negative strides and unsupported packed layouts are
refused rather than silently reinterpreted.

A pointer table must declare every eight-byte slot explicitly:

```python
p.allocation('weights', weight_bytes)
p.allocation('ptrs', bytes(8), relocations=[
    {'offset': 0, 'target': 'weights', 'addend': 0},
])
```

Portable pointer slots contain zeros. After allocating all buffers the runner
writes new addresses using these relocation records. Nested targets must appear
in each consuming operation's access contract. The SDK never guesses pointers
from 64-bit values. Mutable pointer tables, one-past-end targets and missing
relocations are unsupported. The `pointer_fields_complete` assertion is the
fixture author's responsibility; the tool cannot prove arbitrary plugin structs
contain no hidden pointers.

Automatic kBench export merges overlapping **known operand ranges**; it does not
have a general allocator history. It preserves address residue up to 256 bytes
with prefix padding, not exact virtual addresses or physical cache-set placement.
Automatic WTAB export is refused because the caller has not supplied a full
inventory of the pointee allocations. Manually authored packages can use the
explicit relocation facility above. No remote/process-owned pointer is portable.

## Nodes and dependencies

Supported nodes: `kernel`, `copy`, `fill`, `record`, `wait`. Every node names a
stream and phase (`prefix` or `measure`). Prefix nodes come first; at least one
kernel must belong to the measured window. Events have exactly one producer per
iteration; waits require an earlier producer inside the fixture. Stream FIFO and
explicit event ancestry are used to reject unordered overlapping read/write or
write/write accesses. This is a conservative declared-access analysis, not a
GPU race detector.

Kernel nodes freeze the row name, operation name, typed parameters, tensor views,
access modes and scratch bytes. The worker selects that **exact row**, checks its
constraints, validates op-schema geometry and row-local tunable axes separately,
requires every tunable to be explicitly frozen, checks cross-axis legality and
refuses insufficient scratch. There is no silent change to another implementation.

Single-rank, stateless rows are required. Lifecycle hooks, collectives, link-release
protocols, `chain != 0` and hidden state are refused. A GPU row must make Radiance's
concurrent/stateless promise. Single-stream host-reference rows are the documented
exception because libref does not export that promise. The promise is still an
ABI declaration, not independent validation of a plugin's source.

## Checks

Every declared writable view must be completely covered by expected-output
checks. Overlapping checks are allowed only as distinct, nonduplicate views.
Checks are either byte-exact or packed f32/f16/bf16 with explicit elementwise
`atol`, `rtol`, relative-L2 bound and nonfinite policy. Numeric mode rejects
nonfinite results by default; explicit `match` still distinguishes NaN from
infinity and the sign of infinity. Byte mode is a bit-pattern comparison, not a
finiteness test. Pointer slot bytes cannot be compared against stale addresses.

Every warm-up and timed iteration is checked after completion. Guards on both
sides of allocations and scratch, immutable allocation contents and bytes outside
declared writable ranges are checked. They detect **observed writes**, not invalid
reads, writes repaired before checking, underdeclared internal scratch use or
logical output errors outside the oracle's capabilities.

## Python SDK

```python
from pathlib import Path
from radreplay.builder import Package

p = Package(Path('new-case'), 'case-name', backend='host',
            provenance='synthetic', reference='independent', note='Describe oracle')
p.plugin('actual-plugin-name', Path('/path/to/plugin.so'))
p.stream('q0')
p.allocation('input', input_bytes)
p.allocation('output', bytes(len(expected_bytes)), mutable=True)
p.view('x', 'input', nbytes=len(input_bytes), dtype=9, shape=[len(input_bytes)])
p.view('y', 'output', nbytes=len(expected_bytes), dtype=9, shape=[len(expected_bytes)])
p.kernel('op0', 'p0', 'actual-row', 'actual-op', ['x', 'y'],
         [('x', 'r'), ('y', 'w')], params={})
p.check('y', expected_bytes)
p.finish()
```

Replace row, operand order, sizes/dtypes and parameters with the actual plugin
schema. The executable `examples/libref_chain.py` is a complete, tested example;
the snippet above describes the API and is not a generic valid Radiance op.

## Resource and trust boundaries

Default payload/scratch limit: 2 GiB; default maximum nodes: 2048. C++ execution
also counts physical allocation alignment and redzone overhead, so a case at the
payload limit may be refused with a memory-budget error. Adjust budgets explicitly;
no out-of-memory or unsupported case is counted as passed coverage.

Blob paths must remain within the package and may not use symlinks. Verified
blobs are staged privately; the child rechecks their hashes. Native plugins and
the worker are trusted user-selected programs, not sandboxed by this process.
Dynamic shared-library dependencies and privileged device behavior are not
cryptographically attested. Keep snapshots private: activations, weights and
other captured buffers may contain sensitive or licensed data.

## v0.4 compatible extensions (schema remains `/1`)

Kernel nodes may declare `scratch_policy: "zeroed" | "kernel-initialized" |
"unspecified"`. Missing/unspecified is a legacy parseable value, **not** permission
to assume zeroed scratch. For nonzero scratch it blocks contract diagnostics.
The first is a caller initialization requirement; the second requires the kernel
to establish everything it later reads. This release tests three starting patterns,
not every possible uninitialized-read execution.

Optional top-level `checkpoints`:

```json
"checkpoints": [
  {"after": "operation-id", "checks": [
    {"view": "view-id", "expected": {"path": "blobs/HASH", "sha256": "HASH", "bytes": 4},
     "mode": "float", "atol": 0.00002, "rtol": 0.00002, "rel_l2": 0.00002,
     "nonfinite": "reject"}
  ]}
]
```

`HASH` above is a placeholder, not a valid manifest value. The same typed check
rules and exact blob sizes apply as for final checks. Node ids and per-checkpoint
views must be unique. Checkpoints are checked only in contract diagnostics, which
serialize operations. They do not substitute for required complete final output
checks. The construction SDK is `Package.checkpoint(after, view, expected, ...)`.

Checkpoint-declared numeric tolerance governs its ranges. Remaining written bytes
are compared exactly against a same-candidate baseline across perturbation passes.
Custom encodings, padded/strided pure outputs and readable aliases are not blindly
poisoned; counts and limits remain in the report.
