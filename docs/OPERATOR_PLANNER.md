# v0.6 operator model: equations and evidence boundaries

All service quantities are bytes or operations. Resource capacities are units per
microsecond; time is microseconds. Supplied system rates are explicit assumptions
unless separately measured. Merely writing `external-measured-profile` is a local
provenance declaration, not verification. It never produces automatic approval.

## Attention / QSA

A query is described by its sequence ID and **visible causal prefix length**, not
just the final context length. Sparse block lists are supplied per query. A block
j contributes min(block_size, prefix_length-j*block_size) live tokens. Invalid,
duplicate or future blocks are rejected. Useful matrix work is:

    4 * head_dim * q_heads * sum(live_selected_tokens_per_query)

QK and PV each contribute 2*head_dim per query/key pair. A declared WMMA work
scenario pads query/head and key tiles; it does not divide by tile efficiency a
second time. Dense query tiles may reuse the union of causal prefixes; differing
sparse selections are modeled separately. This is conservative for paths that
can exploit more sharing, not proof that no sharing is possible.

GQA reuse is explicit. KV traffic uses kv_heads with reuse and q_heads without it.
FP8 KV is a storage choice; the example default arithmetic path is WMMA F16 after
conversion, NOT an automatic FP8 arithmetic-rate assumption. Source-specific
partial precision is 16 bits in the examples. Split count is clamped to nonempty
tiles for the hypothetical work plan. The actual source dispatch may choose a
different fixed split count and needs a reviewed launcher binding.

Split partial storage is charged as output of the main phase and input of merge:

    sum(effective_splits) * q_heads * (head_dim * partial_bytes + 8)

No full query-by-history score matrix is allocated by the attention model. Local
softmax vectors/exp operations and merge work are explicit service demands. GPR/LDS
placement, dequantization instruction mix, spill traffic and exact wave scheduling
are NOT inferred for these new candidates; v0.5's compiled GEMM resource path
remains a separate capability.

The indexer is a **separate operator**: it scores complete historical blocks,
not just the eventual selected KV. For each query L, history_blocks=floor(L/ratio).
Score work is 2*heads*indexer_head_dim*history_blocks. The retained score plane is
FP32; selection traffic and radix/scan work are explicit hypotheses. Block-key
updates can still occur when scoring/selection is skipped in a dense fallback.
Projection, norm, RoPE and speculative tail state require other nodes.

The partial model importer enforces the supplied source's top complete-block
budget and inclusion of the incomplete tail. The generic attention operator can
represent other explicitly declared selected subsets; it is not itself a QSA
selection-correctness test. Dense fallback is accepted only below the declared
budget + ratio - 1 bound.

## Routed MoE

Inputs are the **actual or explicitly synthetic histogram** n[e], local width f[e],
hidden width H and top-k. The planner requires sum(n[e])=tokens*top_k and
0<=n[e]<=tokens. It never predicts routing from a model name or assumes uniformity.
For a gated FFN, useful routed matrix work is:

    6 * H * sum(n[e] * f[e])

Gate/up and down are independently tiled and padded, with optional explicit work
for empty experts. The source-family example skips empty experts. It aggregates
grouped kernel work; it does not charge one launch per expert. Router GEMM/top-k,
scatter, gate/up, activation/quantization, down, gather and shared expert are
separate cost phases. Their vector/exp counts are declared approximation formulas,
not instruction counts extracted from a compiled binary.

Weight bytes contain packed payload plus declared scale bytes. The cold-weight
fraction applies to tile-requested bytes; cache hits, cache bandwidth and packed
relayout are not independently solved. A zero cold fraction is a warm-cache
scenario, not zero total weight service on real hardware. Shared-expert weight
scales/relayout and hidden state are not fully modeled; do not use this as a precise
quantized byte-accounting oracle.

For supplied W4A8 TP, expert width is split in 128-column blocks. width=640 means
5 blocks: TP2 yields 384/256, rotated by expert parity. It is not a universal
320/320 split. TP4 uses expert modulo four. Shared placement follows the separate
source rotation. These rules are pinned to twelve exact source hashes.
The routed W4A8 source uses FP8 WMMA after INT4 unpacking; the earlier dense W4A8
prefill adapter uses INT8 WMMA. They must not share a rate solely because both are
called W4A8. Other quantization recipes need explicit matching contracts.

## Two-rank communication costs, not a communication implementation

The oneshot hypothesis sends a full message in both directions, then reduces
locally on both ranks. The ring2 hypothesis exchanges halves, reduces, and
allgathers. Both have 2*message_bytes total payload for two ranks, but differ in
reduction work, phase dependencies, workspace and startup count. The real supplied
oneshot source is custom code, not RCCL ring. The ring candidate is an alternative
algorithm hypothesis, NOT a claim it is already a Radiance launcher.

A transfer charges its directional/access link, source and destination memory,
and every declared shared PCIe/root resource. Push and pull require separate
rates. No symmetric speed is inferred. Chunk pipelines preserve directional FIFO.
The cost graph includes two modeled initial launches and per-chunk visibility
latency; it does not emulate flag/epoch/checksum loops, residency, atomics,
cache coherence, rank skew failures, or the exact fused source implementation.
It must not be executed as though it were a validated TP2 replay recipe.

## Dense wrappers / explicit work

A v0.5 dense-GEMM candidate can be wrapped as a fixed-duration task. It reserves
all device kernel slots because its original service breakdown is not reconstructed
for contention modeling. An `explicit` node carries caller-supplied demands; it
is an escape hatch for **declared assumptions**, not a way to claim an unknown
operation has been understood. Neither wrapper silently creates GPU code.

## Sensitivity, search and observations

`plan-operators` ranks conditional scenario times. `plan-graph` searches candidate
assignments with layout transitions and shared resources. `sensitivity` reports
the maximum relative regret across a finite user-supplied profile set. Winner
frequency is not a probability, and scenario ranges are not confidence intervals.

`compare-graph` accepts an external locally trusted observation record bound to
the exact graph report, environment and device set; CPU/instrumented/failed runs
are rejected. The evidence SHA is a provenance identifier, not an automatically
fetched/validated runtime report. This command compares an observation; it does
NOT fit unobserved operators, update profiles, approve deployment, or authenticate
remote execution. New operator numerical/ISA/launcher bindings still need target
work. The inherited dense-GEMM checked importer/fit is unchanged.

`measurement-batch` exports requests with hashes and NOT_EXECUTED state. It cannot
launch unsupported operations automatically; do not replace their state with PASS
merely because this JSON was created.
