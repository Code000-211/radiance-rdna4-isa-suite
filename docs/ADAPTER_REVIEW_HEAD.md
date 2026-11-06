# Adapter review against radiance HEAD (ABI 31)

Date: 2026-10-29. Basis: the v0.6 pin table was taken from the supplied
`radiance-main` snapshot (`RAD_ABI_VERSION 30`). This review re-pins the twelve
files in `radplan/source_ops.py::PINNED` to the current Radiance tree and records
what each drift means for the planner's cost/geometry hypotheses. It is a
source-reading review, **not** new GPU evidence: every cost stays
`NEEDS_MEASUREMENT`.

Ten of the twelve pinned files changed; two attention kernels are byte-identical.

## Conclusion first

No catalog constant had to change. The geometry the catalog encodes is still what
the source declares: `head_dim` 256, GQA 6/12, paged `block_size` 4 (and 16 in the
gqa8 band), 128-element weight scale groups, whole 128-column expert slices,
`moe` BK 128, 16-byte vector alignment and a two-rank one-shot exchange. What
changed is the set of *variants* and *residency facts* around that geometry, so the
catalog stays correct as a scenario space but is now visibly incomplete in six
places, listed per file below as `shortcut:` notes rather than silently widened.

## Per file

`arch/qwen4exp_fp8/qwen4exp_fp8.cpp` (+5): a QSA selection plane (`qsa_nsel`) is
now passed through to attention, and `probe()` declares `kv_stream_ok = 1`. The
catalog's QSA hypothesis has no `nsel`/`seq` plane, and its attention residency
assumes a resident KV group. `shortcut:` the QSA selection/sequence plane and
streamed KV residency are not costed, upgrade when QSA plans must predict
residency or selection-plane traffic.

`arch/common/rad_qsa.h` (+6): the QSA block-key group is now declared with
`rad_kv_group_dense_read`, i.e. every complete history block is read each step and
the group stays resident while a sibling group streams. This *supports* the
catalog's `input_layout: index_keys` (whole-block, per-query scoring) and adds a
residency obligation the planner does not charge. No cost change; documented.

`arch/common/rad_block_moe_fp8.h` (−47/+33): the three F32 scratch buffers
(`moe_f32`, `shared_f32`, `combined_f32`) and their three cast/combine ops are gone,
replaced by one `op_combine` (BF16 in, FP32 add, one BF16 rounding). At TP1 with an
ungated shared arm the gather now folds the shared result, so there is no separate
combine dispatch; at TP>1 each arm is reduced separately and one combine remains.
The planner's MoE model already charged no combine and no F32 scratch, so TP1 is
unchanged and its `workspace_bytes` was never inflated by those buffers.
`shortcut:` one unmodeled combine dispatch for a shared expert at TP>1, upgrade
when the graph model claims TP>1 shared-expert dispatch counts.

`docs/QSA.md` (+5/−2): libref now implements all five QSA ops (`ref_qsa.cpp`), so
each device QSA kernel has a reference row. This does not change a cost; it is the
precondition for numerical comparison work and is recorded as evidence capability,
not as validation of any candidate.

`libr4d/r4d_attn_paged_gqa8.hip` (−8/+13): the split-KV partial buffer is now sized
as a maximum over `q_len` as well as sequence count, because the split law is not
monotone in depth (`r4d_decode_splits`: at `q_len * gqa <= 32` a 256-workgroup
target; above it 128, 512 past 65536 tokens, at least 64 from 16384, capped at
256 and at `tiles/2`). `shortcut:` the catalog searches `split_kv` in {1,4,16}
while the source law reaches 64-256 splits on deep contexts and is
`kv_heads * num_seqs`-dependent; the catalog therefore bounds the search space, not
the implementation. Upgrade when deep-context decode plans are compared to
measurements.

`libr4d/r4d_attn_decode_h256_gqa6.hip` (−21/+98): an optional TREE verify
(`O_TREE`, `spec_parent`, row-dependent visibility, one staged parent plane) is
added, and the merge's `SPAN` comment is reworded. The catalog has no tree-verify
variant. `shortcut:` speculative tree verification (per-row ancestor masks) is not
a modeled attention contract, upgrade only with a declared selection per row.

`libr4d/r4d_attn_prefill_h256_gqa6.hip` (−14/+33): KLATE, a late K prefetch staging
option for the wide K fetch (fp8 and k6_v4 caches), which overlaps a tile's K fetch
with the previous tile's scores. The planner explicitly does not model pipeline
overlap, so its prefill scenario is conservative; no change.

`libr4d/r4d_moe.hip` (−4/+110): a new `moe_w4_rows_kernel` (one sorted row per
block, 64 columns, no retiling/LDS staging) and a new I4 weight load layout
(`moe_w4_load`, 64-bit gathers over the 128-scale fragment tiles). The catalog's
`bm` 16/32 padding scenario therefore no longer describes small per-expert rows.
`shortcut:` no small-row MoE candidate; `bm` padding is an upper bound for
low-occupancy experts. Same 128-element scale group and the same eight WMMAs per
group as the tiled kernels, so the existing scale-group and `bk=128` assumptions
hold. Upgrade when per-expert occupancy is small enough that padding dominates.

`libr4d/r4d_ar_oneshot_2rank_exact.hip` (−16/+29): the oneshot all-reduce now
carries a `stage` argument and addresses the mailbox through its VRAM mirror
(`r4d_archk_md`, `r4d_archk_keep`, `r4d_archk_moved`); the late-push test hook moved
from the environment to `r4d_ar_late_push_pin`. Chunk causality, the two-slot
workspace and the handshake are unchanged. `shortcut:` the per-slot mailbox mirror
and its address translation are not modeled, upgrade only if the mirror becomes a
measurable fraction of a step.

`libr4d/r4d_rows.cpp` (−68/+428): new rows `attn_paged_qrow_tree` (with
`spec_parent`), `dflash_tree` (with `nodes`, `temp`, `parents`) and `kv_resolve`
(the residency map between `qsa_select` and attention); the schema comment that
kept the streamed-store prologue at E4M3-only is gone, so 16-bit caches can stream
too. The registry is what the catalog's "row family" binding refers to, so its
hash matters; the new rows are new operations, not changes to the modeled ones.
`shortcut:` `kv_resolve` and the tree/drafter rows are unmodeled operations, and
16-bit KV streaming changes the residency assumption for the `kv_bits: 16`
attention candidates. Upgrade when those rows enter a measured graph.

`libr4d/r4d_attn_paged_h256_gqa12.hip`, `libr4d/r4d_attn_paged_h256_gqa6_bs4.hip`:
byte-identical to the review basis.

## What this review does not claim

- No instruction-level equivalence, no dispatch mapping, no measured rate.
- The changed numbers in the source are not imported as timings; the catalog's
  rate assumptions stay synthetic until a GPU measurement replaces them.
- Re-pinning authorizes "these are the files the hypotheses were reviewed
  against"; it does not authorize any candidate as calibrated.
