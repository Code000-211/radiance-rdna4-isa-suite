# v0.4.0

* Default contract diagnostics (three serial passes) separate from performance.
* Output-only perturbation respecting input/INOUT aliases; explicit scratch policies.
* Per-operation forbidden writes and optional intermediate reference checkpoints.
* Host-ASan read checking with expressibility/instrumentation checks and negative probes.
* Independently enumerated declared-row coverage; versioned numeric-policy qualification.
* 46-case independent finite-f32 GEMM corpus; long explicit host-state sequence test.
* ABBA interleaving and stop-on-first-nonpass corpus execution.
* Optional diagnostic AQL whitelist/trace before queue reservation; target integration unvalidated.
* Genuine-source standalone host build, including libref, and a single validation script.
* Historical evidence moved to `evidence/v03/`; static v0.2 checker unchanged.

Migration: replace the suite directory rather than mixing old Python modules.
Old fixtures still parse, but scratch users need an explicit truthful scratch
contract before default diagnostics can pass. Captured baseline is not independent
reference evidence. GPU read safety, TP2/raw-tape support and production calibration
remain incomplete; see the gap-closure table. Do not treat a version bump as GPU
stability certification.
