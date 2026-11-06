# v0.6 graph and resource format

Examples in `planner/examples/v06` are executable inputs. All JSON uses the strict
reader (no duplicate keys/NaN), finite quantities, explicit schemas and fresh
output paths. Validation is bounded; large phase/candidate expansions fail rather
than silently truncate. These are **planning contracts**, not a runtime ABI.

## System

`radiance-flow-system/1`: devices, directional links, shared resources, environment.
Each device supplies positive DRAM, matrix-path, vector and exp capacities,
nonnegative launch cost, memory budget and kernel admission slots. Each link
names source, target, push/pull, rate, latency and shared-resource IDs. Rates are
in units per microsecond. `kernel_slots` is a caller assumption, not measured
occupancy; the GPU's registers/LDS/blocks are not dynamically allocated here.

## Workload and candidate

`radiance-operator-workloads/1` wraps explicitly typed `attention`, `qsa_indexer`,
`moe`, `collective`, `gemm` or `explicit` workloads. Candidate parameters and
numerical/layout contracts must match. `radiance-operator-catalog/1` contains finite
cost hypotheses with identities and optional exact source-family hashes. A source
hash or candidate name is NOT a compiled symbol or numerical certification.

## Graph

`radiance-planner-graph/1` contains nodes, dependencies, queues, transitions,
allocation lifetimes, resident bytes and finite-search limits. Every node has a
workload ID and explicit allowed candidates. FIFO edges are added for successive
nodes on the same declared queue. A dependency is either data or control.

A data layout mismatch requires an explicit conversion, which charges declared
read/write bytes and latency. Cross-device data edges require an explicit
collective/transfer; matching layout names do not make remote memory free. Control
edges may cross devices. The planner does not prove tensor shape equivalence or
semantics of user-supplied edges: callers must review those contracts.

## Scheduler assumption

Each phase has dependencies, service demands, optional fixed latency and slot
reservations. Ready phases are admitted in deterministic manifest/topological
order. For a running phase i with demand D[i,r] on resource r and N[r] running
users, the progress rate is:

    min_r(capacity[r] / (N[r]*D[i,r]))

If fixed latency is nonzero, progress is also limited by 1/fixed_latency. Completion
releases slots/bandwidth and allows dependent phases. A shared matrix-issue
resource prevents F16/FP8 matrix paths from each claiming an independent full
matrix engine. Transfer demands contend on shared PCIe-root and device-memory
resources. There is no preemption, exact workgroup placement, cache simulation,
wavefront scoreboard, scheduling-policy search or GPU instruction simulation.

Normalized-progress bottleneck sharing can leave resources idle and is neither a
proven lower nor upper hardware latency bound. The reported `model_floor_us` is
only the resource-volume/critical-path floor conditional on the declared model.
The serialized scenario is a comparison of this model, not a measured baseline.

## Memory

Buffers describe allocations, not tensor views. Name a physical allocation once;
aliases do not get new byte counts. Allocation lives from producer start (or time
zero for inputs) until the last consumer completes. Required dependency paths are
checked. Workspace is added from each operator model. Acquisitions are charged
before releases at equal timestamps, conservatively. Over-budget assignments are
rejected; the planner does not invent RAM spill or remote-memory availability.
Hidden engine state, conversion output allocations, weights, KV pools and allocator
reserves must be declared in buffers/resident_bytes when applicable. An incomplete
caller memory ledger cannot prove a real model fits VRAM.

## Search

Up to the configured combination limit, every candidate assignment is evaluated
under the fixed admission policy. Beyond it, a bounded beam search is used and
`search_complete=false`. Neither mode claims a globally hardware-optimal schedule.
The winning cost hypothesis remains `NEEDS_MEASUREMENT`. Exactly 96 combinations
exist in the supplied two-MoE-plus-collective example (4*4*6).

## Partial model importer

`model-workloads` accepts explicit `qwen4_exp` config fields plus every layer's
routing histogram and every QSA layer's per-query selections/block updates. No
missing dimensions, selections or routing are guessed. Twelve source hashes
must match. TP1/2/4 shape partitioning is supported where whole 128-column blocks
fit; the supplied attention cost catalog specifically covers GQA 6/12. Shape
extraction does not guarantee an executable candidate exists for every result.

The 48-layer example expands to 144 modeled operator occurrences and 120 dense
projection shape requests. It leaves 36 recurrent/GDN layers and additional
normalization/HC/PLE/RoPE/KV-write/drafter/logit/communication work explicitly
unmodeled. Dense requests need actual recipe formats and matching GEMM candidates.
No full graph or model throughput is constructed by adding those missing times
as zero. Actual model weights/config were not fetched or inferred from a name.
