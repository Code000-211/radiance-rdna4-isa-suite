"""SYS_STORE_WAIT (opt-in, unconfirmed): the waits LLVM inserts before system-scope stores on GFX12.0.

Normative basis, LLVM only: SIMemoryLegalizer.cpp, SIGfx12CacheControl::finalizeStore calls
insertWaitsBeforeSystemScopeStore for a store that is not atomic, not a read-modify-write and has
cache policy scope SCOPE_SYS, when the subtarget has FeatureWaitsBeforeSystemScopeStores
("Target requires waits for loads and atomics before system scope stores", GCNFeatures.td; part
of FeatureISAVersion12, which gfx1201 uses). Those waits are S_WAIT_LOADCNT 0,
S_WAIT_SAMPLECNT 0, S_WAIT_BVHCNT 0, S_WAIT_KMCNT 0 and S_WAIT_STORECNT 0. The AMD RDNA4 ISA
guide does not state this requirement.

The literal reading checked here: at such a store, nothing may be outstanding on LOADcnt,
SAMPLEcnt, BVHcnt, KMcnt or STOREcnt on any path (the `waits.solve` may-state). Compiler output
does not meet it: LLVM inserts these waits as "soft" waits, which SIInsertWaitcnts may merge,
relax or drop, so the code it emits can leave a GLOBAL_INV or an earlier store outstanding at
such a store. What the hardware needs is not documented, so the rule is off by default, reports
for review only, and every finding says it is unconfirmed.

Not distinguishable in machine code: an atomic store (`store atomic`) is encoded like a plain
store, and LLVM does not insert these waits for it (its release ordering may insert others); such
a store is reported as well.
"""
from __future__ import annotations

from pathlib import Path
import re

from . import waits

RULE = 'SYS_STORE_WAIT'
SOURCE = ('LLVM SIMemoryLegalizer.cpp SIGfx12CacheControl::finalizeStore / '
          'insertWaitsBeforeSystemScopeStore; GCNFeatures.td FeatureWaitsBeforeSystemScopeStores '
          '(FeatureISAVersion12). Not in the AMD RDNA4 ISA guide')
COUNTERS = ('load', 'sample', 'bvh', 'km', 'store')
_STORE = re.compile(r'^(?:global|flat|scratch|buffer)_store_|^image_store')
_SYS = re.compile(r'\bscope:SCOPE_SYS\b')


def enabled(options: dict | None) -> set[str]:
    """Rules switched on by `options['enable']`: a set/list of names or a comma-separated string."""
    raw = (options or {}).get('enable') or set()
    if isinstance(raw, str):
        raw = raw.split(',')
    return {str(r).strip().upper() for r in raw if str(r).strip()}


def system_scope_store(ins: dict) -> bool:
    """A store with scope:SCOPE_SYS that is not a read-modify-write (atomics are named *_atomic_*)."""
    return bool(_STORE.match(ins['op'])) and bool(_SYS.search(ins['args']))


def kernel_pass(ctx: dict) -> list[dict]:
    """SYS_STORE_WAIT findings for one kernel (or callee); [] unless enabled in ctx['options']."""
    if RULE not in enabled(ctx.get('options')):
        return []
    ins = ctx['instructions']
    stores = [i for i in ins if system_scope_store(i) and i['pc'] in ctx['reachable']]
    if not stores:
        return []
    state_in, _, _ = waits.solve(ins, ctx['edges'], ctx['table'], ctx['wave'], ctx.get('models'),
                                 bool(ctx['kernel'].get('callee')))
    by_pc = {i['pc']: i for i in ins}
    origin = ctx.get('origin') or {}
    out = []
    for i in stores:
        st = state_in.get(i['pc'], {})
        pending = {d: sorted(st[d]) for d in COUNTERS if st.get(d)}
        if not pending:
            continue

        def where(p):
            if p != waits.ENTRY and origin.get(p, {}).get('line'):
                return f"{Path(origin[p]['file']).name}:{origin[p]['line']}"
            return 'issued before the call' if p == waits.ENTRY else f'pc {p:#x}'
        what = '; '.join(f"{waits.WAIT_NAME[d].removeprefix('s_wait_')}: " +
                         ', '.join(f'{waits.producer_op(p, by_pc)} ({where(p)})' for p in ps)
                         for d, ps in pending.items())
        need = [f'{waits.WAIT_NAME[d]} 0' for d in pending]
        out.append({'rule': RULE, 'severity': 'review', 'pc': i['pc'], 'unconfirmed': True,
                    'message': f"{i['op']} with scope:SCOPE_SYS is reached while memory operations may be "
                               f'outstanding ({what}). LLVM inserts {", ".join(need)} before a non-atomic '
                               'system-scope store on GFX12.0 (as soft waits that its wait pass may relax; '
                               'compiler output does not always keep them). What the hardware requires here '
                               'is not documented by AMD: unconfirmed, opt-in rule. An atomic store is '
                               'encoded the same way and does not need these waits',
                    'required_wait': need,
                    'pending': {d: [None if p == waits.ENTRY else p for p in ps] for d, ps in pending.items()},
                    'source': SOURCE})
    return out
