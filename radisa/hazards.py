"""gfx1201 hazards that LLVM's AMDGPU backend fixes, checked over the whole direct-branch CFG.

Normative basis (LLVM main, llvm/lib/Target/AMDGPU/):
- GCNHazardRecognizer.cpp `getWaitStatesSince` walks backwards from the consumer through every
  predecessor block until the hazard producer (hazard) or an instruction for which `IsExpired` holds.
  The forward "may" dataflow below (union over predecessors, producer sets killed by the expiring
  instructions) answers the same question: does some path reach the consumer from a producer
  without passing an expiring instruction? It runs on the edges of `rules.cfg`, which, like the MBB
  predecessor lists LLVM walks, give s_setpc_b64 (return, tail call, long branch) no fall-through.
- `fixWMMAHazards`: a WMMA/SWMMAC whose A (src0) or B (src1) overlaps the D (vdst) of an earlier
  WMMA/SWMMAC; on GFX12 also the SWMMAC index (src2); overlap with C is allowed on GFX12+ (the
  hardware stalls, per the comment there). Expires at `SIInstrInfo::isVALU(I, AllowLDSDMA=true)`,
  i.e. any VALU including V_NOP and including LDS-DMA (no gfx1201 instruction in the AMD table is
  LDS DMA). LLVM's fix: one V_NOP_e32 before the consumer. fixWMMAHazards, fixVcmpxPermlaneHazards
  and their helpers are identical in upstream main and ROCm d6f6cb69; the static
  `getWaitStatesSince` skips meta instructions upstream only, which changes nothing here (a meta
  instruction is never a WMMA, VOPC or VALU).
- `fixVcmpxPermlaneHazards` (ST.hasVcmpxPermlaneHazard(): FeatureVcmpxPermlaneHazard is in
  FeatureISAVersion12: GCNFeatures.td upstream, AMDGPU.td at ROCm d6f6cb69): a V_PERMLANE* after a VOPC/VOP3 compare that writes EXEC
  (`isVCmpXWritesExec`). Expires at a VALU that is not V_NOP (`isVALU(MI, true)` and not
  V_NOP_e32/_e64/_sdwa); SALU, s_nop and v_nop keep it. LLVM's fix: `v_mov_b32 vN, vN` with vN the
  permlane's src0 ("V_NOP will be discarded by SQ").
- SIInsertWaitcnts.cpp, end-of-program VGPR deallocation: `if (ST.requiresNopBeforeDeallocVGPRs())`
  build `S_NOP 0` right before `S_SENDMSG ID_DEALLOC_VGPRS_GFX11Plus`; GCNSubtarget.h
  `requiresNopBeforeDeallocVGPRs() { return !HasGFX1250Insts; }` (true for gfx1201) and its comment
  calls it a hazard. Neither file says what the hardware does without the s_nop.
- SIInsertHardClauses.cpp (the hard-clause former; the task text called it AMDGPUInsertHardClauses):
  `emitClause` writes S_CLAUSE simm16 = length - 1 and treats an existing clause as covering
  `(imm & 63) + 1` instructions; length <= `maxHardClauseLength()` = 32 (FeatureMaxHardClauseLength32,
  in FeatureISAVersion12 and FeatureGFX12). gfx1201 takes the per-kind branch of
  `getHardClauseType`: upstream gates the mixed load/store branch on hasMixedLoadStoreClauses()
  (FeatureMixedLoadStoreClauses, absent from FeatureGFX12 / FeatureISAVersion12), ROCm d6f6cb69 on
  `getGeneration() == GFX10 || hasGFX1250Insts()` (gfx1201 is neither), so in both trees it gives separate MIMG load/store/atomic/sample, BVH, VMEM (buffer, global,
  scratch) load/store/atomic, FLAT load/store/atomic and SMEM types; LDS is "TODO" (illegal);
  only S_NOP is allowed inside (HARDCLAUSE_INTERNAL); SALU, export, branch, message, s_waitcnt and
  anything else is HARDCLAUSE_ILLEGAL. Clauses never cross a basic block (they are formed per MBB).
  The S_CLAUSE bits above [5:0] are not interpreted: the AMD table shipped here has no field layout
  for S_CLAUSE and LLVM never sets them.
"""
from __future__ import annotations
from collections import Counter

from . import isadb

LLVM = 'LLVM llvm/lib/Target/AMDGPU/'
CITE = {
    'wmma': LLVM + 'GCNHazardRecognizer.cpp fixWMMAHazards (GFX12 branch), getWaitStatesSince',
    'permlane': LLVM + 'GCNHazardRecognizer.cpp fixVcmpxPermlaneHazards, isVCmpXWritesExec, isPermlane; '
                'FeatureISAVersion12 FeatureVcmpxPermlaneHazard (GCNFeatures.td upstream; AMDGPU.td at ROCm d6f6cb69)',
    'dealloc': LLVM + 'SIInsertWaitcnts.cpp (S_NOP 0 before S_SENDMSG ID_DEALLOC_VGPRS_GFX11Plus if '
               'ST.requiresNopBeforeDeallocVGPRs()); GCNSubtarget.h requiresNopBeforeDeallocVGPRs()',
    'clause': LLVM + 'SIInsertHardClauses.cpp getHardClauseType, emitClause; '
              'FeatureISAVersion12 FeatureMaxHardClauseLength32 (GCNFeatures.td upstream; AMDGPU.td at '
              'ROCm d6f6cb69)',
}
ISA_SECTION = {'wmma': '7.12.1', 'permlane': '7.2.8', 'dealloc': '3.3.3', 'clause': '5.3'}
MAX_CLAUSE = 32
EXEC = {('s', 126), ('s', 127)}
BRANCHES = ('s_branch', 's_cbranch_')
# Inserted after SIInsertHardClauses by later passes (hazard recognizer, delay-ALU) or allowed
# inside by it (s_nop): never reported inside a clause.
CLAUSE_TOLERATED = {'s_nop', 'v_nop', 's_delay_alu', 's_wait_alu', 's_waitcnt_depctr'}


def _finding(rule: str, severity: str, message: str, pc: int | None, key: str, table: dict, **extra) -> dict:
    return {'rule': rule, 'severity': severity, 'message': message, 'pc': pc,
            'source': f"{isadb.cite(table, ISA_SECTION[key])}; {CITE[key]}", 'llvm': CITE[key], **extra}


def _name(i: dict, m: dict) -> str:
    """Table name (V_WMMA_...) or, for an instruction outside the table, the upper-cased mnemonic."""
    if m.get('name') and m['name'] != 'VOPD':
        return m['name']
    op = i['op'].upper()
    for s in isadb.SUFFIXES:
        if op.endswith(s.upper()):
            return op[:-len(s)]
    return op


def is_valu(i: dict, m: dict) -> bool:
    """SIInstrInfo::isVALU(MI, AllowLDSDMA=true) on the gfx1201 table: VALU group, VOPD pairs.

    An instruction outside the table counts as VALU when its mnemonic starts with v_ (it then ends
    a hazard window; for error-level rules a missed window is preferred to a guessed one).
    """
    if m.get('enc') == 'VOPD' or i.get('vopd_independent_decode'):
        return True
    if m.get('known'):
        return m['entry'].get('group') == 'VALU'
    return i['op'].startswith('v_')


def is_wmma(i: dict, m: dict) -> bool:
    return _name(i, m).startswith(('V_WMMA_', 'V_SWMMAC_'))


def is_vcmpx_exec(i: dict, m: dict) -> bool:
    """isVCmpXWritesExec: a VOPC/VOP3 compare that writes EXEC (V_CMPX_*, or V_CMP_* into exec)."""
    return _name(i, m).startswith('V_CMP') and bool(m['writes'] & EXEC)


def is_permlane(i: dict, m: dict) -> bool:
    return _name(i, m).startswith('V_PERMLANE')


def is_vnop(i: dict, m: dict) -> bool:
    return _name(i, m) == 'V_NOP'


def _operand_regs(i: dict) -> list[set]:
    out = []
    for tok in isadb.split_operands(i['args']):
        try:
            out.append(isadb.registers(tok))
        except Exception:
            out.append(set())
    return out


def _fmt(regs: set) -> str:
    v = sorted(n for k, n in regs if k == 'v')
    if not v:
        return ''
    return f'v{v[0]}' if len(v) == 1 else f'v[{v[0]}:{v[-1]}]' if v[-1] - v[0] == len(v) - 1 else \
        ', '.join(f'v{x}' for x in v)


def _blocks(ins: list[dict], edges: dict, reachable: set) -> tuple[list[int], dict, dict]:
    """Reachable straight-line chains (leader pc -> member pcs) in reverse postorder, and block preds.

    A pc continues its predecessor's chain when it has exactly one predecessor and that one has no
    other successor; everything else (entry, join, branch target) starts a block.
    """
    entry = ins[0]['pc']
    order = [i['pc'] for i in ins if i['pc'] in reachable]
    npred, single = Counter(), {}
    for p in order:
        ss = edges.get(p, [])
        npred.update(ss)
        if len(ss) == 1:
            single[ss[0]] = p
    def inner(pc: int) -> bool:
        return pc != entry and npred[pc] == 1 and pc in single
    members = {}
    for pc in order:
        if inner(pc):
            continue
        chain = [pc]
        while len(edges.get(chain[-1], [])) == 1 and inner(edges[chain[-1]][0]):
            chain.append(edges[chain[-1]][0])
        members[pc] = chain
    succ = {b: [s for s in edges.get(c[-1], []) if s in members] for b, c in members.items()}
    preds: dict[int, list[int]] = {b: [] for b in members}
    for b, ss in succ.items():
        for s in ss:
            preds[s].append(b)
    rpo, seen, stack = [], {entry}, [(entry, iter(succ[entry]))]
    while stack:
        b, it = stack[-1]
        for s in it:
            if s not in seen:
                seen.add(s); stack.append((s, iter(succ[s])))
                break
        else:
            stack.pop(); rpo.append(b)
    rpo.reverse()
    return rpo, members, preds


def _forward(ins: list[dict], edges: dict, reachable: set, events: dict, wanted: set) -> dict[int, tuple]:
    """In-states at the `wanted` pcs of a may-dataflow whose components are producer sets.

    `events[pc]` holds one entry per component: None (keeps the set), -1 (empties it) or k >= 0 (the
    set becomes {producer k}). Sets are int bitmasks over producer numbers. The fixpoint is computed
    per basic block in reverse postorder (a block's effect is the last event in it, or the identity),
    so the cost is O(blocks + edges) bitmask unions per sweep, with a few sweeps for loops, and the
    memory one mask per block and component; per-instruction states exist only for `wanted`.
    """
    if not ins or ins[0]['pc'] not in reachable:
        return {}
    rpo, members, preds = _blocks(ins, edges, reachable)
    n = len(next(iter(events.values()), ()))
    effect = {}
    for b, chain in members.items():
        eff = [None] * n
        for pc in chain:
            for c, e in enumerate(events[pc]):
                if e is not None:
                    eff[c] = e
        effect[b] = tuple(None if e is None else 0 if e < 0 else 1 << e for e in eff)
    empty = (0,) * n
    state_in, state_out = {}, {}
    changed = True
    while changed:
        changed = False
        for b in rpo:
            new = empty
            for p in preds[b]:
                o = state_out.get(p)
                if o is not None and o != new:
                    new = tuple(x | y for x, y in zip(new, o))
            if state_in.get(b) != new:
                state_in[b] = new
                state_out[b] = tuple(x if e is None else e for x, e in zip(new, effect[b]))
                changed = True
    out = {}
    for b, chain in members.items():
        st = state_in.get(b, empty)
        for pc in chain:
            if pc in wanted:
                out[pc] = st
            ev = events[pc]
            if any(e is not None for e in ev):
                st = tuple(x if e is None else 0 if e < 0 else 1 << e for x, e in zip(st, ev))
    return out


def _bits(mask: int) -> list[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


MAX_LISTED = 6


def alu_hazards(ins: list[dict], edges: dict, reachable: set, models: dict, table: dict) -> list[dict]:
    """WMMA_D_TO_AB and PERMLANE_AFTER_CMPX (both 'error': LLVM fixes them for gfx1201)."""
    wmma_pcs, cmpx_pcs, events, wanted = [], [], {}, set()
    for i in ins:
        pc, m = i['pc'], models[i['pc']]
        valu = is_valu(i, m)
        # IsHazard is tested before IsExpired, so a producer that is itself a VALU still counts.
        if is_wmma(i, m):
            w = len(wmma_pcs); wmma_pcs.append(pc); wanted.add(pc)
        else:
            w = -1 if valu else None
        if is_vcmpx_exec(i, m):
            c = len(cmpx_pcs); cmpx_pcs.append(pc)
        else:
            c = -1 if valu and not is_vnop(i, m) else None
        if is_permlane(i, m):
            wanted.add(pc)
        events[pc] = (w, c)
    if not wmma_pcs and not cmpx_pcs:
        return []
    states = _forward(ins, edges, reachable, events, wanted)
    by_pc = {i['pc']: i for i in ins}
    dregs: dict[int, set] = {}
    out = []
    for i in ins:
        pc = i['pc']
        if pc not in states:
            continue
        wmma, cmpx = states[pc]
        m = models[pc]
        if wmma and is_wmma(i, m):
            ops = _operand_regs(i)
            if len(ops) < 3:
                out.append(_finding('WMMA_OPERANDS', 'incomplete', 'WMMA operands not understood', pc, 'wmma', table))
            else:
                swmmac = _name(i, m).startswith('V_SWMMAC_')
                roles = [('A', ops[1]), ('B', ops[2])] + ([('index', ops[3])] if swmmac and len(ops) > 3 else [])
                for k in _bits(wmma):
                    prod = wmma_pcs[k]
                    p = by_pc[prod]
                    if prod not in dregs:
                        pops = _operand_regs(p)
                        dregs[prod] = pops[0] if pops else set()
                    d = dregs[prod]
                    hit = [(r, d & regs) for r, regs in roles if d & regs]
                    if not hit:
                        continue
                    what = ', '.join(f'{r} ({_fmt(x)})' for r, x in hit)
                    out.append(_finding('WMMA_D_TO_AB', 'error',
                        f"{i['op']} reads {what} written as D by {p['op']} at pc {prod:#x} with no VALU in "
                        'between on some path; LLVM (fixWMMAHazards) inserts v_nop before it', pc, 'wmma', table,
                        producer=prod, registers=[f'{k}{n}' for k, n in sorted({r for _, x in hit for r in x})]))
        if cmpx and is_permlane(i, m):
            ops = isadb.split_operands(i['args'])
            src0 = ops[1] if len(ops) > 1 else 'vN'
            prods = [cmpx_pcs[k] for k in _bits(cmpx)]
            names = ', '.join(f"{by_pc[p]['op']} at pc {p:#x}" for p in prods[:MAX_LISTED])
            if len(prods) > MAX_LISTED:
                names += f' and {len(prods) - MAX_LISTED} more'
            out.append(_finding('PERMLANE_AFTER_CMPX', 'error',
                f"{i['op']} follows {names}, which writes EXEC, with no VALU other than v_nop in between on "
                f'some path (s_nop, v_nop and scalar instructions do not separate them); LLVM '
                f'(fixVcmpxPermlaneHazards) inserts v_mov_b32 {src0}, {src0} before the permlane',
                pc, 'permlane', table, producer=prods[0], producers=prods))
    return out


def _is_dealloc(i: dict) -> bool:
    if i['op'] != 's_sendmsg':
        return False
    return 'dealloc_vgprs' in i['args'].lower() or int.from_bytes(bytes.fromhex(i['raw'])[:2], 'little') & 1023 == 3


def dealloc_nop(ins: list[dict], edges: dict, reachable: set, table: dict) -> list[dict]:
    """DEALLOC_NO_NOP ('review'): s_sendmsg MSG_DEALLOC_VGPRS whose CFG predecessor is not s_nop."""
    by_pc = {i['pc']: i for i in ins}
    preds: dict[int, list[int]] = {}
    for p, ss in edges.items():
        for s in ss:
            preds.setdefault(s, []).append(p)
    out = []
    for i in ins:
        if i['pc'] not in reachable or not _is_dealloc(i):
            continue
        bad = [p for p in preds.get(i['pc'], []) if p in reachable and by_pc[p]['op'] != 's_nop']
        if bad or not preds.get(i['pc']):
            out.append(_finding('DEALLOC_NO_NOP', 'review',
                's_sendmsg MSG_DEALLOC_VGPRS is not immediately preceded by s_nop; wherever LLVM emits this '
                'message for gfx1201 it puts s_nop 0 right before it (SIInsertWaitcnts, '
                'requiresNopBeforeDeallocVGPRs) and calls the case a hazard. What the hardware does without it is not documented in that source '
                '(its commit history was not available to this check)', i['pc'], 'dealloc', table,
                predecessors=bad))
    return out


def clause_kind(i: dict, m: dict) -> str | None:
    """SIInsertHardClauses::getHardClauseType for GFX12, or None where the AMD table cannot tell.

    Returns a clause type ('vmem_load', 'smem', ...), 'internal' (allowed inside), or 'illegal:<why>'.
    """
    op = i['op']
    if op in CLAUSE_TOLERATED:
        return 'internal'
    if not m.get('known') or m.get('enc') == 'VOPD':
        return 'illegal:valu' if m.get('enc') == 'VOPD' else None
    name, group, enc = m['name'], m['entry'].get('group'), m.get('enc')
    if group == 'SMEM':
        # S_PREFETCH_*, S_DCACHE_INV, S_ATC_PROBE: mayLoad/mayStore not in the table.
        return 'smem' if name.startswith(('S_LOAD_', 'S_BUFFER_LOAD_')) else None
    if group != 'VMEM':
        return f'illegal:{(group or "other").lower()}'
    if enc in ('ENC_VDS', 'ENC_VDSDIR'):
        return 'illegal:lds'
    if enc == 'ENC_VSAMPLE':
        return 'mimg_sample'
    if enc == 'ENC_VIMAGE':
        if 'BVH' in name:
            return 'bvh'
        if name.startswith('IMAGE_ATOMIC_'):
            return 'mimg_atomic'
        if name.startswith('IMAGE_STORE'):
            return 'mimg_store'
        if name.startswith('IMAGE_LOAD'):
            return 'mimg_load'
        return None
    rd, wr = m.get('mem_read'), m.get('mem_write')
    if not (rd or wr):
        return None          # GLOBAL_INV/GLOBAL_WB and similar: memory effect not in the table
    acc = 'atomic' if rd and wr else 'load' if rd else 'store'
    if enc in ('ENC_VGLOBAL', 'ENC_VSCRATCH', 'ENC_VBUFFER'):
        return 'vmem_' + acc
    if enc == 'ENC_VFLAT':
        return 'flat_' + acc
    return None


def clause_checks(ins: list[dict], edges: dict, reachable: set, models: dict, table: dict) -> list[dict]:
    """CLAUSE_TOO_LONG, CLAUSE_MIXED, CLAUSE_BRANCH ('review'): s_clause bodies LLVM would never emit."""
    targets = set()
    for i in ins:
        if i['op'].startswith(BRANCHES):
            fall = i['pc'] + i['size']
            targets.update(t for t in edges.get(i['pc'], []) if t != fall or i['op'] == 's_branch')
    out = []
    for ix, i in enumerate(ins):
        if i['op'] != 's_clause' or i['pc'] not in reachable:
            continue
        simm = int.from_bytes(bytes.fromhex(i['raw'])[:2], 'little')
        length = (simm & 63) + 1
        body = ins[ix + 1: ix + 1 + length]
        extra = {'clause_length': length, 'simm16': simm}
        if length > MAX_CLAUSE:
            out.append(_finding('CLAUSE_TOO_LONG', 'review',
                f's_clause {simm:#x} covers {length} instructions; LLVM never forms a gfx1201 hard clause longer '
                f'than {MAX_CLAUSE} (maxHardClauseLength, FeatureMaxHardClauseLength32). What the hardware does '
                'with a longer one is not documented in the sources this check cites', i['pc'], 'clause', table, **extra))
        kinds = [(j, clause_kind(j, models[j['pc']])) for j in body]
        bad = [(j, k) for j, k in kinds if k and k.startswith('illegal:')]
        real = {k for _, k in kinds if k and k not in ('internal',) and not k.startswith('illegal:')}
        if bad or len(real) > 1:
            parts = []
            if len(real) > 1:
                parts.append('memory instructions of different clause types (' + ', '.join(sorted(real)) + ')')
            if bad:
                parts.append('instructions LLVM never puts in a clause: ' +
                             ', '.join(f"{j['op']} at pc {j['pc']:#x}" for j, _ in bad[:6]) + (' ...' if len(bad) > 6 else ''))
            out.append(_finding('CLAUSE_MIXED', 'review',
                f's_clause {simm:#x} covers ' + '; and '.join(parts) + '. LLVM (SIInsertHardClauses) clauses only '
                'consecutive memory instructions of one type, with only s_nop between them', i['pc'], 'clause', table,
                kinds=sorted(real), illegal_pcs=[j['pc'] for j, _ in bad], **extra))
        inside = [j['pc'] for j in body if j['pc'] in targets]
        if inside:
            out.append(_finding('CLAUSE_BRANCH', 'review',
                f's_clause {simm:#x}: a branch targets ' + ', '.join(f'pc {p:#x}' for p in inside) + ' inside the clause '
                'body; LLVM forms clauses within one basic block with s_clause first, so a jump into the body '
                'never occurs in its output', i['pc'], 'clause', table, targets=inside, **extra))
    return out


def analyze(ins: list[dict], edges: dict, reachable: set, models: dict | None = None,
            table: dict | None = None, wave: int = 32) -> tuple[list[dict], dict]:
    """All rules of this module; `models` as from waits.operand_models (built here if absent)."""
    if table is None:
        table = isadb.load()
    if models is None:
        from .waits import operand_models
        models = operand_models(ins, table, wave)
    out = alu_hazards(ins, edges, reachable, models, table)
    out += dealloc_nop(ins, edges, reachable, table)
    out += clause_checks(ins, edges, reachable, models, table)
    stats = Counter(f['rule'] for f in out)
    stats = {'hazard_' + k.lower(): v for k, v in stats.items()}
    stats['hazard_clauses_checked'] = sum(1 for i in ins if i['op'] == 's_clause' and i['pc'] in reachable)
    return out, stats
