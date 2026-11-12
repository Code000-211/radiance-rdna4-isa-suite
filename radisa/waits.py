"""Memory dependency counter model over the whole kernel CFG.

Normative basis (RDNA4 ISA §5.7.1 Memory Dependency Counters, PDF pp.62-65):
  * LOADcnt, SAMPLEcnt, BVHcnt, DScnt: instructions of one type return in issue
    order, so `S_WAIT_<cnt> N` proves every op of that type except the N youngest.
  * FLAT increments LOADcnt/STOREcnt *and* DScnt; FLAT and LDS stay in order in DScnt.
  * KMcnt: scalar-memory loads return out of order; only `S_WAIT_KMCNT 0` proves
    completion. EXPcnt mixes export/param-load types; treated the same way.
  * S_WAIT_LOADCNT_DSCNT / S_WAIT_STORECNT_DSCNT: MEM = SIMM16[13:8], DS = SIMM16[5:0]
    (§16.5 SOPP Instructions).

A finding means: on at least one CFG path a register is read or overwritten while
an earlier memory instruction that writes it may still be outstanding. This is a
forward may-analysis; it is sound for the CFG it is given (direct branches only).

Calls and called functions follow LLVM's SIInsertWaitcnts (`CALLS` below), the code that
establishes these conventions; the ISA guide states no calling convention:
  * A call (S_SWAPPC_B64 / S_CALL_B64) waits only for scalar loads that write its address or
    return-address operands. After it every counter except STOREcnt is drained: the callee
    waits at its entry and before it returns. Stores of the callee may still be outstanding.
  * A called function (`callee=True`) starts with whatever the caller left outstanding, on
    every counter, writing registers it cannot know. LLVM waits for LOADcnt, DScnt, KMcnt,
    EXPcnt, SAMPLEcnt and BVHcnt at the entry of every non-kernel function.
  * Before a return (S_SETPC_B64 s[30:31]) LLVM waits for every counter except STOREcnt; a
    LOADcnt that only a GLOBAL_INV holds is not waited for.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import re

from . import isadb, sgpr

IN_ORDER = {'load', 'sample', 'bvh', 'ds'}
WAIT_OPS = {
    's_wait_loadcnt': [('load', 0, 0x3f)], 's_wait_storecnt': [('store', 0, 0x3f)],
    's_wait_samplecnt': [('sample', 0, 0x3f)], 's_wait_bvhcnt': [('bvh', 0, 0x7)],
    's_wait_dscnt': [('ds', 0, 0x3f)], 's_wait_kmcnt': [('km', 0, 0x1f)],
    's_wait_expcnt': [('exp', 0, 0x7)],
    's_wait_loadcnt_dscnt': [('load', 8, 0x3f), ('ds', 0, 0x3f)],
    's_wait_storecnt_dscnt': [('store', 8, 0x3f), ('ds', 0, 0x3f)],
}
DRAIN_ALL = {'s_wait_idle', 's_waitcnt'}
WAIT_NAME = {'load': 's_wait_loadcnt', 'sample': 's_wait_samplecnt', 'bvh': 's_wait_bvhcnt',
             'ds': 's_wait_dscnt', 'km': 's_wait_kmcnt', 'exp': 's_wait_expcnt', 'store': 's_wait_storecnt'}
CAP = 64
CALL_OPS = ('s_swappc_b64', 's_call_b64')
# Producer key of the memory operations a caller left outstanding at a callee's entry.
ENTRY = -1
CALLEE_ENTRY = ('load', 'ds', 'km', 'exp', 'sample', 'bvh', 'store')
# SIRegisterInfo::getReturnAddressReg: SGPR30_SGPR31.
RETURN_ADDRESS = 's[30:31]'
CALLS = ('LLVM SIInsertWaitcnts.cpp: SIInsertWaitcnts::run (non-kernel function entry: '
         's_wait_loadcnt_dscnt 0, s_wait_kmcnt/expcnt/samplecnt/bvhcnt 0, STOREcnt left pending by '
         'setStateOnFunctionEntryOrReturn), generateWaitcntInstBefore (SI_RETURN / S_SETPC_B64_return: '
         'all counters but STOREcnt 0, simplifyVmemInvOnlyWait; a call waits only for SMEM writing its '
         'operands), updateEventWaitcntAfter (after a call: all counters but STOREcnt 0)')
MEMORY_MODEL = ('LLVM AMDGPU User Guide, Memory Model GFX12, code sequences: s_wait_dscnt 0 before a '
                'workgroup barrier; counter: AMD RDNA4 ISA §5.7.1')


def domains(model: dict, op: str) -> list[str]:
    """Counters an instruction increments, from its AMD encoding and memory operands."""
    if not model.get('known'):
        return []
    enc = model['enc']
    name = model['name']
    if 'SMEM' in enc:
        return ['km']
    if name.startswith('S_SENDMSG'):
        return ['km']
    if 'VDSDIR' in enc or 'VEXPORT' in enc:
        return ['exp']
    if 'VDS' in enc:
        return ['ds']
    if 'VSAMPLE' in enc:
        return ['sample']
    if 'BVH' in model['entry']['subgroups']:
        return ['bvh']
    if any(e in enc for e in ('VGLOBAL', 'VSCRATCH', 'VBUFFER', 'VIMAGE', 'VFLAT')):
        if name in ('GLOBAL_INV',):
            kind = 'load'
        elif name.startswith('GLOBAL_WB'):
            kind = 'store'
        elif model['writes'] and model.get('returns', True):
            kind = 'load'
        else:
            kind = 'store'
        return [kind, 'ds'] if 'VFLAT' in enc else [kind]
    return []


_STORE_HALF = re.compile(r'^(?:global|flat|ds)_store_(?:d16_hi_)?b(?:8|16)(?:_d16_hi)?$')
_REG = re.compile(r'\bv(\d+)\b|\bv\[(\d+):(\d+)\]')


def _load_half(op: str) -> str | None:
    """Half of the VGPR a D16 load writes: `*_d16_*` the low 16 bits, `*_d16_hi_*` the high."""
    if '_load_' not in op or 'd16' not in op:
        return None
    return 'hi' if 'd16_hi' in op else 'lo'


def _store_half(ins: dict) -> tuple[tuple[str, int], str] | None:
    """(data register, half) for an 8/16-bit store: it reads only that half of its data
    VGPR. Only the `address, data` operand order of GLOBAL/FLAT/DS stores is understood,
    and a data register that is also part of the address is read whole."""
    op = ins['op']
    if not _STORE_HALF.match(op):
        return None
    ops = [o.strip() for o in ins['args'].split(',')]
    if len(ops) < 2:
        return None
    m = re.match(r'v(\d+)(?:\s|$)', ops[1])
    if not m:
        return None
    data = int(m[1])
    for a, lo, hi in _REG.findall(ops[0]):
        if a and int(a) == data or lo and int(lo) <= data <= int(hi):
            return None
    return ('v', data), ('hi' if 'd16_hi' in op else 'lo')


def _wait_effect(ins: dict) -> list[tuple[str, int]] | None:
    """(counter, N) pairs an S_WAIT_* instruction (or a call, see `CALLS`) waits for."""
    op = ins['op']
    if op in DRAIN_ALL:
        return [(d, 0) for d in WAIT_NAME]
    if op in CALL_OPS:
        # updateEventWaitcntAfter: applyWaitcnt(getAllZeroWaitcnt(IncludeVSCnt=false)).
        return [(d, 0) for d in WAIT_NAME if d != 'store']
    spec = WAIT_OPS.get(op)
    if not spec:
        return None
    raw = bytes.fromhex(ins['raw'])
    if len(raw) < 2:
        return None
    imm = int.from_bytes(raw[:2], 'little')
    return [(d, (imm >> shift) & mask) for d, shift, mask in spec]


# State: {domain: {producer_pc: (frozenset(dest regs), younger_min)}}
def _issue(state: dict, dom: str, pc: int, dest: frozenset) -> dict:
    cur = state.get(dom, {})
    new = {p: (r, min(CAP, y + 1)) for p, (r, y) in cur.items()}
    new[pc] = (dest, 0)
    return {**state, dom: new}


def _wait(state: dict, dom: str, n: int) -> dict:
    cur = state.get(dom)
    if not cur:
        return state
    if dom in IN_ORDER:
        kept = {p: v for p, v in cur.items() if v[1] < n}
    else:
        kept = {} if n == 0 else cur
    return {**state, dom: kept}


def _join(a: dict, b: dict) -> dict:
    out = {}
    for dom in set(a) | set(b):
        x, y = a.get(dom, {}), b.get(dom, {})
        merged = dict(x)
        for p, (r, age) in y.items():
            if p in merged:
                merged[p] = (merged[p][0] | r, min(merged[p][1], age))
            else:
                merged[p] = (r, age)
        out[dom] = merged
    return out


BARRIERS = ('s_barrier_signal', 's_barrier_signal_isfirst', 's_barrier_wait')


def producer_op(prod: int, by_pc: dict) -> str:
    """Mnemonic of a producer, or what a synthetic producer (ENTRY, a call's callee stores) stands for."""
    if prod == ENTRY:
        return 'a memory operation of the caller'
    op = by_pc[prod]['op']
    return f'a store of the function {op} calls' if op in CALL_OPS else op


def _where(prod: int, origin: dict | None) -> str:
    if prod == ENTRY:
        return 'issued before the call'
    if origin and origin.get(prod, {}).get('line'):
        return f"{Path(origin[prod]['file']).name}:{origin[prod]['line']}"
    return f'pc {prod:#x}'


def _sync_findings(instructions, state_in, reachable, by_pc, origin) -> list[dict]:
    """Local-memory writes that may be outstanding when a workgroup barrier is reached.

    The barrier is the one synchronisation the machine code itself states. Which waits an
    atomic, a fence or a cache invalidate needs depends on the ordering and scope written in
    the source, which machine code does not carry, so those are not checked. An outstanding
    read at a barrier is not reported either: the compiler leaves reads whose value is not
    used on that path outstanding.
    """
    out, seen = [], {}
    for i in instructions:
        pc, op = i['pc'], i['op']
        if op not in BARRIERS or pc not in state_in or pc not in reachable:
            continue
        for prod in sorted(p for p in state_in[pc].get('ds', {}) if p == ENTRY or '_load' not in by_pc[p]['op']):
            if prod in seen:
                seen[prod]['later_uses'].append(pc)
                continue
            at = _where(prod, origin)
            extra = {}
            if origin and prod in origin:
                extra['producer_origin'] = origin[prod]
            f = {'rule': 'LDS_STORE_BEFORE_BARRIER', 'severity': 'error', 'pc': pc,
                 'producer': None if prod == ENTRY else prod, 'counter': 'ds',
                 'message': f"{op} is reached while {producer_op(prod, by_pc)} ({at}) may still be outstanding: "
                            'the waves the barrier releases may read local memory before the write '
                            'has landed; insert s_wait_dscnt 0 before the barrier',
                 'required_wait': 's_wait_dscnt 0', 'later_uses': [], 'source': MEMORY_MODEL, **extra}
            seen[prod] = f
            out.append(f)
    return out


def operand_models(instructions: list[dict], table: dict, wave: int = 32) -> dict[int, dict]:
    """pc -> operand model: `isadb.operand_model`, VOPD pairs from their own decoder (`vopd.decode`)."""
    from . import vopd
    models = {}
    for i in instructions:
        if i.get('vopd_independent_decode'):
            p = vopd.decode(bytes.fromhex(i['raw']))
            m = {'known': True, 'name': 'VOPD', 'enc': 'VOPD', 'entry': {'subgroups': []},
                 'reads': {('v', x) for x in p.x.reads() | p.y.reads()},
                 'writes': {('v', p.x.dst), ('v', p.y.dst)}}
        else:
            m = isadb.operand_model(table, i['op'], i['args'], wave)
        models[i['pc']] = m
    return models


def coverage(instructions: list[dict], models: dict) -> dict:
    """How many instructions the operand model knows, and the mnemonics it does not."""
    stats = {'operand_model_known': 0, 'operand_model_unknown': 0, 'unknown_mnemonics': {}}
    for i in instructions:
        if models[i['pc']]['known']:
            stats['operand_model_known'] += 1
        else:
            stats['operand_model_unknown'] += 1
            stats['unknown_mnemonics'][i['op']] = stats['unknown_mnemonics'].get(i['op'], 0) + 1
    return stats


def callee_entry_registers(instructions: list[dict], models: dict) -> frozenset:
    """Registers a load of the caller may still be writing at a callee's entry: every VGPR and
    SGPR (up to VCC) the function names. The caller's loads are not visible in the callee. Not
    s[30:31]: the call itself writes the return address there, after waiting for scalar loads
    of it (generateWaitcntInstBefore, the call's dst operand)."""
    regs = set()
    for i in instructions:
        m = models[i['pc']]
        regs |= {(k, n) for k, n in m['reads'] | m['writes'] if k == 'v' or k == 's' and n <= 107}
    return frozenset(regs - {('s', 30), ('s', 31)})


def _transfer(state: dict, i: dict, models: dict) -> dict:
    eff = _wait_effect(i)
    if eff is not None:
        for dom, n in eff:
            state = _wait(state, dom, n)
        if i['op'] in CALL_OPS:
            # setStateOnFunctionEntryOrReturn: stores of the callee may still be outstanding.
            state = _issue(state, 'store', i['pc'], frozenset())
        return state
    m = models[i['pc']]
    for dom in domains(m, i['op']):
        dest = frozenset(m['writes']) if dom != 'store' else frozenset()
        state = _issue(state, dom, i['pc'], dest)
    return state


_SOLVED: tuple = ()


def solve(instructions: list[dict], edges: dict, table: dict, wave: int = 32, models: dict | None = None,
          callee: bool = False) -> tuple[dict, dict, str]:
    """The counter fixpoint: (state_in, models, 'converged' | 'aborted').

    state_in: pc -> {counter: {producer: (registers it writes, youngest-count)}}, the may-state
    before the instruction; counters are 'load', 'store', 'sample', 'bvh', 'ds', 'km', 'exp'.
    A producer is the pc of a memory instruction, the pc of a call (stores of the callee, on
    'store'), or ENTRY (what a caller left outstanding, `callee=True`). youngest-count is the
    least number of younger operations on the same counter on any path (capped at CAP).
    The result of the last `analyze`/`solve` on the very same instructions, edges and models
    objects is reused, so a kernel pass gets it without solving again.
    """
    global _SOLVED
    if models is None:
        models = operand_models(instructions, table, wave)
    last = _SOLVED
    if last and last[0] is instructions and last[1] is edges and last[2] is models and last[3] == callee:
        return last[4]
    if not instructions:
        return {}, models, 'converged'
    by_pc = {i['pc']: i for i in instructions}
    entry = instructions[0]['pc']
    init = {}
    if callee:
        regs = callee_entry_registers(instructions, models)
        init = {d: {ENTRY: (regs if d != 'store' else frozenset(), 0)} for d in CALLEE_ENTRY}
    state_in = {entry: init}
    work = deque([entry])
    guard, status = 0, 'converged'
    while work:
        guard += 1
        if guard > 200 * len(instructions) + 1000:
            status = 'aborted'
            break
        pc = work.popleft()
        out = _transfer(state_in[pc], by_pc[pc], models)
        for s in edges.get(pc, []):
            old = state_in.get(s)
            new = out if old is None else _join(old, out)
            if old is None or new != old:
                state_in[s] = new
                work.append(s)
    result = (state_in, models, status)
    _SOLVED = (instructions, edges, models, callee, result)
    return result


def _return_findings(instructions, state_in, reachable, by_pc, origin) -> list[dict]:
    """CALLEE_RETURN_OUTSTANDING: a return of a called function with counters LLVM drains there.

    LLVM waits before SI_RETURN / S_SETPC_B64_return for every counter but STOREcnt
    (generateWaitcntInstBefore) and a LOADcnt held only by GLOBAL_INV (simplifyVmemInvOnlyWait),
    and every caller it compiles relies on that (updateEventWaitcntAfter), as `analyze` does for
    calls. A return without those waits breaks the caller's assumption; what a particular caller
    then reads too early is not visible here, so this is for review.
    """
    out = []
    for i in instructions:
        pc = i['pc']
        if i['op'] != 's_setpc_b64' or i['args'].strip() != RETURN_ADDRESS or pc not in reachable:
            continue
        pending = {}
        for dom, entries in sorted(state_in.get(pc, {}).items()):
            if dom == 'store' or not entries:
                continue
            if dom == 'load' and all(p != ENTRY and by_pc[p]['op'] == 'global_inv' for p in entries):
                continue
            pending[dom] = sorted(entries)
        if not pending:
            continue
        what = '; '.join(f"{WAIT_NAME[d].removeprefix('s_wait_')}: " +
                         ', '.join(f'{producer_op(p, by_pc)} ({_where(p, origin)})' for p in ps)
                         for d, ps in pending.items())
        waits_needed = [f'{WAIT_NAME[d]} 0' for d in pending]
        out.append({'rule': 'CALLEE_RETURN_OUTSTANDING', 'severity': 'review', 'pc': pc,
                    'message': f'the function returns while memory operations may still be outstanding ({what}). '
                               'LLVM waits for every counter but STOREcnt before a return, and a caller it '
                               'compiles reads the results after the call without a wait; insert '
                               f"{', '.join(waits_needed)} before the return",
                    'required_wait': waits_needed,
                    'pending': {d: [None if p == ENTRY else p for p in ps] for d, ps in pending.items()},
                    'source': CALLS})
    return out


def analyze(instructions: list[dict], edges: dict, reachable: set, table: dict, wave: int = 32,
            origin: dict | None = None, models: dict | None = None,
            callee: bool = False) -> tuple[list[dict], dict]:
    """Return MISSING_WAIT findings and coverage statistics.

    `models`: `operand_models()` of these instructions, when the caller has built them already.
    `callee`: the code is a called function, not a kernel (see `CALLS`): the caller's memory
    operations may be outstanding at entry, returns are checked (CALLEE_RETURN_OUTSTANDING),
    and every SGPR pair is tracked for the ALU hazard (sgpr.analyze).
    """
    by_pc = {i['pc']: i for i in instructions}
    if models is None:
        models = operand_models(instructions, table, wave)
    stats = {**coverage(instructions, models), 'memory_ops_tracked': 0, 'd16_other_half_stores': 0}
    if not instructions:
        return [], stats
    state_in, models, stats['fixpoint'] = solve(instructions, edges, table, wave, models, callee)

    findings = []
    reported: dict[int, dict] = {}
    for i in instructions:
        pc = i['pc']
        if pc not in state_in or pc not in reachable:
            continue
        st = state_in[pc]
        m = models[pc]
        call = i['op'] in CALL_OPS
        if _wait_effect(i) is not None and not call:
            continue
        own = set(domains(m, i['op']))
        half = _store_half(i)
        for dom, entries in st.items():
            if call and dom != 'km':
                # A call waits only for scalar loads of its address / return-address registers;
                # the callee waits for the rest at its entry.
                continue
            for prod, (dest, younger) in entries.items():
                reads = m['reads']
                if (half and prod != ENTRY and half[0] in dest
                        and _load_half(by_pc[prod]['op']) not in (None, half[1])):
                    # The store reads bits 15:0 (or 31:16) of its data register and the
                    # outstanding D16 load writes the other 16: no data flows between them.
                    reads = reads - {half[0]}
                    stats['d16_other_half_stores'] += 1
                hit = dest & (reads | m['writes'])
                if not hit:
                    continue
                # Same in-order counter writing the same registers: the later
                # write lands later, no wait is required (WAW in issue order).
                if dom in own and dom in IN_ORDER and not (dest & reads):
                    continue
                if prod in reported:
                    # One finding per unwaited producer; later uses are the same bug.
                    reported[prod]['later_uses'].append(pc)
                    continue
                need = younger if dom in IN_ORDER else 0
                required = f'{WAIT_NAME[dom]} {need}'
                if prod == ENTRY:
                    # Every counter the caller may still hold for these registers.
                    required = ', '.join(f'{WAIT_NAME[d]} {e[ENTRY][1] if d in IN_ORDER else 0}'
                                         for d, e in st.items() if ENTRY in e and e[ENTRY][0] & hit)
                regs = ', '.join(sorted(f'{a}{b}' for a, b in hit))
                use = 'reads' if hit & reads else 'overwrites'
                msg = (f"{i['op']} {use} {regs} while {producer_op(prod, by_pc)} ({_where(prod, origin)}) may "
                       f"still be outstanding; insert {required} (or lower) before it")
                extra = {}
                source = isadb.cite(table, '5.7.1')
                if prod == ENTRY:
                    msg += ('. This is a called function: LLVM waits for every counter but STOREcnt at the '
                            'entry of a non-kernel function, because the registers the caller loads are '
                            'not known there')
                    extra['producer_kind'] = 'caller'
                    source += '; ' + CALLS
                if origin and prod in origin:
                    extra['producer_origin'] = origin[prod]
                    if origin[prod].get('inline_asm'):
                        msg += ('. The producer is inline asm: the compiler does not insert '
                                'waits for memory instructions inside asm statements')
                f = {'rule': 'MISSING_WAIT', 'severity': 'error', 'message': msg,
                     'pc': pc, 'producer': None if prod == ENTRY else prod, 'counter': dom,
                     'required_wait': required,
                     'registers': sorted(f'{a}{b}' for a, b in hit), 'later_uses': [],
                     'consumer_operand_model': 'xml' if m['known'] else 'text-superset',
                     'source': source, **extra}
                reported[prod] = f
                findings.append(f)
    findings += _sync_findings(instructions, state_in, reachable, by_pc, origin)
    if callee:
        rf = _return_findings(instructions, state_in, reachable, by_pc, origin)
        stats['callee_returns_outstanding'] = len(rf)
        findings += rf
    sf, sstats = sgpr.analyze(instructions, edges, reachable, models, wave, callee)
    findings += sf
    stats.update(sstats)
    stats['memory_ops_tracked'] = sum(1 for i in instructions if domains(models[i['pc']], i['op']))
    return findings, stats
