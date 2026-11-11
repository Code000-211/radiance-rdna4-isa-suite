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
    op = ins['op']
    # A call returns only after the callee's prologue waited for every counter
    # (LLVM AMDGPU calling convention: non-kernel functions start with all
    # S_WAIT_*CNT 0). The continuation therefore sees drained counters.
    if op in DRAIN_ALL or op == 's_swappc_b64':
        return [(d, 0) for d in WAIT_NAME]
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
        for prod in sorted(p for p in state_in[pc].get('ds', {}) if '_load' not in by_pc[p]['op']):
            if prod in seen:
                seen[prod]['later_uses'].append(pc)
                continue
            at = f'pc {prod:#x}'
            extra = {}
            if origin and prod in origin:
                extra['producer_origin'] = origin[prod]
                if origin[prod].get('line'):
                    at = f"{Path(origin[prod]['file']).name}:{origin[prod]['line']}"
            f = {'rule': 'LDS_STORE_BEFORE_BARRIER', 'severity': 'error', 'pc': pc, 'producer': prod,
                 'counter': 'ds',
                 'message': f"{op} is reached while {by_pc[prod]['op']} ({at}) may still be outstanding: "
                            'the waves the barrier releases may read local memory before the write '
                            'has landed; insert s_wait_dscnt 0 before the barrier',
                 'required_wait': 's_wait_dscnt 0', 'later_uses': [], 'source': MEMORY_MODEL, **extra}
            seen[prod] = f
            out.append(f)
    return out


def analyze(instructions: list[dict], edges: dict, reachable: set, table: dict, wave: int = 32,
            origin: dict | None = None) -> tuple[list[dict], dict]:
    """Return MISSING_WAIT findings and coverage statistics."""
    by_pc = {i['pc']: i for i in instructions}
    models = {}
    stats = {'operand_model_known': 0, 'operand_model_unknown': 0, 'unknown_mnemonics': {},
             'memory_ops_tracked': 0, 'd16_other_half_stores': 0}
    for i in instructions:
        if i.get('vopd_independent_decode'):
            m = {'known': True, 'name': 'VOPD', 'reads': set(), 'writes': set(), 'enc': 'VOPD',
                 'entry': {'subgroups': []}}
            from . import vopd
            p = vopd.decode(bytes.fromhex(i['raw']))
            m['reads'] = {('v', x) for x in p.x.reads() | p.y.reads()}
            m['writes'] = {('v', p.x.dst), ('v', p.y.dst)}
        else:
            m = isadb.operand_model(table, i['op'], i['args'], wave)
        models[i['pc']] = m
        if m['known']:
            stats['operand_model_known'] += 1
        else:
            stats['operand_model_unknown'] += 1
            stats['unknown_mnemonics'][i['op']] = stats['unknown_mnemonics'].get(i['op'], 0) + 1

    def transfer(state: dict, i: dict) -> dict:
        eff = _wait_effect(i)
        if eff is not None:
            for dom, n in eff:
                state = _wait(state, dom, n)
            return state
        m = models[i['pc']]
        for dom in domains(m, i['op']):
            dest = frozenset(m['writes']) if dom != 'store' else frozenset()
            state = _issue(state, dom, i['pc'], dest)
        return state

    if not instructions:
        return [], stats
    entry = instructions[0]['pc']
    state_in = {entry: {}}
    work = deque([entry])
    guard = 0
    while work:
        guard += 1
        if guard > 200 * len(instructions) + 1000:
            stats['fixpoint'] = 'aborted'
            break
        pc = work.popleft()
        out = transfer(state_in[pc], by_pc[pc])
        for s in edges.get(pc, []):
            old = state_in.get(s)
            new = out if old is None else _join(old, out)
            if old is None or new != old:
                state_in[s] = new
                work.append(s)
    else:
        stats['fixpoint'] = 'converged'

    findings = []
    reported: dict[int, dict] = {}
    for i in instructions:
        pc = i['pc']
        if pc not in state_in or pc not in reachable:
            continue
        st = state_in[pc]
        m = models[pc]
        if _wait_effect(i) is not None:
            continue
        own = set(domains(m, i['op']))
        half = _store_half(i)
        for dom, entries in st.items():
            for prod, (dest, younger) in entries.items():
                reads = m['reads']
                if half and half[0] in dest and _load_half(by_pc[prod]['op']) not in (None, half[1]):
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
                p = by_pc[prod]
                regs = ', '.join(sorted(f'{a}{b}' for a, b in hit))
                use = 'reads' if hit & reads else 'overwrites'
                at = f'pc {prod:#x}'
                if origin and origin.get(prod, {}).get('line'):
                    o = origin[prod]
                    at = f"{Path(o['file']).name}:{o['line']}"
                msg = (f"{i['op']} {use} {regs} while {p['op']} ({at}) may still be "
                       f"outstanding; insert {WAIT_NAME[dom]} {need} (or lower) before it")
                extra = {}
                if origin and prod in origin:
                    extra['producer_origin'] = origin[prod]
                    if origin[prod].get('inline_asm'):
                        msg += ('. The producer is inline asm: the compiler does not insert '
                                'waits for memory instructions inside asm statements')
                f = {'rule': 'MISSING_WAIT', 'severity': 'error', 'message': msg,
                     'pc': pc, 'producer': prod, 'counter': dom,
                     'required_wait': f'{WAIT_NAME[dom]} {need}',
                     'registers': sorted(f'{a}{b}' for a, b in hit), 'later_uses': [],
                     'consumer_operand_model': 'xml' if m['known'] else 'text-superset',
                     'source': isadb.cite(table, '5.7.1'), **extra}
                reported[prod] = f
                findings.append(f)
    findings += _sync_findings(instructions, state_in, reachable, by_pc, origin)
    sf, sstats = sgpr.analyze(instructions, edges, reachable, models, wave)
    findings += sf
    stats.update(sstats)
    stats['memory_ops_tracked'] = sum(1 for i in instructions if domains(models[i['pc']], i['op']))
    return findings, stats
