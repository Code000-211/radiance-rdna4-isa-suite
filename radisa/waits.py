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
    every counter, writing registers it cannot know: any VGPR it names on the vector counters,
    any SGPR on KMcnt (`ENTRY_FILE`). LLVM waits for LOADcnt, DScnt, KMcnt, EXPcnt, SAMPLEcnt
    and BVHcnt at the entry of every non-kernel function.
  * Before a return (S_SETPC_B64 s[30:31]) LLVM waits for every counter except STOREcnt; a
    LOADcnt that only a GLOBAL_INV holds is not waited for.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
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
MEMORY_MODEL = ('LLVM AMDGPU User Guide, Memory Model GFX12 (s_wait_dscnt 0 synchronises LDS with vector '
                'memory between waves); LLVM SIMemoryLegalizer.cpp SIGfx12CacheControl::insertWait (LDS at '
                'workgroup scope: DSCnt |= IsCrossAddrSpaceOrdering); SIInsertWaitcnts.cpp (no wait at '
                'S_BARRIER with FeatureBackOffBarrier); counter: AMD RDNA4 ISA §5.7.1')


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


# Counter state, internally: {counter: (young, old)}. `young` maps a producer to the least number of
# younger operations on the same counter on any path (< CAP); `old` is a bit set (a Python int, bit
# = the producer's index in `_Problem.prods[counter]`) of the producers with CAP or more younger
# operations. A producer is in at most one of the two. Every S_WAIT_* of a counter drops all `old`
# producers at once (an in-order counter waits for at most 63 < CAP younger operations; an
# out-of-order one only for 0), so they need no age, and adding one costs one bit, not a copy of
# every producer: the fixpoint stays near-linear in a kernel with thousands of unwaited stores.
# The registers a producer writes are fixed per producer (`_Problem.dest`), not part of the state.
# Counters with nothing outstanding are absent.
_NONE: tuple = ({}, 0)


def _issue(state: dict, dom: str, p: int, bit: dict) -> dict:
    young, old = state.get(dom, _NONE)
    bit = bit[dom]
    new = {}
    for q, a in young.items():
        if a + 1 < CAP:
            new[q] = a + 1
        else:
            old |= 1 << bit[q]
    b = 1 << bit[p]
    if old & b:
        old ^= b
    new.pop(p, None)
    new[p] = 0
    return {**state, dom: (new, old)}


def _wait(state: dict, dom: str, n: int) -> dict:
    cur = state.get(dom)
    if not cur:
        return state
    young, old = cur
    if dom in IN_ORDER:
        young = {p: a for p, a in young.items() if a < n}
        old = old if CAP < n else 0
    elif n == 0:
        young, old = {}, 0
    out = dict(state)
    if young or old:
        out[dom] = (young, old)
    else:
        del out[dom]
    return out


def _join(a: dict, b: dict, bit: dict) -> dict:
    if a is b:
        return a
    out = dict(a)
    for dom, (yb, ob) in b.items():
        cur = a.get(dom)
        if cur is None:
            out[dom] = (yb, ob)
            continue
        ya, oa = cur
        if ya is yb and oa == ob:
            continue
        young = dict(ya)
        for p, age in yb.items():
            if p not in young or age < young[p]:
                young[p] = age
        old = oa | ob
        if old:
            for p in young:
                m = 1 << bit[dom][p]
                if old & m:
                    old ^= m
        out[dom] = (young, old)
    return out


def _bits(x: int) -> list[int]:
    """Indices of the set bits of `x`, ascending."""
    s = bin(x)[:1:-1]
    return [i for i, c in enumerate(s) if c == '1']


class _Problem:
    """What the fixpoint needs to know about the producers of one code unit, built once per solve:
    per counter, every producer (`prods`, its index is its bit in `old`), the registers it writes
    (`dest`), and per register the bit set of the producers writing it (`writers`)."""

    def __init__(self, instructions: list[dict], models: dict, callee: bool):
        self.prods: dict[str, list[int]] = {}
        self.bit: dict[str, dict[int, int]] = {}
        self.dest: dict[str, dict[int, frozenset]] = {}
        self.writers: dict[str, dict] = {}
        if callee:
            for dom, regs in callee_entry_registers(instructions, models).items():
                self._add(dom, ENTRY, regs)
        for i in instructions:
            if i['op'] in CALL_OPS:
                self._add('store', i['pc'], frozenset())
            elif _wait_effect(i) is None:
                m = models[i['pc']]
                for dom in domains(m, i['op']):
                    self._add(dom, i['pc'], frozenset(m['writes']) if dom != 'store' else frozenset())

    def _add(self, dom: str, p: int, dest: frozenset):
        bits = self.bit.setdefault(dom, {})
        if p in bits:
            return
        bits[p] = len(bits)
        self.prods.setdefault(dom, []).append(p)
        self.dest.setdefault(dom, {})[p] = dest
        w = self.writers.setdefault(dom, {})
        for r in dest:
            w[r] = w.get(r, 0) | 1 << bits[p]

    def pending(self, st: dict, dom: str, regs=None) -> list[tuple[int, int]]:
        """[(producer, youngest-count)] outstanding on `dom` in raw state `st`; with `regs`, only the
        `old` producers that write one of them (young ones are all listed)."""
        young, old = st.get(dom, _NONE)
        out = []
        if old:
            if regs is not None:
                w = self.writers.get(dom, {})
                sel = 0
                for r in regs:
                    sel |= w.get(r, 0)
                old &= sel
            prods = self.prods[dom]
            out = [(prods[b], CAP) for b in _bits(old)]
        return out + list(young.items())


class Pending(Mapping):
    """Read-only {producer: (registers it writes, youngest-count)} of one counter (see `solve`)."""
    __slots__ = ('_pairs', '_dest')

    def __init__(self, problem: _Problem, st: dict, dom: str):
        self._pairs = dict(problem.pending(st, dom))
        self._dest = problem.dest[dom]

    def __getitem__(self, p):
        return self._dest[p], self._pairs[p]

    def __iter__(self):
        return iter(self._pairs)

    def __len__(self):
        return len(self._pairs)


class State(Mapping):
    """Read-only {counter: Pending} before one instruction; counters with nothing outstanding are absent."""
    __slots__ = ('_p', '_st', '_cache')

    def __init__(self, problem: _Problem, st: dict):
        self._p, self._st, self._cache = problem, st, {}

    def __getitem__(self, dom):
        if dom not in self._st:
            raise KeyError(dom)
        if dom not in self._cache:
            self._cache[dom] = Pending(self._p, self._st, dom)
        return self._cache[dom]

    def __iter__(self):
        return iter(self._st)

    def __len__(self):
        return len(self._st)


class States(Mapping):
    """pc -> State, built on access from the raw fixpoint result."""
    __slots__ = ('problem', 'raw')

    def __init__(self, problem: _Problem, raw: dict):
        self.problem, self.raw = problem, raw

    def __getitem__(self, pc):
        return State(self.problem, self.raw[pc])

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)

    def __contains__(self, pc):
        return pc in self.raw


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

    Severity 'review', not 'error': LLVM does not enforce this wait for gfx1201. SIInsertWaitcnts
    adds none at S_BARRIER (gfx1201 has FeatureBackOffBarrier), and SIGfx12CacheControl::insertWait
    (SIMemoryLegalizer.cpp, identical in upstream main and ROCm d6f6cb69) adds S_WAIT_DSCNT for LDS at
    workgroup scope only when the fence orders across address spaces (`DSCnt |=
    IsCrossAddrSpaceOrdering`, "LDS operations for all waves are executed in a total global
    ordering"). So the compiler emits s_wait_dscnt 0 before the barrier for __syncthreads() (fences
    at the cross-address-space "workgroup" scope), and correctly emits none for a "-one-as" fence
    (e.g. __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup-one-as")), an OpenCL local fence or a
    bare __builtin_amdgcn_s_barrier(). Machine code does not say which fence the source had; with an
    LDS write outstanding a cross-address-space release would have left its wait in place, so on
    compiler output this finding means the source synchronised only LDS (correct per LLVM's memory
    model) or had no fence (a source-level race), and in hand-edited code that a wait was removed.

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
            f = {'rule': 'LDS_STORE_BEFORE_BARRIER', 'severity': 'review', 'pc': pc,
                 'producer': None if prod == ENTRY else prod, 'counter': 'ds',
                 'message': f"{op} is reached while {producer_op(prod, by_pc)} ({at}) may still be outstanding. "
                            'LLVM emits s_wait_dscnt 0 here only for a release fence that orders LDS together '
                            'with global memory (the __syncthreads() fences); a "-one-as" or LDS-only fence '
                            'or a bare barrier gets no wait. If other waves use this write together with '
                            'global memory after the barrier, or the source had no fence, insert '
                            's_wait_dscnt 0 before the barrier',
                 'required_wait': 's_wait_dscnt 0', 'later_uses': [], 'source': MEMORY_MODEL, **extra}
            seen[prod] = f
            out.append(f)
    return out


def operand_models(instructions: list[dict], table: dict, wave: int = 32) -> Models:
    """pc -> operand model: `isadb.operand_model`, VOPD pairs from their own decoder (`vopd.decode`)."""
    from . import vopd
    models = Models()
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


# Register file each counter's operations return data to (RDNA4 ISA §5.7.1): vector memory,
# LDS, sample, BVH and export/parameter-load results are VGPRs, scalar-memory (and
# S_SENDMSG_RTN) results SGPRs. The AMD XML operand table agrees: every destination operand of an
# instruction `domains` puts on load/ds/sample/bvh/exp is a VGPR field.
ENTRY_FILE = {'load': 'v', 'ds': 'v', 'sample': 'v', 'bvh': 'v', 'exp': 'v', 'km': 's'}


def callee_entry_registers(instructions: list[dict], models: dict) -> dict[str, frozenset]:
    """Per counter, the registers a memory operation of the caller may still be writing at a
    callee's entry: the VGPRs the function names on the vector counters, its SGPRs up to VCC on
    KMcnt (`ENTRY_FILE`), nothing on STOREcnt. The caller's loads are not visible in the callee.
    Not s[30:31]: the call itself writes the return address there, after waiting for scalar loads
    of it (generateWaitcntInstBefore, the call's dst operand)."""
    regs = set()
    for i in instructions:
        m = models[i['pc']]
        regs |= {(k, n) for k, n in m['reads'] | m['writes'] if k == 'v' or k == 's' and n <= 107}
    regs -= {('s', 30), ('s', 31)}
    out = {d: frozenset(r for r in regs if r[0] == f) for d, f in ENTRY_FILE.items()}
    out['store'] = frozenset()
    return {d: out[d] for d in CALLEE_ENTRY}


def _transfer(state: dict, i: dict, models: dict, bit: dict) -> dict:
    eff = _wait_effect(i)
    if eff is not None:
        for dom, n in eff:
            state = _wait(state, dom, n)
        if i['op'] in CALL_OPS:
            # setStateOnFunctionEntryOrReturn: stores of the callee may still be outstanding.
            state = _issue(state, 'store', i['pc'], bit)
        return state
    m = models[i['pc']]
    for dom in domains(m, i['op']):
        state = _issue(state, dom, i['pc'], bit)
    return state


class Models(dict):
    """pc -> operand model (`operand_models`). It also holds the last `solve` result for these
    models, so a kernel pass that gets the same instructions, edges and models as `analyze` reuses
    the fixpoint; the cache lives and dies with the one code unit the models belong to."""
    solved: tuple = ()


def solve(instructions: list[dict], edges: dict, table: dict, wave: int = 32, models: dict | None = None,
          callee: bool = False) -> tuple[Mapping, dict, str]:
    """The counter fixpoint: (state_in, models, 'converged' | 'aborted').

    state_in: pc -> {counter: {producer: (registers it writes, youngest-count)}} (read-only
    mappings, `States`), the may-state before the instruction; counters are 'load', 'store',
    'sample', 'bvh', 'ds', 'km', 'exp', and one with nothing outstanding is absent. A producer is
    the pc of a memory instruction, the pc of a call (stores of the callee, on 'store'), or ENTRY
    (what a caller left outstanding, `callee=True`). youngest-count is the least number of younger
    operations on the same counter on any path (capped at CAP). `state_in.raw` / `.problem` are
    the internal form `analyze` works on.
    When `models` is a `Models` (as `operand_models` returns) the result is kept in it, and a
    second call with the very same instructions, edges and models objects returns it again.
    """
    if models is None:
        models = operand_models(instructions, table, wave)
    last = getattr(models, 'solved', ())
    if last and last[0] is instructions and last[1] is edges and last[2] == callee:
        return last[3]
    problem = _Problem(instructions, models, callee)
    raw: dict = {}
    status = 'converged'
    if instructions:
        by_pc = {i['pc']: i for i in instructions}
        entry = instructions[0]['pc']
        bit = problem.bit
        init = {}
        if callee:
            init = {d: ({ENTRY: 0}, 0) for d in CALLEE_ENTRY}
        raw[entry] = init
        work = deque([entry])
        queued = {entry}
        guard = 0
        npred: dict = {entry: 1}      # the entry state counts as one more predecessor of the entry
        for succ in edges.values():
            for t in succ:
                npred[t] = npred.get(t, 0) + 1
        while work:
            guard += 1
            if guard > 200 * len(instructions) + 1000:
                status = 'aborted'
                break
            pc = work.popleft()
            queued.discard(pc)
            out = _transfer(raw[pc], by_pc[pc], models, bit)
            for s in edges.get(pc, []):
                old = raw.get(s)
                # One predecessor: its out-state is the in-state, no join needed (most instructions).
                new = out if old is None or npred[s] == 1 else _join(old, out, bit)
                if old is None or new is not old and new != old:
                    raw[s] = new
                    if s not in queued:
                        queued.add(s)
                        work.append(s)
    result = (States(problem, raw), models, status)
    if isinstance(models, Models):
        models.solved = (instructions, edges, callee, result)
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
        for dom, entries in sorted(state_in[pc].items()) if pc in state_in else ():
            if dom == 'store' or not entries:
                continue
            if dom == 'load' and all(p != ENTRY and by_pc[p]['op'] == 'global_inv' for p in entries):
                continue
            pending[dom] = sorted(entries)
        if not pending:
            continue
        what = '; '.join(f"{WAIT_NAME[d].removeprefix('s_wait_')}: " + describe(ps, by_pc, origin)
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


LISTED = 8


def describe(prods: list[int], by_pc: dict, origin: dict | None) -> str:
    """'op (where), ...' for outstanding producers (in the order given); past LISTED of them, the
    last LISTED and the number of the others."""
    shown = prods if len(prods) <= LISTED else prods[-LISTED:]
    text = ', '.join(f'{producer_op(p, by_pc)} ({_where(p, origin)})' for p in shown)
    if len(prods) > LISTED:
        text = f'{len(prods) - LISTED} older ones and ' + text
    return text


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
    raw, problem = state_in.raw, state_in.problem

    findings = []
    # One finding per unwaited producer; ENTRY counts once per register file (`ENTRY_FILE`).
    reported: dict = {}
    for i in instructions:
        pc = i['pc']
        if pc not in raw or pc not in reachable:
            continue
        st = raw[pc]
        if not st:
            continue
        m = models[pc]
        call = i['op'] in CALL_OPS
        if _wait_effect(i) is not None and not call:
            continue
        own = set(domains(m, i['op']))
        half = _store_half(i)
        used = m['reads'] | m['writes']
        for dom in st:
            if dom == 'store' or call and dom != 'km':
                # Stores write no register. A call waits only for scalar loads of its address /
                # return-address registers; the callee waits for the rest at its entry.
                continue
            dests = problem.dest[dom]
            for prod, younger in problem.pending(st, dom, used):
                dest = dests[prod]
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
                key = (ENTRY, ENTRY_FILE[dom]) if prod == ENTRY else prod
                if key in reported:
                    # One finding per unwaited producer; later uses are the same bug.
                    f = reported[key]
                    if f['pc'] != pc and f['later_uses'][-1:] != [pc]:
                        f['later_uses'].append(pc)
                    continue
                need = younger if dom in IN_ORDER else 0
                required = f'{WAIT_NAME[dom]} {need}'
                if prod == ENTRY:
                    # Every counter the caller may still hold for these registers.
                    held = []
                    for d in st:
                        if d != 'store' and problem.dest[d].get(ENTRY, frozenset()) & hit:
                            age = dict(problem.pending(st, d, hit)).get(ENTRY)
                            if age is not None:
                                held.append(f'{WAIT_NAME[d]} {age if d in IN_ORDER else 0}')
                    required = ', '.join(held)
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
                reported[key] = f
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
