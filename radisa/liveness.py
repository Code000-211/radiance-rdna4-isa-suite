"""UNINITIALIZED_READ: a register read on some path from the kernel entry before anything wrote it.

Normative basis. What a kernel finds in its registers at entry is the AMDHSA ABI, not the ISA
guide: LLVM AMDGPU User Guide (llvm/docs/AMDGPUUsage), "Initial Kernel Execution State", and the
code that implements it, LLVM SIISelLowering.cpp (allocateSpecialEntryInputVGPRs,
allocateSystemSGPRs, getPreloadedValue). gfx1201 is FeatureISAVersion12 (GCNFeatures.td), which
has FeaturePackedTID, FeatureArchitectedFlatScratch and FeatureArchitectedSGPRs. Defined at entry:

  * User SGPRs s0..s(N-1). They are dense in the order private segment buffer (4), dispatch ptr
    (2), queue ptr (2), kernarg segment ptr (2), dispatch id (2), flat scratch init (2), private
    segment size (1), preloaded kernargs (KERNARG_PRELOAD_SPEC_LENGTH SGPRs); N is the larger of
    COMPUTE_PGM_RSRC2.USER_SGPR_COUNT and that sum (kernel_code_properties bits 0-6).
  * System SGPRs right after them: work-group id X/Y/Z (RSRC2 bits 7-9), work-group info (bit 10),
    private segment wave offset (bit 0). With FeatureArchitectedSGPRs LLVM reads the work-group ids
    from TTMP9 / TTMP7 (getPreloadedValue) and assigns them no SGPR (allocateSystemSGPRs). Whether
    the hardware still writes them into SGPRs when the enable bits are set is not documented, so
    every enabled system SGPR counts as defined: no finding rather than a guessed one.
  * TTMP0-15: trap temporaries and, on gfx12, the architected work-group ids (TTMP7, TTMP9) and
    wave id (TTMP8[29:25], SIISelLowering). The program does not own them; never reported.
  * v0 only. Packed work-item IDs (FeaturePackedTID, AMDGPUUsage "Register Layout for Packed
    Work-Item ID Method"): X, Y and Z are v0 bits 9:0, 19:10, 29:20; v1 and v2 are not set up
    whatever ENABLE_VGPR_WORKITEM_ID says (allocateSpecialEntryInputVGPRs).
  * EXEC (both halves).

Not defined: other SGPRs, VCC, M0, SCC, all other VGPRs. What the hardware leaves in them is not
documented: the program computes with a value nobody defined ('review' or 'info', below).

The analysis runs two forward problems over the direct-branch CFG (`rules.cfg`), on 32-bit
registers, using the operand model of `isadb.operand_model` (explicit and implicit operands): the
registers defined on EVERY path to an instruction (must) and those defined on SOME path (may).
What is reported, and how:
  * 'review': a read of a register that no path from the entry writes (undefined on EVERY path),
    after the exemptions below. Not seen in clang 18 output of well-defined HIP.
  * 'info': a read of a register written on some paths to it but not on all (maybe undefined).
    Machine code cannot tell a bug from LLVM's phi with an undef input (`; implicit-def: $vgpr3`
    in the asm): a value set under a condition and read under the same condition, a value carried
    round a loop back edge, the lane-mask phis of SILowerI1Copies.cpp (insertUndefLaneMask emits
    the IMPLICIT_DEF; buildMergeLaneMasks merges it with S_AND_NOT1/S_AND/S_OR/S_OR_NOT1 and EXEC).
    LLVM selects (v_cndmask, s_cselect), merges or even computes speculatively with such a value
    (v_add_co_u32, v_cmp_lt_f64, v_mul_lo_u32 in clang 18 -O2/-O3 output of well-defined HIP) and
    drops the result where it is undefined. So such a read is only noted; a missing write on one
    branch of a uniform if, or a loop-carried register never written before the first iteration,
    is found at this level only.
  * A partial write defines the whole register: a D16 load or a 16-bit (true16 `.l`/`.h`) result
    writes one half and keeps the other. The AMD XML lists such destinations, like every tied one
    (V_FMAC* accumulators, the DPP "old" value), as output only, so the kept half is not counted
    as read either: LLVM may leave it undef on purpose. A register range (v[2:3]) defines each
    register it names; VCC and EXEC are their halves (vcc_lo/vcc_hi = s106/s107). S_CMOV_B32/B64
    and S_CMOVK_I32 write only when SCC=1, so they also read their destination (the kept value).
  * VGPRs are per lane. Of the two successors of s_cbranch_execz / s_cbranch_execnz, the one
    taken only when EXEC is zero (the target of execz, the fall-through of execnz) carries no VGPR
    state: no lane runs or skips anything along it. Without this every divergent if/else would be
    a path past both writes (LLVM's `implicit-def` phis). Except when that edge leaves a loop: the
    branch lies in the layout range [target, source] of a backward branch and the zero-EXEC
    successor outside it (an s_cbranch_execnz latch, whose fall-through is the exit; an
    s_cbranch_execz to the loop exit). EXEC is zero there because every lane has left the loop;
    EXEC is restored right after, and the lanes go on with the VGPRs they had in the loop, so the
    edge carries the loop's state. Loops are taken from the layout: the range between a backward
    branch and its target (in clang 18 output a loop is contiguous, its rotated header may sit
    after a flow block; LLVM's MachineBlockPlacement may move cold blocks out of it, which this
    does not see: a known gap). Writes under a partial EXEC count as full definitions: lane masks
    are not tracked, so a lane that never ran the write is not found.
  * A multi-register operand (v[1:4], s[4:7]) of which at least one register is defined is not
    reported for the others when it is data: LLVM leaves unused elements of a vector or a wide
    value undef (a float4 store with two elements never set, seen in clang 18 output). The
    excuse does not apply to memory addresses and descriptors (every SGPR tuple of a VMEM/SMEM
    instruction: SADDR, SBASE, SRSRC, SSAMP; the VGPR address of FLAT/GLOBAL/SCRATCH/BUFFER/IMAGE
    /SAMPLE) nor to the sources of an F64 VALU operation, whose high half is always consumed. It
    still applies to 64-bit integer and lane-mask operands (an any-extended i64 whose high half
    is undef and unused).
  * A call or indirect jump (s_swappc_b64, s_call_b64, s_setpc_b64, s_rfe_b64) defines every
    register afterwards: the callee's effect is not visible.
  * An instruction outside the operand table defines every register it names; its reads are a
    guess and are not reported.
  * Reads whose result does not depend on the value (SAME_SOURCE, SELF_MOVE): x^x and x-x on
    integers, and a move of a register onto itself (GCNHazardRecognizer::fixVcmpxPermlaneHazards
    emits `v_mov_b32 vN, vN` with an undef source); a self-move is neither read nor definition.
    Not with DPP, op_sel or SDWA: a source then comes from another lane or another half.
  * The implicit M0 operand the XML gives s_sendmsg/s_sendmsghalt is not counted: LLVM sends
    MSG_DEALLOC_VGPRS without setting M0 (clang 18 output); which messages read M0 is not
    documented in the sources used here.

Each exemption above is needed for clang 18 -O0..-O3 output of tests/test_liveness.py (wave32)
to stay clean; LLVM 20-24 output was not available here. One finding per register and severity,
at its first read in pc order; `later_uses` lists up to MAX_LATER further reads and
`later_use_count` counts them all.

Skipped: called functions (`kernel['callee']`) and snippets without a kernel descriptor, unless
`options['liveness_entry']` lists the registers defined at the snippet's entry (EXEC is always
defined). `options['liveness'] = False` turns the rule off.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from collections import deque

from . import isadb
from .binary import FormatError

RULE = 'UNINITIALIZED_READ'
SOURCE = ('LLVM AMDGPU User Guide (AMDGPUUsage), Initial Kernel Execution State: SGPR Register Set Up '
          'Order, Packed Work-Item ID Method, Preloaded Kernel Arguments; LLVM SIISelLowering.cpp '
          'allocateSpecialEntryInputVGPRs, allocateSystemSGPRs, getPreloadedValue (gfx1201: '
          'FeaturePackedTID, FeatureArchitectedSGPRs in GCNFeatures.td FeatureISAVersion12)')
MERGE_SOURCE = ('; LLVM SILowerI1Copies.cpp buildMergeLaneMasks (AndN2Opc/AndOpc/OrOpc/OrN2Opc with ExecReg), '
                'insertUndefLaneMask (IMPLICIT_DEF)')

# Bit numbers of the register sets: SGPR n -> n, SCC -> 128, VGPR n -> 256 + n.
SCC_BIT = 128
VBASE = 256
NV = 1024
ALL = (1 << (VBASE + NV)) - 1
VMASK = ALL & ~((1 << VBASE) - 1)
TTMP = sum(1 << n for n in range(108, 124))
EXEC = (1 << 126) | (1 << 127)
M0 = 1 << 125

EXEC_BRANCHES = ('s_cbranch_execz', 's_cbranch_execnz')
OPAQUE = ('s_swappc_b64', 's_call_b64', 's_setpc_b64', 's_rfe_b64')
SENDMSG = ('S_SENDMSG', 'S_SENDMSGHALT')
# Integer ops whose result is the same for any value when both sources are one register.
SAME_SOURCE = {'V_XOR_B32', 'S_XOR_B32', 'S_XOR_B64', 'V_SUB_NC_U32', 'V_SUBREV_NC_U32',
               'V_SUB_CO_U32', 'V_SUBREV_CO_U32', 'S_SUB_U32', 'S_SUB_I32', 'S_SUB_CO_U32',
               'V_SUB_NC_U16', 'V_XOR_B16'}
SELF_MOVE = {'V_MOV_B32', 'S_MOV_B32', 'S_MOV_B64', 'V_MOV_B16'}
# Modifiers under which a source comes from another lane or another half (no SAME_SOURCE idiom).
LANE_MODIFIERS = re.compile(r'\b(?:op_sel|op_sel_hi|row_\w+|quad_perm|dpp8|row_mask|bank_mask|wave_\w+|'
                            r'dst_sel|src0_sel|src1_sel|fi)\s*:|\.(?:h|l)\b')
# Conditional moves: write only when SCC=1, else keep the destination (which is therefore read).
CMOVES = {'S_CMOV_B32', 'S_CMOV_B64', 'S_CMOVK_I32'}
# Encodings whose VGPR address is the operand after VDATA/VDST (else after a returned VDST).
VADDR_SECOND = ('ENC_VBUFFER', 'ENC_VIMAGE', 'ENC_VSAMPLE')
FLAT_ENCS = ('ENC_VGLOBAL', 'ENC_VFLAT', 'ENC_VSCRATCH')
MAX_LATER = 32
# User SGPR enables, kernel_code_properties bits 0-6 (AMDHSAKernelDescriptor.h), and their sizes.
USER_SGPRS = ((0, 4, 'private segment buffer'), (1, 2, 'dispatch ptr'), (2, 2, 'queue ptr'),
              (3, 2, 'kernarg segment ptr'), (4, 2, 'dispatch id'), (5, 2, 'flat scratch init'),
              (6, 1, 'private segment size'))
# System SGPR enables in COMPUTE_PGM_RSRC2, in set-up order.
SYSTEM_SGPRS = ((7, 'work-group id X'), (8, 'work-group id Y'), (9, 'work-group id Z'),
                (10, 'work-group info'), (0, 'private segment wave offset'))


def bit(reg: tuple[str, int]) -> int:
    kind, n = reg
    if kind == 's':
        return n
    if kind == 'scc':
        return SCC_BIT
    return VBASE + n


def mask(regs) -> int:
    m = 0
    for r in regs:
        if r[0] == 'v' and r[1] >= NV:
            continue
        m |= 1 << bit(r)
    return m


def name(b: int) -> str:
    if b >= VBASE:
        return f'v{b - VBASE}'
    if b == SCC_BIT:
        return 'scc'
    return {106: 'vcc_lo', 107: 'vcc_hi', 125: 'm0', 126: 'exec_lo', 127: 'exec_hi'}.get(b, f's{b}')


def bits(m: int) -> list[int]:
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


def entry_state(descriptor: dict, packed_tid: bool = True) -> tuple[int, dict]:
    """(defined-at-entry mask, explanation) from a kernel descriptor (`binary.kernel_inventory`)."""
    props = descriptor.get('properties', 0)
    rsrc2 = descriptor.get('rsrc2', 0)
    preload = descriptor.get('kernarg_preload', 0) & 0x7f
    enabled = [(what, size) for b, size, what in USER_SGPRS if props >> b & 1]
    if preload:
        enabled.append(('preloaded kernargs', preload))
    user = max((rsrc2 >> 1) & 0x1f, sum(s for _, s in enabled))
    system = [what for b, what in SYSTEM_SGPRS if rsrc2 >> b & 1]
    n_sgpr = user + len(system)
    tid = (rsrc2 >> 11) & 3
    vgprs = [0] if packed_tid else list(range(tid + 1))
    n_sgpr = min(n_sgpr, 106)
    m = EXEC | TTMP | sum(1 << n for n in range(n_sgpr))
    m |= sum(1 << (VBASE + v) for v in vgprs)
    sg = [f's0-s{n_sgpr - 1}' if n_sgpr > 1 else 's0'] if n_sgpr else []
    why = {'user_sgprs': user, 'user_sgpr_fields': [w for w, _ in enabled], 'system_sgprs': system,
           'vgprs': [f'v{v}' for v in vgprs], 'workitem_id_fields': tid, 'packed_tid': packed_tid,
           'defined': ', '.join(sg + [f'v{v}' for v in vgprs] + ['EXEC', 'TTMP0-15'])}
    return m, why


def _same_sources(positional: list[str], n: int = 2) -> bool:
    srcs = positional[-n:]
    return len(positional) > n and len(set(srcs)) == 1 and bool(isadb.REG_TOKEN.search(srcs[0]))


def _positional(ins: dict) -> list[str]:
    try:
        return [p.strip() for p in isadb.split_operands(ins['args'])]
    except (FormatError, ValueError):
        return []


def _regs(token: str) -> int:
    try:
        return mask(isadb.registers(token))
    except FormatError:
        return 0


def access(ins: dict, model: dict) -> tuple[int, int, bool]:
    """(read mask, write mask, reads reported) of one instruction."""
    if not model.get('known'):
        return 0, mask(model['writes']), False
    reads, writes = mask(model['reads']), mask(model['writes'])
    name_ = model.get('name') or ''
    if name_ in SENDMSG:
        reads &= ~M0
    if name_ in CMOVES:
        reads |= writes & ~(1 << SCC_BIT)
    if name_ in SAME_SOURCE or name_ in SELF_MOVE:
        pos = _positional(ins)
        if 'dpp' in ins['op'] or 'sdwa' in ins['op'] or LANE_MODIFIERS.search(ins['args']):
            return reads, writes, True
        if name_ in SELF_MOVE and len(pos) == 2 and pos[0] == pos[1]:
            return 0, 0, True
        if name_ in SAME_SOURCE and _same_sources(pos):
            reads &= ~_regs(pos[-1])
    return reads, writes, True


def _taken(ins: dict) -> int | None:
    """Target of a SOPP branch (signed DWORD displacement from the next instruction)."""
    raw = bytes.fromhex(ins['raw'])
    if ins['size'] != 4 or len(raw) < 2:
        return None
    return ins['pc'] + 4 + 4 * int.from_bytes(raw[:2], 'little', signed=True)


def _zero_edge(ins: dict) -> int | None:
    """The successor of an EXEC branch taken only when EXEC is zero: the target of
    s_cbranch_execz, the fall-through of s_cbranch_execnz."""
    taken = _taken(ins)
    if taken is None:
        return None
    return taken if ins['op'] == 's_cbranch_execz' else ins['pc'] + ins['size']


def _loops(edges: dict) -> list[tuple[int, int]]:
    """Layout ranges (target, source) of the backward branches, sorted by target."""
    return sorted((s, pc) for pc, ss in edges.items() for s in ss if s <= pc)


def _leaves_loop(pc: int, succ: int, loops: list[tuple[int, int]], starts: list[int]) -> bool:
    """Whether the edge pc -> succ leaves the layout range of a loop around pc."""
    for lo, hi in loops[:bisect_right(starts, pc)]:
        if hi >= pc and not lo <= succ <= hi:
            return True
    return False


def _addresses(ins: dict, model: dict, pos: list[str]) -> set[int]:
    """Positions of the operands of a memory instruction that hold an address or a descriptor."""
    entry = model.get('entry') or {}
    group, enc = entry.get('group'), model.get('enc')
    if group not in ('VMEM', 'SMEM'):
        return set()
    out = {k for k, t in enumerate(pos) if _regs(t) and _regs(t) < (1 << VBASE)}   # SGPR operands
    if enc in VADDR_SECOND and len(pos) > 1:
        out.add(1)
    elif enc in FLAT_ENCS and pos:
        first = _regs(pos[0])
        out.add(1 if first and first & ~mask(model['writes']) == 0 else 0)
    return out


def _tuple_excuse(ins: dict, model: dict, defined: int) -> int:
    """Registers of a multi-register data operand (v[1:4], s[4:7]) of which some other register is
    defined: LLVM leaves unused elements of a vector or a wide value undef. Not addresses,
    descriptors or F64 sources (module docstring)."""
    if 'F64' in (model.get('name') or ''):
        return 0
    pos = _positional(ins)
    skip = _addresses(ins, model, pos)
    addresses = 0
    for k in skip:
        addresses |= _regs(pos[k])
    out = 0
    for k, t in enumerate(pos):
        if '[' not in t or k in skip:
            continue
        m = _regs(t)
        if m & (m - 1) and m & defined:
            out |= m & ~defined
    return out & ~addresses


def solve(instructions: list[dict], edges: dict, models: dict, entry: int) -> tuple[dict, dict, dict]:
    """pc -> registers defined on every path to it (must mask), pc -> registers defined on some
    path to it (may mask), and pc -> (reads, writes, reported)."""
    acc = {i['pc']: access(i, models[i['pc']]) for i in instructions}
    by_pc = {i['pc']: i for i in instructions}
    loops = _loops(edges)
    starts = [lo for lo, _ in loops]
    zero = {}
    for i in instructions:
        if i['op'] in EXEC_BRANCHES:
            z = _zero_edge(i)
            if z is not None and not _leaves_loop(i['pc'], z, loops, starts):
                zero[i['pc']] = z
    start = instructions[0]['pc']
    must, may = {start: entry}, {start: entry}
    work = deque([start])
    queued = {start}
    while work:
        pc = work.popleft()
        queued.discard(pc)
        i = by_pc[pc]
        if i['op'] in OPAQUE:
            out_must = out_may = ALL
        else:
            out_must, out_may = must[pc] | acc[pc][1], may[pc] | acc[pc][1]
        for s in edges.get(pc, []):
            o_must, o_may = out_must, out_may
            if zero.get(pc) == s:     # no lane travels this edge: no VGPR state along it
                o_must, o_may = o_must | VMASK, o_may & ~VMASK
            old = must.get(s)
            new_must = o_must if old is None else old & o_must
            new_may = may.get(s, 0) | o_may
            if old is None or new_must != old or new_may != may[s]:
                must[s], may[s] = new_must, new_may
                if s not in queued:
                    queued.add(s)
                    work.append(s)
    return must, may, acc


def _message(i: dict, regs: list[str], kind: str, hint: dict) -> str:
    where = ('on every path from the kernel entry (no path to it writes it)' if kind == 'never' else
             'on at least one path from the kernel entry, but other paths write it')
    msg = (f"{i['op']} reads {', '.join(regs)} before any instruction writes it {where}, and the entry state "
           f"does not define it (defined at entry: {hint.get('defined')}). The AMDHSA ABI gives such a "
           'register no value at launch')
    if kind == 'some':
        msg += ('. Noted only: LLVM gives a phi an undef input on the edge where the value is not set '
                '(`; implicit-def` in the asm: a value set under a condition, carried round a loop back edge, '
                'or the lane-mask phis of SILowerI1Copies) and may select, merge or compute speculatively with '
                'it, dropping the result on the paths where it is undefined; the machine code does not show '
                'whether a later use depends on it')
    if any(r in ('v1', 'v2') for r in regs) and hint.get('packed_tid'):
        msg += ('. gfx1201 packs the work-item IDs into v0 (X bits 9:0, Y 19:10, Z 29:20); v1 and v2 are '
                'not set up at launch')
    return msg


def analyze(instructions: list[dict], edges: dict, reachable: set, models: dict, entry: int,
            hint: dict | None = None) -> list[dict]:
    """UNINITIALIZED_READ findings: one per register and severity, at its first read in pc order."""
    if not instructions:
        return []
    must, may, acc = solve(instructions, edges, models, entry)
    hint = hint or {}
    out, seen = [], {}
    for i in instructions:
        pc = i['pc']
        if pc not in reachable or pc not in must:
            continue
        reads, _, report = acc[pc]
        undef = reads & ~must[pc]
        if not undef or not report:
            continue
        undef &= ~_tuple_excuse(i, models[pc], must[pc])
        if not undef:
            continue
        for kind, severity, m in (('never', 'review', undef & ~may[pc]), ('some', 'info', undef & may[pc])):
            fresh = 0
            for b in bits(m):
                f = seen.get((b, severity))
                if f is None:
                    fresh |= 1 << b
                elif f['pc'] != pc and pc not in f['_later']:
                    f['_later'].add(pc)
                    f['later_use_count'] += 1
                    if len(f['later_uses']) < MAX_LATER:
                        f['later_uses'].append(pc)
            if not fresh:
                continue
            regs = [name(b) for b in bits(fresh)]
            f = {'rule': RULE, 'severity': severity, 'pc': pc, 'message': _message(i, regs, kind, hint),
                 'registers': regs, 'undefined_on': 'every path' if kind == 'never' else 'some paths',
                 'later_uses': [], 'later_use_count': 0, '_later': set(), 'entry_state': hint,
                 'source': SOURCE + (MERGE_SOURCE if kind == 'some' else '')}
            for b in bits(fresh):
                seen[(b, severity)] = f
            out.append(f)
    for f in out:
        del f['_later']
    return out


def kernel_pass(ctx: dict) -> list[dict]:
    """check.KERNEL_PASSES entry: UNINITIALIZED_READ for one kernel (see the module docstring)."""
    opts = ctx.get('options') or {}
    kernel = ctx['kernel']
    if opts.get('liveness') is False or kernel.get('callee'):
        return []
    desc = kernel.get('descriptor') or {}
    if desc:
        entry, hint = entry_state(desc)
    elif opts.get('liveness_entry') is not None:
        given = list(opts['liveness_entry'])
        entry = EXEC | TTMP
        for text in given:
            entry |= mask(isadb.registers(text))
        hint = {'snippet': True, 'given': given, 'defined': ', '.join(given + ['EXEC', 'TTMP0-15'])}
    else:
        return []
    table = ctx.get('table') or {}
    out = analyze(ctx['instructions'], ctx['edges'], ctx['reachable'], ctx['models'], entry, hint)
    if table:
        cite = '; '.join(isadb.cite(table, s) for s in ('3.5.3', '3.5.4'))
        for f in out:
            f['source'] = f"{f['source']}; {cite}"
    return out
