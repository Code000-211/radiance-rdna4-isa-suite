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
documented, so a finding is 'review': the program computes with a value nobody defined.

The analysis is a forward must-analysis over the direct-branch CFG (`rules.cfg`), on 32-bit
registers, using the operand model of `isadb.operand_model` (explicit and implicit operands):
  * A partial write defines the whole register: a D16 load or a 16-bit (true16 `.l`/`.h`) result
    writes one half and keeps the other. The AMD XML lists such destinations, like every tied one
    (V_FMAC* accumulators, the DPP "old" value), as output only, so the kept half is not counted
    as read either: LLVM may leave it undef on purpose. A register range (v[2:3]) defines each
    register it names; VCC and EXEC are their halves (vcc_lo/vcc_hi = s106/s107).
  * VGPRs are per lane. Of the two successors of s_cbranch_execz / s_cbranch_execnz, the one taken
    only when EXEC is zero (the target of execz, the fall-through of execnz) carries SGPRs only: no
    lane runs or skips anything along it. Without this every divergent if/else would be a path
    past both writes (LLVM's `implicit-def` phis). Writes under a partial EXEC count as full
    definitions: lane masks are not tracked, so a lane that never ran the write is not found.
  * A multi-register operand (v[1:4], s[4:7]) of which at least one register is defined is not
    reported for the others: LLVM leaves unused elements of a vector or a wide value undef (a
    float4 store with two elements never set, seen in clang 18 output).
  * A call or indirect jump (s_swappc_b64, s_call_b64, s_setpc_b64, s_rfe_b64) defines every
    register afterwards: the callee's effect is not visible.
  * An instruction outside the operand table defines every register it names; its reads are a
    guess and are not reported.
  * Reads whose result does not depend on the value (SAME_SOURCE, SELF_MOVE): x^x and x-x on
    integers, and a move of a register onto itself (GCNHazardRecognizer::fixVcmpxPermlaneHazards
    emits `v_mov_b32 vN, vN` with an undef source); a self-move is neither read nor definition.
  * The implicit M0 operand the XML gives s_sendmsg/s_sendmsghalt is not counted: LLVM sends
    MSG_DEALLOC_VGPRS without setting M0 (clang 18 output); which messages read M0 is not
    documented in the sources used here.

Each exemption above is needed for clang 18 -O1/-O3 output of tests/test_liveness.py to stay clean.

Skipped: called functions (`kernel['callee']`) and snippets without a kernel descriptor, unless
`options['liveness_entry']` lists the registers defined at the snippet's entry (EXEC is always
defined). `options['liveness'] = False` turns the rule off.
"""
from __future__ import annotations

from collections import deque

from . import isadb
from .binary import FormatError

RULE = 'UNINITIALIZED_READ'
SOURCE = ('LLVM AMDGPU User Guide (AMDGPUUsage), Initial Kernel Execution State: SGPR Register Set Up '
          'Order, Packed Work-Item ID Method, Preloaded Kernel Arguments; LLVM SIISelLowering.cpp '
          'allocateSpecialEntryInputVGPRs, allocateSystemSGPRs, getPreloadedValue (gfx1201: '
          'FeaturePackedTID, FeatureArchitectedSGPRs in GCNFeatures.td FeatureISAVersion12)')

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
    out, b = [], 0
    while m:
        if m & 1:
            out.append(b)
        m >>= 1
        b += 1
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


def access(ins: dict, model: dict) -> tuple[int, int, bool]:
    """(read mask, write mask, reads reported) of one instruction."""
    if not model.get('known'):
        return 0, mask(model['writes']), False
    reads, writes = mask(model['reads']), mask(model['writes'])
    name_ = model.get('name') or ''
    if name_ in SENDMSG:
        reads &= ~M0
    if name_ in SAME_SOURCE or name_ in SELF_MOVE:
        try:
            pos = isadb.split_operands(ins['args'])
        except (FormatError, ValueError):
            pos = []
        pos = [p.strip() for p in pos]
        if name_ in SELF_MOVE and len(pos) == 2 and pos[0] == pos[1]:
            return 0, 0, True
        if name_ in SAME_SOURCE and _same_sources(pos):
            regs = mask(isadb.registers(pos[-1]))
            reads &= ~regs
    return reads, writes, True


def _taken(ins: dict) -> int | None:
    """Target of a SOPP branch (signed DWORD displacement from the next instruction)."""
    raw = bytes.fromhex(ins['raw'])
    if ins['size'] != 4 or len(raw) < 2:
        return None
    return ins['pc'] + 4 + 4 * int.from_bytes(raw[:2], 'little', signed=True)


def _lane_edge(ins: dict) -> int | None:
    """The successor of an EXEC branch that lanes reach: the fall-through of s_cbranch_execz,
    the target of s_cbranch_execnz. The other one is taken only when EXEC is zero."""
    taken = _taken(ins)
    if taken is None:
        return None
    return ins['pc'] + ins['size'] if ins['op'] == 's_cbranch_execz' else taken


def _tuple_excuse(ins: dict, defined: int) -> int:
    """Registers of a multi-register operand (v[1:4], s[4:7]) of which some other register is
    defined: LLVM leaves unused elements of a vector or a wide value undef."""
    try:
        tokens = isadb.split_operands(ins['args'])
    except (FormatError, ValueError):
        return 0
    out = 0
    for t in tokens:
        if '[' not in t:
            continue
        try:
            m = mask(isadb.registers(t))
        except FormatError:
            continue
        if m & (m - 1) and m & defined:
            out |= m & ~defined
    return out


def solve(instructions: list[dict], edges: dict, models: dict, entry: int) -> tuple[dict, dict]:
    """pc -> registers defined on every path to it (mask), and pc -> (reads, writes, reported)."""
    acc = {i['pc']: access(i, models[i['pc']]) for i in instructions}
    by_pc = {i['pc']: i for i in instructions}
    start = instructions[0]['pc']
    state = {start: entry}
    work = deque([start])
    queued = {start}
    while work:
        pc = work.popleft()
        queued.discard(pc)
        i = by_pc[pc]
        out = ALL if i['op'] in OPAQUE else state[pc] | acc[pc][1]
        lanes = _lane_edge(i) if i['op'] in EXEC_BRANCHES else None
        for s in edges.get(pc, []):
            o = out | VMASK if lanes is not None and s != lanes else out
            old = state.get(s)
            new = o if old is None else old & o
            if old is None or new != old:
                state[s] = new
                if s not in queued:
                    queued.add(s)
                    work.append(s)
    return state, acc


def analyze(instructions: list[dict], edges: dict, reachable: set, models: dict, entry: int,
            hint: dict | None = None) -> list[dict]:
    """UNINITIALIZED_READ findings: one per register, at its first read in pc order."""
    if not instructions:
        return []
    state, acc = solve(instructions, edges, models, entry)
    hint = hint or {}
    out, seen = [], {}
    for i in instructions:
        pc = i['pc']
        if pc not in reachable or pc not in state:
            continue
        reads, _, report = acc[pc]
        undef = reads & ~state[pc]
        if not undef or not report:
            continue
        undef &= ~_tuple_excuse(i, state[pc])
        if not undef:
            continue
        fresh = 0
        for b in bits(undef):
            if b in seen:
                if pc not in seen[b]['later_uses'] and seen[b]['pc'] != pc:
                    seen[b]['later_uses'].append(pc)
            else:
                fresh |= 1 << b
        if not fresh:
            continue
        regs = [name(b) for b in bits(fresh)]
        msg = (f"{i['op']} reads {', '.join(regs)} before any instruction writes it on at least one path from "
               f"the kernel entry, and the entry state does not define it (defined at entry: {hint.get('defined')}). "
               'The AMDHSA ABI gives such a register no value at launch')
        if any(r in ('v1', 'v2') for r in regs) and hint.get('packed_tid'):
            msg += ('. gfx1201 packs the work-item IDs into v0 (X bits 9:0, Y 19:10, Z 29:20); v1 and v2 are '
                    'not set up at launch')
        f = {'rule': RULE, 'severity': 'review', 'pc': pc, 'message': msg, 'registers': regs,
             'later_uses': [], 'entry_state': hint, 'source': SOURCE}
        for b in bits(fresh):
            seen[b] = f
        out.append(f)
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
            f['source'] = f"{SOURCE}; {cite}"
    return out
