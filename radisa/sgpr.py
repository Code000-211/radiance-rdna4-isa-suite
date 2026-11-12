"""SGPR read hazard of GFX12: missing `s_wait_alu`.

A VALU instruction that reads an SGPR makes the hardware treat that SGPR pair as slow to
commit; from then on a read of a register of the pair has to wait until the last write to it
has been committed: `s_wait_alu depctr_sa_sdst(0)` after a scalar write,
`depctr_va_sdst(0)` after a vector write that a vector instruction reads, and for VCC
`depctr_sa_sdst(0)` / `depctr_va_vcc(0)`.

AMD's ISA document does not describe this; the rule is the one the compiler implements, and
this module is the same forward dataflow as LLVM's AMDGPUWaitSGPRHazards pass, run over the
finished machine code: where the pass would still insert a wait, the wait is missing.

The pass numbers every ttmp register as SGPR 0 (SIRegisterInfo.td gives the generic TTMPn the
hardware encoding 0; the real 108+n is chosen when the instruction is encoded). A scalar read
of a ttmp register therefore counts for it as a read of s0, which ends the pending vector
writes, and it emits no wait where the real register numbers ask for one. The analysis runs
with both numberings: a wait both ask for is missing (error); a wait only the real numbers ask
for is what the compiler leaves out, reported for review as ALU_WAIT_TTMP.
"""
from __future__ import annotations

from . import isadb

SOURCE = ('LLVM AMDGPUWaitSGPRHazards.cpp (VALU SGPR read hazard of GFX12); '
          's_wait_alu fields: AMD RDNA4 ISA S_WAITCNT_DEPCTR')
SCALAR = ('ENC_SOP1', 'ENC_SOP2', 'ENC_SOPC', 'ENC_SOPK', 'ENC_SOPP', 'SOP2_INST_LITERAL', 'SOPK_INST_LITERAL')
CLEARING = ('ENC_SMEM', 'ENC_VBUFFER', 'ENC_VIMAGE', 'ENC_VSAMPLE')   # LLVM: VMEM that is not FLAT, or SMEM
SET_PC = ('s_swappc_b64', 's_setpc_b64', 's_call_b64', 's_rfe_b64')
CALLS = ('s_swappc_b64', 's_call_b64')
VCC = (106, 107)
WA_VALU, WA_SALU, WA_VCC = 1, 2, 4
H_SALU, H_VALU = 1, 2
ALL_PAIRS = (1 << 64) - 1
FIELD = {WA_SALU: 'depctr_sa_sdst(0)', WA_VALU: 'depctr_va_sdst(0)', WA_VCC: 'depctr_va_vcc(0)'}
TTMP0 = 108


def classify(ins: dict, model: dict, ttmp_as_s0: bool = False,
             jump: bool = False) -> tuple[str, list[int], list[int]]:
    """('valu' | 'salu' | 'clear' | 'wait' | 'setpc' | 'call' | '', SGPR uses, SGPR defs).

    `jump`: the instruction is an s_setpc_b64 that rules.cfg resolves to a target inside the
    function (a long branch, rules.long_branch_target). LLVM's pass never sees it: BranchRelaxation
    builds the s_getpc/s_add/s_setpc sequence after the hazard passes and flushes the SGPR writes
    it adds itself with s_wait_alu depctr_sa_sdst(0) (SIInstrInfo::insertIndirectBranch,
    ApplyHazardWorkarounds). It is a scalar read of its address pair followed by a jump, so the
    hazard state flows on to the target as over s_branch."""
    def _sgprs(regs) -> list[int]:
        # The registers the hazard applies to: not null, m0, exec, SCC.
        return sorted({0 if ttmp_as_s0 and n >= TTMP0 else n for kind, n in regs if kind == 's' and n <= 123})
    op = ins['op']
    if op == 's_wait_alu':
        return 'wait', [], []
    if ins.get('vopd_independent_decode') or op.startswith('v_dual_'):
        uses = set(isadb.registers(ins['args']))
        if 'v_dual_cndmask' in ins['op'] + ins['args']:
            uses.add(('s', VCC[0]))                  # the select mask of the VOPD form is VCC
        return 'valu', _sgprs(uses), []
    if not model.get('known'):
        kind = 'valu' if op.startswith('v_') else 'salu' if op.startswith('s_') else ''
        return kind, _sgprs(model['reads']), _sgprs(model['writes'])
    enc = model['enc']
    if enc in CLEARING:
        return 'clear', [], []
    if op == 's_setpc_b64' and jump:
        return 'salu', _sgprs(model['reads']), []
    if op in SET_PC:
        return ('call' if op in CALLS else 'setpc'), [], _sgprs(model['writes'])
    if enc in SCALAR:
        return 'salu', _sgprs(model['reads']), _sgprs(model['writes'])
    if model['entry']['group'] == 'VALU':
        return 'valu', _sgprs(model['reads']), _sgprs(model['writes'])
    return '', [], []


def wait_fields(ins: dict) -> tuple[bool, bool, bool]:
    """(sa_sdst, va_sdst, va_vcc) is zero in the S_WAITCNT_DEPCTR immediate."""
    imm = int.from_bytes(bytes.fromhex(ins['raw'])[:2], 'little')
    return not imm & 1, not (imm >> 9) & 7, not (imm >> 1) & 1


def step(state: tuple, kind: str, uses: list[int], defs: list[int], fields=None) -> tuple[tuple, int, list[int]]:
    """One instruction of the LLVM pass: (state after it, wait it needs, registers that need it)."""
    tracked, salu, valu, vcc = state
    if kind == 'clear':
        return (tracked, 0, 0, 0), 0, []
    if kind == 'wait':
        sa, va, vc = fields
        if vc: vcc &= ~H_VALU
        if sa: salu, vcc = 0, vcc & ~H_SALU
        if va: valu = 0
        return (tracked, salu, valu, vcc), 0, []
    if not kind:
        return state, 0, []
    is_valu, wait, why = kind == 'valu', 0, []
    if kind in ('setpc', 'call'):
        if vcc & H_VALU: wait |= WA_VCC
        if salu or vcc & H_SALU: wait |= WA_SALU
        if valu: wait |= WA_VALU
    else:
        for n in uses:
            pair = 1 << (n >> 1)
            if not tracked & pair:
                if is_valu: tracked |= pair
                continue
            w = 0
            if not is_valu:
                if n in VCC:
                    if vcc & H_VALU: vcc = 0
                else:
                    valu = 0
            if salu >> n & 1: w |= WA_SALU
            if is_valu and valu >> n & 1: w |= WA_VALU
            if n in VCC and vcc:
                if vcc & H_SALU: w |= WA_SALU
                if vcc & H_VALU: w |= WA_VCC
            if w:
                wait |= w
                why.append(n)
    if wait & WA_VCC: vcc &= ~H_VALU
    if wait & WA_SALU: salu, vcc = 0, vcc & ~H_SALU
    if wait & WA_VALU: valu = 0
    if kind == 'call':
        tracked = ALL_PAIRS
    for n in defs:
        if not tracked >> (n >> 1) & 1:
            continue
        if n in VCC:
            vcc = H_VALU if is_valu else H_SALU
        elif is_valu:
            valu |= 1 << n
        else:
            salu |= 1 << n
    return (tracked, salu, valu, vcc), wait, why


def _name(n: int) -> str:
    return {106: 'vcc_lo', 107: 'vcc_hi'}.get(n, f'ttmp{n - 108}' if n >= 108 else f's{n}')


def _fixpoint(entry: int, edges: dict, init, transfer, join) -> dict:
    """State before every instruction reached from `entry`: the least solution, joined where paths meet."""
    state_in = {entry: init}
    work, queued = [entry], {entry}
    while work:
        pc = work.pop()
        queued.discard(pc)
        state = state_in[pc]
        while True:
            out = transfer(pc, state)
            succ = edges.get(pc, [])
            follow = None
            for s in succ:
                old = state_in.get(s)
                new = out if old is None else join(old, out)
                if new == old:
                    continue
                state_in[s] = new
                if follow is None and len(succ) == 1:
                    follow = s                      # a straight line needs no worklist
                elif s not in queued:
                    queued.add(s)
                    work.append(s)
            if follow is None:
                break
            pc, state = follow, state_in[follow]
    return state_in


def _required(instructions, edges, reachable, models, callee, ttmp_as_s0, stats, wave=32) -> dict[int, tuple[int, list[int]]]:
    """pc -> (wait the instruction needs, registers that need it), for one register numbering.

    LLVM's pass is one dataflow over (tracked pairs, hazards) that is not monotone: a pair that
    becomes tracked later makes a scalar read end hazards that an earlier round let through, so
    what it leaves at a join depends on the order it visits the blocks in, and that order is not
    in the machine code. The tracked pairs alone are monotone, so their least solution is the
    one the pass ends with. With them fixed the hazards are monotone too, and their least
    solution is contained in whatever state the pass ends with: every wait found here is one the
    pass inserts, in any order.
    """
    cls = {}
    for i in instructions:
        kind, uses, defs = classify(i, models[i['pc']], ttmp_as_s0, bool(edges.get(i['pc'])))
        cls[i['pc']] = (kind, uses, defs, wait_fields(i) if kind == 'wait' else None)
        if stats is not None:
            stats['alu_waits'] += kind == 'wait'
    edges = {pc: succ for pc, succ in edges.items() if pc in reachable}
    entry = instructions[0]['pc']
    # Enough ds_nop in a row (4 for wave32, 8 for wave64) make the hardware forget every pair.
    forget, run = set(), 0
    for i in instructions:
        run = run + 1 if i['op'] == 'ds_nop' else 0
        if run >= (8 if wave == 64 else 4):
            forget.add(i['pc'])

    def track(pc, tracked):
        if pc in forget:
            return 0
        kind, uses, _, _ = cls[pc]
        if kind == 'call':
            return ALL_PAIRS
        if kind == 'valu':
            for n in uses:
                tracked |= 1 << (n >> 1)
        return tracked

    tracked_in = _fixpoint(entry, edges, ALL_PAIRS if callee else 0, track, lambda a, b: a | b)

    def hazards(pc, state):
        return step((tracked_in[pc],) + state, *cls[pc])[0][1:]

    hazard_in = _fixpoint(entry, edges, (0, 0, 0), hazards,
                          lambda a, b: (a[0] | b[0], a[1] | b[1], a[2] | b[2]))
    need = {}
    for i in instructions:
        pc = i['pc']
        if pc in hazard_in and pc in reachable:
            _, wait, why = step((tracked_in[pc],) + hazard_in[pc], *cls[pc])
            if wait:
                need[pc] = (wait, why)
    return need


def analyze(instructions: list[dict], edges: dict, reachable: set, models: dict, wave: int = 32,
            callee: bool = False) -> tuple[list[dict], dict]:
    """MISSING_ALU_WAIT findings. `callee`: a called function, every SGPR pair may be tracked."""
    stats = {'alu_waits': 0, 'alu_wait_required_at': 0, 'alu_wait_ttmp': 0}
    if not instructions:
        return [], stats
    hardware = _required(instructions, edges, reachable, models, callee, False, stats, wave)
    compiler = _required(instructions, edges, reachable, models, callee, True, None, wave)
    findings = []
    for i in instructions:
        pc = i['pc']
        if pc not in hardware:
            continue
        wait, why = hardware[pc]
        need = ' '.join(FIELD[b] for b in (WA_SALU, WA_VALU, WA_VCC) if wait & b)
        if i['op'] in SET_PC and not (i['op'] == 's_setpc_b64' and edges.get(pc)):
            what = f"{i['op']} leaves the function while SGPR writes are not committed"
        else:
            what = (f"{i['op']} reads {', '.join(_name(n) for n in why)} before the last write to it is committed "
                    '(the SGPR pair is read by vector instructions)')
        f = {'pc': pc, 'required_wait': f's_wait_alu {need}', 'registers': [_name(n) for n in why], 'source': SOURCE}
        if pc in compiler:
            stats['alu_wait_required_at'] += 1
            f.update(rule='MISSING_ALU_WAIT', severity='error',
                     message=f'{what}; insert s_wait_alu {need} before it')
        else:
            stats['alu_wait_ttmp'] += 1
            f.update(rule='ALU_WAIT_TTMP', severity='review',
                     message=f'{what}, and the compiler emits no s_wait_alu {need} here: a scalar read of a ttmp '
                             'register lies in between, which LLVM counts as a read of s0 that ends the pending '
                             'vector writes. Not confirmed on hardware; the wait before this instruction is the '
                             'safe form')
        findings.append(f)
    return findings, stats
