"""Bounded static checks, not a general proof of AMD ISA conformance.

Each finding cites the supplied 2025-04-07 RDNA4 guide. Exact encoding checks
are separated from local pattern diagnostics and performance observations.
"""
from __future__ import annotations
from collections import Counter, deque
import re
from .binary import FormatError
from . import hazards, isadb, vopd

SOURCES = {
    'resources': 'AMD RDNA4 ISA §3.3.2.1, printed p.17; LLVM AMDGPUUsage kernel descriptor',
    'vopd': 'AMD RDNA4 ISA §§7.8,15.3.7, printed pp.82–84,188–190',
    'flow': 'AMD RDNA4 ISA §§3.2.1,5.5, printed pp.14,47–48',
    'padding': 'AMD RDNA4 ISA §2.4, printed p.12',
    'dealloc': 'AMD RDNA4 ISA §§3.3.2.1,3.3.3.1,5.4, printed pp.17,19,47',
    'wmma': 'AMD RDNA4 ISA §7.12.1, printed pp.90–91',
    'wait': 'AMD RDNA4 ISA §5.7, printed pp.51–56',
    'permlane': 'AMD RDNA4 ISA §7.2.8, printed p.75',
    'index': 'AMD RDNA4 ISA §7.11, printed pp.88–89 (loop diagnosis is a heuristic)',
    'cache': 'AMD RDNA4 ISA §4.1.1, printed pp.39–42',
}


def finding(rule: str, severity: str, message: str, pc: int | None = None,
            source: str = 'flow', **extra) -> dict:
    return {'rule': rule, 'severity': severity, 'message': message, 'pc': pc,
            'source': SOURCES.get(source, source), **extra}


def occupancy(vgprs: int, wave: int, encoded_vgprs: int | None = None) -> dict:
    if type(vgprs) is not int or not 0 <= vgprs <= 256 or wave not in (32, 64):
        raise FormatError('invalid gfx1201 resource inputs')
    gran, capacity = (24, 1536) if wave == 32 else (12, 768)
    # At least one allocation block exists even for a kernel with zero live VGPRs.
    demand = max(1, vgprs, encoded_vgprs or 0)
    allocated = ((demand+gran-1)//gran)*gran
    return {'profile': 'gfx1201-static', 'logical_vgprs': vgprs,
            'descriptor_vgprs': encoded_vgprs, 'physical_allocation': allocated,
            'granule': gran, 'wave_equivalent_capacity': capacity,
            'register_limited_waves_per_simd': min(16, capacity//allocated),
            'upper_bound_only': True}


def operands(text: str) -> list[str]:
    """Split commas, respecting register ranges and modifier brackets."""
    out, start, depth = [], 0, 0
    for i, c in enumerate(text):
        if c in '[(': depth += 1
        elif c in '])': depth -= 1
        elif c == ',' and depth == 0:
            out.append(text[start:i].strip()); start = i+1
    if text[start:].strip(): out.append(text[start:].strip())
    return out


REG = re.compile(r'(?<![\w])([vs])(?:\[(\d+):(\d+)\]|(\d+))(?![\w])')


def regs(text: str) -> set[tuple[str, int]]:
    out = set()
    for kind, lo, hi, single in REG.findall(text):
        first, last = (int(single), int(single)) if single else (int(lo), int(hi))
        if first > last or last > 4096:
            raise FormatError('invalid register range in disassembly')
        out.update((kind, n) for n in range(first, last+1))
    for name, ids in [('vcc_lo', [106]), ('vcc_hi', [107]), ('vcc', [106,107])]:
        if re.search(r'(?<![\w])'+name+r'(?![\w])', text):
            out.update(('s', n) for n in ids)
    return out


def rw(ins: dict) -> tuple[set, set, bool]:
    """Reads/writes for a conservative local subset. Unknown != side-effect free."""
    op, args = ins['op'], ins['args']
    a = operands(args)
    if ins.get('vopd_independent_decode'):
        p = vopd.decode(bytes.fromhex(ins['raw']))
        r = {('v', x) for x in p.x.reads() | p.y.reads()}
        r |= {('s', x) for x in p.x.scalar_sources() | p.y.scalar_sources()
              if isinstance(x, int) and x < 108}
        return r, {('v', p.x.dst), ('v', p.y.dst)}, True
    allregs = regs(args)
    if op.startswith(('s_wait_', 's_delay_', 's_nop', 's_endpgm', 's_code_end', 'v_nop')):
        return set(), set(), True
    if op.startswith(('s_branch', 's_cbranch_', 's_call_', 's_setpc_', 's_swappc_',
                      's_sendmsg', 's_alloc_', 's_barrier', 's_get_barrier', 'exp',
                      'ds_direct_', 'ds_param_', 's_getreg', 's_setreg')):
        return allregs, set(), False
    if '_atomic_' in op or op.startswith(('v_movrel', 'v_swap', 's_movrel')):
        return allregs, set(), False
    if op.startswith(('global_store_', 'flat_store_', 'scratch_store_', 'buffer_store_', 'ds_store_')):
        return allregs, set(), True
    if op.startswith(('s_cmp', 's_bitcmp')):
        return allregs, set(), True
    if op.startswith('v_cmpx'):
        return allregs, set(), True
    if op.startswith('v_cmp'):
        # e32 comparisons may print their implicit VCC result explicitly.
        dest = regs(a[0]) if a and 'vcc' in a[0] else set()
        return regs(','.join(a[1:])), dest, False
    if not a or not op.startswith(('v_', 's_', 'global_load_', 'flat_load_',
                                  'scratch_load_', 'buffer_load_', 'ds_load_')):
        return allregs, set(), False
    dst = regs(a[0])
    reads = regs(','.join(a[1:]))
    # Partial d16 writes, carry/mask operands, and multi-destination operations
    # require semantics beyond this subset. Do not claim them as supported.
    unsupported = ('_co_', '_ci_', '_d16', '_b16', 'fmac', 'dot2acc', 'writelane',
                   'readlane', 'readfirstlane', 'cndmask', 'div_fmas', 'interp',
                   'saveexec', 'wrexec', 'cmov', 'load_block')
    known = not any(t in op for t in unsupported)
    return reads, dst, known


DIRECT_CONDITIONS = {f's_cbranch_{s}' for s in ('scc0','scc1','vccz','vccnz','execz','execnz')}
TERMINALS = {'s_endpgm', 's_endpgm_saved'}
CALLS = ('s_swappc', 's_call')
JUMPS = ('s_setpc', 's_rfe')          # write PC: a jump, return or tail call, never a fall-through
LLVM_TRAP = 2                         # AMDGPUUsage Trap Handler ABI: llvm.trap = s_trap 0x02
# Instructions LLVM may place inside its long-branch sequence: s_wait_alu (s_waitcnt_depctr) from
# insertIndirectBranch's ApplyHazardWorkarounds, s_delay_alu and s_nop from later passes.
_LONG_BRANCH_FILLER = {'s_wait_alu', 's_waitcnt_depctr', 's_delay_alu', 's_nop'}


def _imm(i: dict) -> int:
    return int.from_bytes(bytes.fromhex(i['raw'])[:2], 'little')


def long_branch_target(ins: list[dict], ix: int) -> int | None:
    """Target of the long branch that ends at ins[ix] (`s_setpc_b64 s[N:N+1]`), or None.

    LLVM BranchRelaxation expands an out-of-range branch with SIInstrInfo::insertIndirectBranch
    (llvm/lib/Target/AMDGPU/SIInstrInfo.cpp) into
        s_getpc_b64 s[N:N+1]                      ; .Lpost_getpc follows it
        s_add_u32  sN,   sN,   (dest-.Lpost_getpc)&0xffffffff    (gfx12 spelling s_add_co_u32)
        s_addc_u32 sN+1, sN+1, (dest-.Lpost_getpc)>>32           (gfx12 spelling s_add_co_ci_u32)
        s_setpc_b64 s[N:N+1]
    with s_wait_alu fillers on LLVM 20+. The offset is relative to the instruction after s_getpc_b64
    (the comment in SIInstrInfo::expandPostRAPseudo: s_getpc returns the address of the next
    instruction). Only exactly this shape is resolved; anything else stays an indirect branch.
    """
    target = ins[ix]['args'].strip()
    m = re.fullmatch(r's\[(\d+):(\d+)\]', target)
    if not m or int(m[2]) != int(m[1])+1: return None
    lo, hi = f's{m[1]}', f's{m[2]}'
    j, imms = ix-1, []
    for ops, reg in ((('s_add_co_ci_u32', 's_addc_u32'), hi), (('s_add_co_u32', 's_add_u32'), lo)):
        while j >= 0 and ins[j]['op'] in _LONG_BRANCH_FILLER: j -= 1
        if j < 0 or ins[j]['op'] not in ops: return None
        a = operands(ins[j]['args'])
        if len(a) != 3 or a[0] != reg or a[1] != reg: return None
        imm = isadb.immediate(a[2])                # `0x20`, or `lit(0x20)` from LLVM 24
        if imm is None: return None
        imms.append(imm & 0xffffffff)
        j -= 1
    while j >= 0 and ins[j]['op'] in _LONG_BRANCH_FILLER: j -= 1
    if j < 0 or ins[j]['op'] != 's_getpc_b64' or ins[j]['args'].strip() != target: return None
    offset = imms[1] | imms[0] << 32
    if offset >= 1 << 63: offset -= 1 << 64
    return ins[j]['pc'] + ins[j]['size'] + offset


def cfg(instructions: list[dict]) -> tuple[dict[int, list[int]], set[int], list[dict]]:
    """Direct-branch CFG of one function extent: (edges pc -> sorted successor pcs, reachable, issues).

    Successors (AMD RDNA4 ISA §5.5 program control; where the ISA leaves a choice, what LLVM emits):
    - s_endpgm, s_code_end: none. s_trap 0x02 (llvm.trap): none, the AMDGPUUsage Trap Handler ABI
      says the wave is halted and its queue put into the error state; any other s_trap falls through
      (0x03 llvm.debugtrap continues when no debugger is attached).
    - s_branch / s_cbranch_{scc,vcc,exec}*: the SIMM16 target; the conditional ones also fall through.
    - s_setpc_b64 / s_rfe_b64: never fall through. LLVM emits s_setpc_b64 for a return
      (s[30:31]), a tail call or a long branch, and always ends the block with it. A long branch in
      the exact BranchRelaxation shape (`long_branch_target`) gets its target as its only successor;
      otherwise there is no successor and CFG_INDIRECT ('incomplete') is reported.
    - s_swappc_b64 / s_call_b64: CFG_INDIRECT; the callee returns to the next instruction, which is
      the successor. If the call is the last instruction of the extent, or only S_CODE_END padding
      follows it, there is no successor and no CFG_FALLTHROUGH: LLVM ends the block of a call to a
      noreturn function with the call itself (e.g. `__builtin_trap()` in the callee), and a
      returning call is always followed by more code of the same function. The CFG_INDIRECT finding
      then carries `noreturn: True`: if that callee does return, execution runs past the end.
    - anything else: the next instruction; if there is none, CFG_FALLTHROUGH ('error' when
      reachable: that path never executes s_endpgm).
    """
    if not instructions: return {}, set(), []
    pcs = {i['pc'] for i in instructions}
    bypc = {i['pc']: i for i in instructions}
    edges, issues, off_end = {}, [], []
    for ix, i in enumerate(instructions):
        pc, op = i['pc'], i['op']
        nxt, succ = pc+i['size'], []
        if op in TERMINALS or op == 's_code_end' or (op == 's_trap' and _imm(i) == LLVM_TRAP):
            pass
        elif op.startswith(JUMPS):
            target = long_branch_target(instructions, ix) if op == 's_setpc_b64' else None
            if target is None:
                issues.append(finding('CFG_INDIRECT', 'incomplete',
                    'indirect branch or return: its target needs interprocedural/target evidence', pc))
            elif target in pcs: succ.append(target)
            else:
                issues.append(finding('CFG_TARGET', 'incomplete',
                    'long-branch target is outside the kernel or not a decoded instruction boundary', pc,
                    target=target))
        elif op.startswith(CALLS):
            if nxt in pcs and bypc[nxt]['op'] != 's_code_end':
                succ.append(nxt)
                issues.append(finding('CFG_INDIRECT', 'incomplete',
                    'call requires interprocedural/target evidence', pc))
            else:
                issues.append(finding('CFG_INDIRECT', 'incomplete',
                    'call at the end of the code is taken as a call to a noreturn function (LLVM ends '
                    'such a block with the call); if the callee returns, execution runs past the end', pc,
                    noreturn=True))
        elif op == 's_branch' or op in DIRECT_CONDITIONS:
            # SOPP immediate is signed DWORD displacement from next instruction.
            if i['size'] != 4:
                issues.append(finding('CFG_ENCODING', 'incomplete', 'unexpected branch encoding size', pc))
            else:
                imm = int.from_bytes(bytes.fromhex(i['raw'])[:2], 'little', signed=True)
                target = pc+4+4*imm
                if target not in pcs:
                    issues.append(finding('CFG_TARGET', 'incomplete',
                        'branch target is outside the kernel or not a decoded instruction boundary', pc,
                        target=target))
                else: succ.append(target)
            if op != 's_branch':
                if nxt in pcs: succ.append(nxt)
                else: off_end.append(pc)
        elif op.startswith('s_cbranch'):
            issues.append(finding('CFG_INDIRECT', 'incomplete',
                'unsupported conditional branch requires target evidence', pc))
            if nxt in pcs: succ.append(nxt)
        else:
            if nxt in pcs: succ.append(nxt)
            else: off_end.append(pc)
        edges[pc] = sorted(set(succ))
    reached, queue = set(), deque([instructions[0]['pc']])
    while queue:
        pc = queue.popleft()
        if pc in reached: continue
        reached.add(pc); queue.extend(edges.get(pc, []))
    for pc in off_end:
        # Past the last instruction lies the next function or S_CODE_END padding: a path
        # that gets there never executes S_ENDPGM (§5.5 program control).
        if pc in reached:
            issues.append(finding('CFG_FALLTHROUGH', 'error',
                'execution runs past the end of the kernel: this path has no s_endpgm', pc))
        else:
            issues.append(finding('CFG_FALLTHROUGH', 'incomplete',
                'function extent ends without a recognized terminator (not reached by direct branches)', pc))
    for i in instructions:
        if i['pc'] in reached and i['op'] == 's_code_end':
            issues.append(finding('REACHABLE_CODE_END', 'error', 'reachable illegal S_CODE_END padding', i['pc']))
    return edges, reached, issues


def is_dealloc(i: dict) -> bool:
    if i['op'] != 's_sendmsg': return False
    return 'dealloc_vgprs' in i['args'].lower() or int.from_bytes(bytes.fromhex(i['raw'])[:2], 'little') & 1023 == 3


def deallocation_checks(ins: list[dict], edges: dict, reachable: set) -> list[dict]:
    bypc = {i['pc']: i for i in ins}
    out = []
    dynamic = any(i['op'] == 's_alloc_vgpr' for i in ins)
    for i in ins:
        if i['pc'] not in reachable or not is_dealloc(i): continue
        if dynamic:
            out.append(finding('DEALLOC_DYNAMIC', 'review',
                'S_ALLOC_VGPR and full deallocation coexist; verify dynamic-mode/path exclusion', i['pc'], 'dealloc'))
        todo, seen = deque(edges[i['pc']]), set()
        while todo:
            pc = todo.popleft()
            if pc in seen: continue
            seen.add(pc); j = bypc[pc]
            if j['op'].startswith(('v_', 'ds_', 'global_', 'flat_', 'scratch_', 'buffer_', 'image_', 'exp')):
                out.append(finding('USE_AFTER_DEALLOC', 'error',
                    'reachable vector/memory operation after MSG_DEALLOC_VGPRS', pc, 'dealloc', producer=i['pc']))
            elif j['op'] not in TERMINALS and not j['op'].startswith(('s_wait_', 's_nop')):
                out.append(finding('DEALLOC_EPILOGUE', 'review',
                    'nontrivial epilogue after VGPR release requires manual verification', pc, 'dealloc'))
            todo.extend(edges.get(pc, []))
    return out


def local_checks(ins: list[dict], edges: dict, reachable: set, wave: int = 32, models: dict | None = None,
                 table: dict | None = None) -> tuple[list[dict], dict]:
    """Pattern-level checks within basic blocks. No inter-wave memory proof. `models`, `table`: passed
    on to hazards.analyze (radisa check builds the operand models once per kernel).

    The LLVM-fixed ALU hazards (WMMA_D_TO_AB, PERMLANE_AFTER_CMPX), DEALLOC_NO_NOP and the s_clause
    lint run over the whole direct CFG in `hazards.analyze`; their findings are appended here.
    """
    out, count = [], Counter()
    preds = Counter(x for xs in edges.values() for x in xs)
    branch_targets = {x for i in ins if i['op'].startswith(('s_branch','s_cbranch')) for x in edges.get(i['pc'], [])}
    pending: dict[str, list[tuple[set, int]]] = {'load': [], 'ds': [], 'km': [], 'exp': []}
    previous = None
    for i in ins:
        pc, op = i['pc'], i['op']
        if pc not in reachable: continue
        if pc in branch_targets or preds[pc] > 1 or (previous and edges.get(previous['pc']) != [pc]):
            pending = {k: [] for k in pending}
            count['basic_block_resets'] += 1
        reads, writes, known = rw(i)
        if not known: count['instructions_outside_rw_subset'] += 1
        else: count['instructions_in_rw_subset'] += 1
        # RAW/WAW candidates: narrow supported instruction subset only.
        if known:
            for domain, records in pending.items():
                for dest, producer in records:
                    if dest & (reads | writes):
                        out.append(finding('LOCAL_MEMORY_HAZARD', 'review',
                            f'local {domain.upper()} result consumed/overwritten without a proved wait',
                            pc, 'wait', producer=producer,
                            registers=sorted(f'{a}{b}' for a,b in dest & (reads|writes))))
        wait = re.fullmatch(r's_wait_(loadcnt|dscnt|kmcnt|expcnt|loadcnt_dscnt|storecnt_dscnt)', op)
        if wait:
            value = int.from_bytes(bytes.fromhex(i['raw'])[:2], 'little')
            domains = {'loadcnt': 'load', 'dscnt': 'ds', 'kmcnt': 'km', 'expcnt': 'exp'}
            if wait[1] in domains:
                key = domains[wait[1]]
                if value == 0: pending[key] = []
                else:
                    # Nonzero waits require a full counter history, including ops
                    # outside our read/write subset. Never infer completion here.
                    count['nonzero_waits_requiring_full_counter_model'] += 1
            else:
                if value & 255 == 0: pending['ds'] = []
                if op == 's_wait_loadcnt_dscnt' and value >> 8 == 0: pending['load'] = []
        if known and writes:
            if op.startswith(('global_load_', 'scratch_load_', 'buffer_load_')):
                pending['load'].append((writes, pc))
            elif op.startswith('flat_load_'):
                pending['load'].append((writes, pc)); pending['ds'].append((writes, pc))
            elif op.startswith('ds_load_'): pending['ds'].append((writes, pc))
            elif op.startswith(('s_load_', 's_buffer_load_')): pending['km'].append((writes, pc))
        previous = i
    # Exact three-/four-instruction instance of the documented SGPR-pair hazard.
    # Broader ALU scoreboard modeling remains explicitly outside coverage.
    for ix in range(len(ins)-2):
        a,b,c = ins[ix:ix+3]
        if not (a['op']=='v_mov_b32' and b['op']=='s_mov_b32'): continue
        aa,bb = operands(a['args']),operands(b['args'])
        if len(aa)<2 or len(bb)<2: continue
        ar, bw = regs(aa[1]),regs(bb[0])
        if not (len(ar)==len(bw)==1): continue
        ra, rb = next(iter(ar)),next(iter(bw))
        if ra[0]!='s' or rb[0]!='s' or ra[1]//2!=rb[1]//2: continue
        cr,cw,ck=rw(c)
        if ck and rb in cr and c['op'].startswith(('s_','v_')) and c['pc'] in reachable:
            if edges.get(a['pc'])==[b['pc']] and edges.get(b['pc'])==[c['pc']]:
                out.append(finding('SGPR_PAIR_GUARD', 'review',
                    'documented VALU-read/SALU-write/read SGPR-pair pattern lacks S_WAIT_ALU guard',
                    c['pc'], 'wait', producer=a['pc']))
    hz, hstats = hazards.analyze(ins, edges, reachable, models, table, wave=wave)
    out += hz; count.update(hstats)
    return out, dict(count)


def waterfall_checks(ins: list[dict], edges: dict) -> list[dict]:
    bypc = {i['pc']:i for i in ins}
    out = []
    for i in ins:
        if not i['op'].startswith('s_cbranch_exec'): continue
        for target in edges.get(i['pc'], []):
            if target >= i['pc']: continue
            body = [j for j in ins if target <= j['pc'] <= i['pc']]
            relative = any(j['op'].startswith(('v_movrel','v_swaprel')) for j in body)
            firstlane = any(j['op']=='v_readfirstlane_b32' for j in body)
            execmask = any('exec' in j['args'] or 'saveexec' in j['op'] for j in body[:-1])
            if relative and firstlane and execmask:
                out.append(finding('WATERFALL_CANDIDATE', 'info',
                    'EXEC backedge encloses lane selection and relative VGPR access; measure cost',
                    i['pc'], 'index', loop_start=target))
    return out


def audit_kernel(kernel: dict, decoded: dict, models: dict | None = None, table: dict | None = None) -> dict:
    """The per-kernel audit (`radisa audit`, and the base of `radisa check`). `models` (as from
    waits.operand_models for the kernel's wave size) and `table`: see local_checks."""
    ins = decoded['instructions']
    issues = [finding('KERNEL_EVIDENCE', 'incomplete', m) for m in kernel['issues']]
    issues += [finding('DECODE', 'incomplete', e['message'], e.get('pc')) for e in decoded['issues']]
    if decoded['decoded_bytes'] != kernel.get('size', 0) or not ins:
        issues.append(finding('BYTE_COVERAGE', 'incomplete', 'not every kernel byte was decoded'))
    md, descriptor = kernel['metadata'], kernel.get('descriptor', {})
    resource = None
    if descriptor:
        resource = occupancy(md['.vgpr_count'], md['.wavefront_size'], descriptor['encoded_vgprs'])
    for i in ins:
        if i.get('vopd_independent_decode'):
            for rule, text in vopd.validate(vopd.decode(bytes.fromhex(i['raw'])), md['.wavefront_size'],
                                             descriptor.get('encoded_vgprs')):
                # ANTIDEP is a note: LLVM emits it on purpose (isVOPDAntidependencyAllowed); 3065 hits, all
                # compiler output, in the 114 libr4d units measured with AMD clang 24.
                severity = 'info' if rule == 'ANTIDEP' else 'review' if rule in ('REGISTER_RANGE', 'SOURCE_HI') else 'error'
                issues.append(finding('VOPD_'+rule, severity,
                                      text, i['pc'], 'resources' if rule == 'REGISTER_RANGE' else 'vopd'))
        elif 'v_dual_' in i['text']:
            issues.append(finding('VOPD_IDENTITY', 'incomplete', 'LLVM VOPD does not match raw format detector',i['pc'],'vopd'))
        if i['op']=='s_alloc_vgpr':
            if md['.wavefront_size']==64:
                issues.append(finding('DYNAMIC_WAVE64','error','S_ALLOC_VGPR is illegal in Wave64',i['pc'],'dealloc'))
            issues.append(finding('DYNAMIC_VGPR_MODE','incomplete',
                'dynamic VGPR launch mode and forward progress not modeled',i['pc'],'dealloc'))
        if re.search(r'\bth:(?:TH_)?LU\b',i['args'],re.I):
            issues.append(finding('LAST_USE_LIFETIME','review',
                'last-use dirty discard requires whole-buffer lifetime proof',i['pc'],'cache'))
    edges, reachable, cissues = cfg(ins); issues += cissues
    issues += deallocation_checks(ins, edges, reachable)
    local, stats = local_checks(ins, edges, reachable, md.get('.wavefront_size', 32), models, table); issues += local
    issues += waterfall_checks(ins, edges)
    counts = Counter(i['op'] for i in ins)
    return {'name': kernel['name'], 'symbol': kernel['symbol'],
            'code_sha256': kernel.get('code_sha256'), 'address': kernel.get('address'),
            'expected_bytes':kernel.get('size',0), 'decoded_bytes':decoded['decoded_bytes'],
            'instruction_count':len(ins), 'reachable_instruction_count':len(reachable),
            'opcode_counts':dict(sorted(counts.items())), 'resources':resource,
            'scratch_bytes_per_workitem':md.get('.private_segment_fixed_size'),
            'findings':issues, 'local_scanner_stats':stats,
            'control_flow':{hex(k):[hex(x) for x in v] for k,v in edges.items()},
            'checks':{
                'vopd':'raw-binary legality subset', 'direct_cfg':'intra-kernel direct edges',
                'deallocation':'reachable uses, explicit CFG only',
                'wmma':'whole direct-CFG D-to-A/B/index hazards (LLVM fixWMMAHazards)',
                'alu_hazards':'whole direct-CFG CMPX->PERMLANE (LLVM fixVcmpxPermlaneHazards)',
                'clauses':'s_clause length/type/branch lint (LLVM SIInsertHardClauses)',
                'memory_waits':'local supported-op diagnostic; not proof',
                'sgpr_pair':'documented narrow pattern diagnostic; not full scoreboard',
                'waterfall':'measured-cost candidate, not correctness failure',
                'global_memory_safety':'not_proved', 'p2p_visibility':'not_proved',
                'numerical_correctness':'not_proved', 'gpu_wedge_freedom':'not_proved'}}
