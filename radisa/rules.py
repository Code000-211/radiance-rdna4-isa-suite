"""Bounded static checks, not a general proof of AMD ISA conformance.

Each finding cites the supplied 2025-04-07 RDNA4 guide. Exact encoding checks
are separated from local pattern diagnostics and performance observations.
"""
from __future__ import annotations
from collections import Counter, deque
import re
from .binary import FormatError
from . import vopd

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


def cfg(instructions: list[dict]) -> tuple[dict[int, list[int]], set[int], list[dict]]:
    if not instructions: return {}, set(), []
    pcs = {i['pc'] for i in instructions}
    edges, issues, off_end = {}, [], []
    for i in instructions:
        pc, op = i['pc'], i['op']
        nxt, succ = pc+i['size'], []
        if op in TERMINALS or op == 's_code_end':
            pass
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
        elif op.startswith(('s_setpc', 's_swappc', 's_call', 's_rfe', 's_cbranch')):
            issues.append(finding('CFG_INDIRECT', 'incomplete',
                'call/indirect or unsupported branch requires interprocedural/target evidence', pc))
            if nxt in pcs: succ.append(nxt)
            elif op.startswith(('s_swappc', 's_call')): off_end.append(pc)   # the callee returns here
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


def local_checks(ins: list[dict], edges: dict, reachable: set) -> tuple[list[dict], dict]:
    """Pattern-level checks within basic blocks. No inter-wave memory proof."""
    out, count = [], Counter()
    preds = Counter(x for xs in edges.values() for x in xs)
    branch_targets = {x for i in ins if i['op'].startswith(('s_branch','s_cbranch')) for x in edges.get(i['pc'], [])}
    pending: dict[str, list[tuple[set, int]]] = {'load': [], 'ds': [], 'km': [], 'exp': []}
    last_wmma = None
    previous = None
    for i in ins:
        pc, op = i['pc'], i['op']
        if pc not in reachable: continue
        if pc in branch_targets or preds[pc] > 1 or (previous and edges.get(previous['pc']) != [pc]):
            pending = {k: [] for k in pending}; last_wmma = None
            count['basic_block_resets'] += 1
        reads, writes, known = rw(i)
        if not known: count['instructions_outside_rw_subset'] += 1
        else: count['instructions_in_rw_subset'] += 1
        # Independent of rw modeling: immediate CMPX -> PERMLANE restriction.
        if previous and op.startswith('v_permlane') and previous['op'].startswith('v_cmpx'):
            out.append(finding('PERMLANE_AFTER_CMPX', 'error',
                'PERMLANE immediately follows V_CMPX without intervening VALU', pc, 'permlane'))
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
        if op.startswith(('v_wmma_', 'v_swmmac_')):
            a = operands(i['args'])
            if len(a) < 4:
                out.append(finding('WMMA_OPERANDS', 'incomplete', 'WMMA operands not understood', pc, 'wmma'))
            else:
                dst = regs(a[0])
                abindex = regs(','.join(a[1:3]+a[4:]))
                if last_wmma and last_wmma[1] & abindex:
                    out.append(finding('WMMA_D_TO_AB', 'error',
                        'WMMA D feeds next WMMA A/B/index without V_NOP or proved independent VALU',
                        pc, 'wmma', producer=last_wmma[0]))
                last_wmma = (pc, dst)
        elif op == 'v_nop': last_wmma = None
        elif op.startswith('v_') and last_wmma:
            if known and not (last_wmma[1] & (reads|writes)):
                last_wmma = None
            else:
                # A dependent/unknown VALU might add the required scheduling
                # separation but we cannot certify independence. Do not turn a
                # later overlap into a definite violation.
                last_wmma = None
                count['wmma_separation_unknown'] += 1
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


def audit_kernel(kernel: dict, decoded: dict) -> dict:
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
                issues.append(finding('VOPD_'+rule, 'review' if rule in ('REGISTER_RANGE', 'ANTIDEP', 'SOURCE_HI') else 'error',
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
    local, stats = local_checks(ins, edges, reachable); issues += local
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
                'wmma':'basic-block explicit D-to-A/B/index hazards',
                'memory_waits':'local supported-op diagnostic; not proof',
                'sgpr_pair':'documented narrow pattern diagnostic; not full scoreboard',
                'waterfall':'measured-cost candidate, not correctness failure',
                'global_memory_safety':'not_proved', 'p2p_visibility':'not_proved',
                'numerical_correctness':'not_proved', 'gpu_wedge_freedom':'not_proved'}}
