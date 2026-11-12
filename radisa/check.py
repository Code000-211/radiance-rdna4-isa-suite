"""`radisa check`: lint one kernel source or one code object against RDNA4 rules.

Inputs
  *.hip / *.cpp / *.cu   compiled for gfx1201 (device only, -g), then checked as machine
                         code; findings point at source lines
  *.s                    assembled with llvm-mc; text without .amdhsa_kernel is checked
                         as a single wave32 instruction sequence ("snippet")
  anything else          ELF code object, offload bundle, archive, fat binary or HIP
                         executable/library; every gfx1201 kernel inside is checked

Exit status: 0 clean, 1 findings at the failing severity, 3 analysis incomplete (including
kernel bytes that could not be decoded: the rest of such a kernel was never analysed).
"""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .binary import ELF, FormatError, extract_code_objects, kernel_inventory
from .decode import decode_object, select_backend
from . import asmclobber, formats, functions, hipsrc, isadb, liveness, memorder, rules, sgpr, suppress, waits
from .incremental import check_hip_incremental

SEVERITY_LABEL = {'error': 'error', 'review': 'warning', 'incomplete': 'note', 'info': 'note'}
RULE_SECTION = {'VOPD_': '7.8', 'WMMA_': '7.12.1', 'PERMLANE': '7.2.8', 'DEALLOC': '3.3.3',
                'USE_AFTER_DEALLOC': '3.3.3', 'REACHABLE_CODE_END': '2.4', 'SGPR_PAIR': '5.8',
                'WATERFALL': '7.11', 'LAST_USE': '4.1.1', 'DYNAMIC_': '3.3.3', 'VGPR_OUT': '3.3.2',
                'CLAUSE_': '5.3'}
DROP_KERNEL_ISSUES = ('relocatable ELF',)


def _section(rule: str) -> str | None:
    for prefix, sec in RULE_SECTION.items():
        if rule.startswith(prefix):
            return sec
    return None


def _vgpr_range(kernel: dict, ins: list[dict], findings: list[dict]) -> list[dict]:
    """Instructions that name a VGPR the kernel descriptor does not allocate.

    The ISA gives out-of-range register references a fallback instead of a trap (§3.3.2), so the
    code runs on with values the program did not compute. Compiler output never does it;
    hand-written assembly and inline asm can. Not checked with dynamic VGPRs (s_alloc_vgpr), where the allocation changes at run time.
    """
    allocated = (kernel.get('descriptor') or {}).get('encoded_vgprs')
    if not allocated or any(i['op'] == 's_alloc_vgpr' for i in ins):
        return []
    out = []
    for i in ins:
        try:
            regs = isadb.registers(i['args'])
        except FormatError:
            continue
        high = max((n for kind, n in regs if kind == 'v'), default=-1)
        if high >= allocated:
            out.append({'rule': 'VGPR_OUT_OF_RANGE', 'severity': 'error', 'pc': i['pc'],
                        'message': f"{i['op']} uses v{high}, but the kernel descriptor allocates only "
                                   f'{allocated} VGPRs (v0-v{allocated - 1}); the hardware does not trap on it, '
                                   'the instruction works with a register the program does not own',
                        'registers': [f'v{high}'], 'allocated_vgprs': allocated})
    if out:
        # The VOPD check says the same for its pairs, as a warning.
        findings[:] = [f for f in findings if f['rule'] != 'VOPD_REGISTER_RANGE']
    return out


# Extension points. Each entry of KERNEL_PASSES is called as fn(ctx) -> list[finding] once per
# kernel, after the built-in rules; ctx keys: kernel, instructions, edges, reachable, table, wave,
# origin, models, options (see check_kernel). A pass cites its normative basis in 'source'.
KERNEL_PASSES: list = []   # registered at the end of this module
# Each entry of REPORT_FILTERS is called as fn(report, args) -> None once all findings exist and
# before finalize() computes counts and status; it may drop or re-grade findings in place.
REPORT_FILTERS: list = []   # registered at the end of this module
INCOMPLETE_RULES = ('CHECK_FAILED', 'ASSEMBLER', 'SOURCE_MAP',
                    # Code that exists but was not analysed: a function without a symbol size, a
                    # descriptor without metadata (OBJECT_EVIDENCE); paths behind a branch whose
                    # target is not an instruction of the kernel (CFG_TARGET, CFG_ENCODING); a VOPD
                    # pair or WMMA whose own rules could not run (VOPD_IDENTITY, WMMA_OPERANDS).
                    # Not here, on purpose: KERNEL_EVIDENCE (a kernel without readable bytes already
                    # has no instructions; the remaining message is about prefetch padding, not about
                    # code that went unchecked), CFG_FALLTHROUGH 'incomplete' (unreached code at the
                    # end, analysed), DYNAMIC_VGPR_MODE (every instruction is analysed; only the VGPR
                    # range check is off), CFG_INDIRECT (callees: the CALLS note, NOT_CHECKED).
                    'OBJECT_EVIDENCE', 'CFG_TARGET', 'CFG_ENCODING', 'VOPD_IDENTITY', 'WMMA_OPERANDS')
# The analysis did not see the whole kernel: the rest of its bytes were never checked.
COVERAGE_RULES = ('DECODE', 'BYTE_COVERAGE')


def _unknown_operand_models(ins: list[dict], models: dict, stats: dict, wave: int,
                            edges: dict, reachable: set, findings: list[dict],
                            callee: bool = False) -> list[dict]:
    """OPERAND_MODEL_UNKNOWN, and SGPR findings that rest only on the guess for such instructions.

    An instruction the AMD XML table cannot model is taken to read and write every register its
    text names (isadb.operand_model). Where a MISSING_ALU_WAIT exists only because of an assumed
    operand (the guessed write as producer, or the guessed read as consumer or as the earlier VALU
    read the hazard needs), the hardware facts behind it are unknown: it is re-graded to 'review'.
    LLVM computes these waits from the uses of an instruction only (AMDGPUWaitSGPRHazards.cpp,
    processOperand on MI->all_uses()), so a guessed read is as uncertain as a guessed write.
    """
    unknown = stats.get('unknown_mnemonics') or {}
    if not unknown:
        return []
    pcs = [i['pc'] for i in ins if not models[i['pc']]['known']]
    guessed = {pc: {**m, 'reads': set(), 'writes': set()} if not m['known'] else m for pc, m in models.items()}
    without, _ = sgpr.analyze(ins, edges, reachable, guessed, wave, callee)
    certain = {f['pc'] for f in without if f['rule'] == 'MISSING_ALU_WAIT'}
    for f in findings:
        if f['rule'] == 'MISSING_ALU_WAIT' and f['pc'] not in certain:
            f.update(severity='review', operand_model='assumed',
                     message=f['message'] + '. This rests on the assumption that an instruction outside '
                             'the AMD operand table reads and writes the registers it names; its real '
                             'operands are not documented in the table')
    listed = ', '.join(f'{op} ({n}x)' for op, n in sorted(unknown.items()))
    return [{'rule': 'OPERAND_MODEL_UNKNOWN', 'severity': 'review', 'pc': pcs[0] if pcs else None,
             'message': f'{len(pcs)} instruction(s) outside the AMD RDNA4 operand table: {listed}. They are '
                        'treated as reading and writing every register they name; counters they '
                        'increment and implicit operands they use are not modelled',
             'mnemonics': dict(sorted(unknown.items())), 'pcs': pcs,
             'source': 'AMD RDNA4 ISA XML (radisa/data); radisa/isadb.py operand_model'}]


def check_kernel(kernel: dict, decoded: dict, table: dict, origin: dict | None = None,
                 options: dict | None = None) -> dict:
    """Check one kernel, or one non-kernel function (`kernel['callee']`, see functions.py).

    `options` reaches every KERNEL_PASSES entry as ctx['options'], without the private keys
    check_object_bytes adds for the CALLS note ('_callees': functions of the object checked in this
    run, '_functions': all of them, '_unchecked': functions found without a symbol size).
    """
    callee = bool(kernel.get('callee'))
    kernel = {**kernel, 'issues': [m for m in kernel.get('issues', [])
                                   if not m.startswith(DROP_KERNEL_ISSUES)]}
    base = rules.audit_kernel(kernel, decoded)
    findings = [f for f in base['findings'] if f['rule'] not in ('LOCAL_MEMORY_HAZARD', 'CFG_INDIRECT')]
    returns = {i['pc'] for i in decoded['instructions'] if callee and i['op'] == 's_setpc_b64'
               and i['args'].strip() == waits.RETURN_ADDRESS}
    calls = [f for f in base['findings'] if f['rule'] == 'CFG_INDIRECT' and f['pc'] not in returns]
    if calls:
        checked = sorted((options or {}).get('_callees') or ())
        present = (options or {}).get('_functions') or ()
        unchecked = sorted((options or {}).get('_unchecked') or ())
        if checked:
            where = f"the functions of this object ({', '.join(checked)}) are checked on their own; "
        elif present:
            where = 'the functions of this object are not checked in this run (--kernel); '
        elif not unchecked:
            where = 'the object holds no function body to check (callees elsewhere are not analysed); '
        else:
            where = ''
        if unchecked:
            where += (f"function(s) {', '.join(unchecked)} of this object are NOT checked (no symbol size, "
                      'see OBJECT_EVIDENCE); ')
        findings.append({'rule': 'CALLS', 'severity': 'info', 'pc': calls[0]['pc'],
                         'message': f'{len(calls)} call(s)/indirect branch(es); {where}every counter but '
                                    'STOREcnt is assumed drained on return (LLVM SIInsertWaitcnts)',
                         'call_pcs': [f['pc'] for f in calls], 'callees_checked': checked})
    if callee:
        for f in findings:
            if f['rule'] == 'CFG_FALLTHROUGH':
                f.update(severity='review', message='a path runs past the end of the function without a '
                                                    'return (s_setpc_b64 s[30:31]) or s_endpgm')
    elif not kernel.get('descriptor'):
        # A snippet is a fragment: where it ends says nothing about a kernel's end.
        for f in findings:
            if f['rule'] == 'CFG_FALLTHROUGH':
                f.update(severity='info', message='the snippet ends without s_endpgm (not an error for a fragment)')
    findings += _vgpr_range(kernel, decoded['instructions'], findings)
    for f in findings:
        sec = _section(f['rule'])
        if sec and 'llvm' not in f:
            f['source'] = isadb.cite(table, sec)
    ins = decoded['instructions']
    edges, reachable, _ = rules.cfg(ins)
    md = kernel['metadata']
    wave = md.get('.wavefront_size', 32)
    models = waits.operand_models(ins, table, wave)
    wf, stats = waits.analyze(ins, edges, reachable, table, wave, origin, models, callee=callee)
    findings += wf
    findings += _unknown_operand_models(ins, models, stats, wave, edges, reachable, wf, callee)
    spills = (md.get('.vgpr_spill_count') or 0) + (md.get('.sgpr_spill_count') or 0)
    scratch = md.get('.private_segment_fixed_size') or 0
    if spills:
        findings.append({'rule': 'REGISTER_SPILL', 'severity': 'review', 'pc': None,
                         'message': f"{md.get('.vgpr_spill_count', 0)} VGPR / {md.get('.sgpr_spill_count', 0)} "
                                    f'SGPR spills to scratch ({scratch} bytes per lane)',
                         'source': 'AMDGPU code object metadata'})
    elif scratch and not callee and any(i['op'].startswith(('s_swappc', 's_call')) for i in ins):
        # The kernel's private segment size includes the stack of its callees (LLVM
        # AMDGPUMCResourceInfo.cpp gatherResourceInfo: own stack + max over callees, or
        # AssumedStackSizeForExternalCall for indirect/recursive calls), so scratch is expected.
        findings.append({'rule': 'SCRATCH_USE', 'severity': 'info', 'pc': None,
                         'message': f'{scratch} bytes of scratch per lane without spills; the kernel makes '
                                    'calls, and the call stack of its callees is part of this size (a '
                                    'private array may be too; the metadata does not split the two)',
                         'source': 'AMDGPU code object metadata; LLVM AMDGPUMCResourceInfo.cpp '
                                   'gatherResourceInfo, AMDGPUResourceUsageAnalysis.cpp'})
    elif scratch:
        findings.append({'rule': 'SCRATCH_USE', 'severity': 'review', 'pc': None,
                         'message': f'{scratch} bytes of scratch per lane without spills: a private '
                                    'array the compiler could not keep in registers (dynamic index?)',
                         'source': 'AMDGPU code object metadata'})
    if KERNEL_PASSES:
        ctx = {'kernel': kernel, 'instructions': ins, 'edges': edges, 'reachable': reachable,
               'table': table, 'wave': wave, 'origin': origin, 'models': models,
               'options': {k: v for k, v in (options or {}).items() if not k.startswith('_')}}
        for fn in KERNEL_PASSES:
            findings += fn(ctx) or []
    if origin:
        for f in findings:
            if f.get('pc') is not None and f['pc'] in origin and 'loc' not in f:
                o = origin[f['pc']]
                if o.get('line'):
                    f['loc'] = {k: o[k] for k in ('file', 'line', 'column')}
    by_pc = {i['pc']: i for i in ins}
    for f in findings:
        if f.get('pc') in by_pc:
            f['instruction'] = by_pc[f['pc']]['text']
        if isinstance(f.get('producer'), int) and f['producer'] in by_pc:
            f['producer_instruction'] = by_pc[f['producer']]['text']
    return {'name': kernel['name'], 'symbol': kernel.get('symbol'), 'address': kernel.get('address', 0),
            'instructions': len(ins), 'resources': base['resources'],
            'metadata': {k: md.get(k) for k in ('.wavefront_size', '.vgpr_count', '.sgpr_count',
                                                '.group_segment_fixed_size', '.private_segment_fixed_size',
                                                '.vgpr_spill_count', '.sgpr_spill_count',
                                                '.max_flat_workgroup_size')},
            'operand_model': stats, 'findings': findings, **({'callee': True} if callee else {})}


def _objects(path: Path) -> list[bytes]:
    objs, events = extract_code_objects(path.read_bytes(), str(path))
    out = [o['data'] for o in objs if o['target'] == 'gfx1201']
    if not out:
        targets = sorted({o['target'] for o in objs})
        raise FormatError(f'no gfx1201 code object in {path} (found: {targets or "none"})')
    return out


def check_object_bytes(data: bytes, label: str, table: dict, backend: dict, kernel_filter: str | None,
                       work: Path, asm_functions: dict | None = None,
                       timeout: float = 600, options: dict | None = None,
                       map_severity: str = 'incomplete') -> tuple[list[dict], list[dict]]:
    """Check every gfx1201 kernel of one code object, and every non-kernel function in it
    (functions.function_inventory, checked with callee semantics). `map_severity` grades the
    SOURCE_MAP notes for `asm_functions` that do not line up with the object."""
    elf = ELF(data)
    ks, issues = kernel_inventory(elf)
    fs = functions.function_inventory(elf, ks, issues)
    notes = [{'rule': 'OBJECT_EVIDENCE', 'severity': 'incomplete', 'message': m, 'pc': None}
             for m in issues]
    # Functions found but not checked (functions.function_inventory reports them in `issues`).
    unchecked = tuple(m[1] for m in (re.match(r'function (\S+) has no symbol size', i) for i in issues) if m)
    present = tuple(f['name'] for f in fs)
    if kernel_filter:
        ks = [k for k in ks if kernel_filter in k['name'] or kernel_filter in k['symbol']]
        fs = [f for f in fs if kernel_filter in f['name']]
    options = {**(options or {}), '_callees': tuple(f['name'] for f in fs), '_functions': present,
               '_unchecked': unchecked}
    ks = ks + fs
    usable = [k for k in ks if 'code_hex' in k]
    tmp = work / 'object.co'
    tmp.write_bytes(data)
    decoded = decode_object(tmp, usable, backend, timeout) if usable else {}
    results = []
    for k in ks:
        d = decoded.get(k['symbol'], {'instructions': [], 'issues': [], 'decoded_bytes': 0})
        origin = None
        if asm_functions is not None:
            fn = asm_functions.get(k['name'])
            if fn is None:
                notes.append({'rule': 'SOURCE_MAP', 'severity': map_severity, 'pc': None,
                              'message': f"{k['name']}: no matching function in the assembly"})
            else:
                origin, why = hipsrc.source_map(d['instructions'], fn)
                if why:
                    notes.append({'rule': 'SOURCE_MAP', 'severity': map_severity, 'pc': None,
                                  'message': f"{k['name']}: {why}; findings carry pcs only"})
                    origin = None
        r = check_kernel(k, d, table, origin, options)
        r['object'] = label
        results.append(r)
    return results, notes


def snippet_kernel(obj: Path, wave: int) -> tuple[dict, bytes]:
    elf = ELF(obj.read_bytes())
    text = [s for s in elf.sections if s.name == '.text']
    if not text:
        raise FormatError('assembled snippet has no .text section')
    code = elf.data[text[0].offset:text[0].offset + text[0].size]
    md = {'.wavefront_size': wave, '.vgpr_count': 0, '.sgpr_count': 0,
          '.group_segment_fixed_size': 0, '.private_segment_fixed_size': 0}
    return {'name': '<snippet>', 'symbol': '<snippet>', 'metadata': md, 'issues': [], 'address': 0,
            'size': len(code), 'code_hex': code.hex(), 'descriptor': {}}, code


def _code_lines(text: str) -> list[str]:
    """The lines of an assembly file with every comment blanked (`;` and `//` to the end of the
    line, `/* ... */` across lines) and string literals kept, so columns and line numbers stay."""
    out, cur, i, n, block = [], [], 0, len(text), False
    while i < n:
        c = text[i]
        if c == '\n':
            out.append(''.join(cur))
            cur = []
        elif block:
            if text.startswith('*/', i):
                block = False
                cur.append('  ')
                i += 2
                continue
            cur.append(' ')
        elif text.startswith('/*', i):
            block = True
            cur.append('  ')
            i += 2
            continue
        elif c == ';' or text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        elif c == '"':
            j = i + 1
            while j < n and text[j] not in '"\n':
                j += 2 if text[j] == '\\' else 1
            cur.append(text[i:j + 1] if j < n and text[j] == '"' else text[i:j])
            i = j + 1 if j < n and text[j] == '"' else j
            continue
        else:
            cur.append(c)
        i += 1
    out.append(''.join(cur))
    return out


def asm_line_map(text: str, path: str, snippet: bool) -> dict:
    """Instructions of a hand-written .s file per function, in the shape of hipsrc.parse_asm, with
    `loc` = the line of the instruction in the .s file itself (column 1, as for every .s line).

    Labels are skipped wherever they stand (also before an instruction on the same line). A
    function starts at the label of a symbol declared `.type name,@function` or `.amdhsa_kernel
    name`; other labels (`loop:`, `.L...`) stay inside it. In a snippet every instruction of the
    text sections belongs to the one function '<snippet>'. Instructions that llvm-mc expands
    (macros, .rept) do not line up with the object: source_map then reports the divergence.
    """
    code = _code_lines(text)
    names = set() if snippet else (
        set(re.findall(r'^\s*\.type\s+([^\s,]+)\s*,\s*[@%]function\b', '\n'.join(code), re.M))
        | set(re.findall(r'^\s*\.amdhsa_kernel\s+(\S+)', '\n'.join(code), re.M)))
    out: dict[str, list[dict]] = {'<snippet>': []} if snippet else {}
    current = '<snippet>' if snippet else None
    in_text, in_metadata, in_macro = True, False, False
    for no, raw in enumerate(code, 1):
        line = raw.strip()
        if in_macro or re.match(r'\.macro\b', line):   # a body is code where it is invoked
            in_macro = not re.match(r'\.endm\b', line)
            continue
        if in_metadata:
            in_metadata = not line.startswith('.end_amdgpu_metadata')
            continue
        if line.startswith('.amdgpu_metadata'):
            in_metadata = True
            continue
        if re.match(r'\.(text\b|section\s+\.text)', line):
            in_text = True
            continue
        if re.match(r'\.(section|data|bss|rodata)\b', line):
            in_text = False
            continue
        while True:
            m = hipsrc.LABEL.match(line)
            if not m:
                break
            if m[1] in names and in_text:
                current = m[1]
                out.setdefault(current, [])
            line = line[m.end():].lstrip()
        if not line or line.startswith('.') or not in_text or current is None:
            continue
        out[current].append({'mnemonic': line.split()[0], 'text': line, 'inline_asm': False,
                              'loc': {'file': path, 'line': no, 'column': 1}})
    return out


def check_assembly(path: Path, table: dict, backend: dict, args, work: Path):
    """A .s file. Compiler output with `.loc` directives maps findings to the HIP lines named
    there (hipsrc.parse_asm); hand-written assembly maps them to its own lines (asm_line_map), so
    `; radisa: ignore[...]` comments in the .s file apply. Where that mapping fails, a SOURCE_MAP
    note says so ('info': the analysis itself is complete; findings carry pcs only)."""
    text = path.read_text(encoding='utf-8', errors='replace')
    mc = hipsrc.find_tool('llvm-mc', args.llvm_mc)
    obj = work / 'asm.o'
    p = subprocess.run([mc, '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-filetype=obj',
                        *(['-mattr=+wavefrontsize64'] if args.wave == 64 else []),
                        str(path), '-o', str(obj)], capture_output=True, text=True, check=False)
    if p.returncode:
        return [], [{'rule': 'ASSEMBLER', 'severity': 'error', 'pc': None,
                     'message': p.stderr.strip() or 'llvm-mc failed'}]
    if '.amdhsa_kernel' in text:
        if re.search(r'^\s*\.loc\s', text, re.M):
            return check_object_bytes(obj.read_bytes(), str(path), table, backend, args.kernel, work,
                                      hipsrc.parse_asm(text), args.timeout, pass_options(args))
        return check_object_bytes(obj.read_bytes(), str(path), table, backend, args.kernel, work,
                                  asm_line_map(text, str(path), False), args.timeout, pass_options(args),
                                  map_severity='info')
    k, _ = snippet_kernel(obj, args.wave)
    decoded = decode_object(obj, [k], backend)
    d = decoded[k['symbol']]
    origin, why = hipsrc.source_map(d['instructions'], asm_line_map(text, str(path), True)['<snippet>'])
    notes = []
    if why:
        notes.append({'rule': 'SOURCE_MAP', 'severity': 'info', 'pc': None,
                      'message': f'{why}; findings carry pcs only, so line-scoped `radisa: ignore[...]` '
                                 'comments of this file cannot match them'})
    r = check_kernel(k, d, table, origin or None, pass_options(args))
    r['object'] = str(path)
    return [r], notes


def check_hip(path: Path, table: dict, backend: dict, args, work: Path):
    notes = []
    src_findings = hipsrc.source_checks(path, lambda s: isadb.cite(table, s))
    flags = []
    for f in args.flag or []:
        flags += f.split()
    c = hipsrc.compile_hip(path, work, args.cxx, flags,
                           Path(args.compile_commands) if args.compile_commands else None)
    if c['returncode'] or not Path(c['asm']).exists():
        return [], src_findings + [{'rule': 'COMPILE', 'severity': 'error', 'pc': None,
                                    'message': 'gfx1201 device compile failed:\n' + c['stderr'].strip(),
                                    'command': ' '.join(c['command'])}]
    if c.get('assembler_returncode'):
        return [], src_findings + [{'rule': 'ASSEMBLER', 'severity': 'incomplete', 'pc': None,
                                    'message': 'llvm-mc could not assemble the compiler output: '
                                               + c['assembler_stderr'].strip()}]
    if c['stderr'].strip():
        notes.append({'rule': 'COMPILER_DIAGNOSTICS', 'severity': 'info', 'pc': None,
                      'message': c['stderr'].strip()})
    functions = hipsrc.parse_asm(Path(c['asm']).read_text(encoding='utf-8', errors='replace'), c['cwd'])
    results, n2 = check_object_bytes(Path(c['object']).read_bytes(), str(path), table, backend,
                                     args.kernel, work, functions, args.timeout, pass_options(args))
    # ASM_UNDECLARED_CLOBBER works on the optimised IR (a second clang run with the same flags),
    # as the incremental path does with its module.
    ir = hipsrc.compile_ir(path, work, args.cxx, flags,
                           Path(args.compile_commands) if args.compile_commands else None, timeout=args.timeout)
    if ir['ir'] is not None:
        names = {r['name'] for r in results}
        n2 += asmclobber.attach(results, [f for f in asmclobber.check_ir_text(ir['ir'], table, str(path))
                                          if not f['kernel'] or f['function'] in names])
    else:
        n2.append({'rule': 'ASM_CLOBBER_SKIPPED', 'severity': 'info', 'pc': None,
                   'message': 'the device IR for the inline-asm clobber check could not be produced: '
                              + (ir['stderr'].strip() or f"exit {ir['returncode']}")})
    return results, src_findings + notes + n2


FAILING = {'error'}


def render(report: dict, strict: bool) -> str:
    lines = []

    def emit(f, prefix):
        sev = SEVERITY_LABEL.get(f['severity'], f['severity'])
        where = prefix
        if f.get('loc'):
            where = f"{f['loc']['file']}:{f['loc']['line']}:{f['loc'].get('column', 1)}"
        elif f.get('pc') is not None and prefix:
            where = f"{prefix}+{f['pc']:#x}"
        text = f"{where}: {sev}: [{f['rule']}] {f['message']}"
        if f.get('instruction'):
            text += f"\n    {f['instruction']}"
        if f.get('source'):
            text += f"\n    see {f['source']}"
        lines.append(text)
    for f in report['findings']:
        emit(f, report['input'])
    for k in report['kernels']:
        for f in k['findings']:
            emit(f, f"{Path(k['object']).name}:{k['name']}")
    c = report['counts']
    om = report['operand_model']
    nf = sum(1 for k in report['kernels'] if k.get('callee'))
    units = f"{len(report['kernels']) - nf} kernel(s)" + (f' + {nf} function(s)' if nf else '')
    lines.append(f"radisa check: {units}, {om['known']}/{om['known'] + om['unknown']} "
                 f"instructions in the AMD operand model; {c.get('error', 0)} error(s), "
                 f"{c.get('review', 0)} warning(s), {c.get('incomplete', 0) + c.get('info', 0)} note(s) "
                 f"-> {report['status']}")
    inc = report.get('incremental')
    if inc:
        lines[-1] += (f"; code generation: {inc['compiled']}/{inc['kernels']} kernel(s) compiled, "
                      f"{inc['reused']} from cache ({inc['mode']})")
    lines[-1] += suppress.summary_suffix(report)
    return '\n'.join(lines)


NOT_CHECKED = ['numerical correctness',
               'waits that only a memory ordering asks for (atomics, fences, global_inv/global_wb, '
               'cross-device release/acquire): the ordering is not in the machine code',
               'an s_endpgm missing in the middle of a kernel (the code falls into its next block)',
               's_delay_alu timing', 'callees outside the checked object, indirect branch targets',
               'per-lane register state (UNINITIALIZED_READ treats a write under partial EXEC as full)',
               'memory ordering waits at system-scope stores unless SYS_STORE_WAIT is enabled',
               'global memory safety', 'performance beyond the listed resource notes']


def register_list(text: str) -> list[str]:
    """The registers of one --liveness-entry value ("s[0:1], v0, exec"): each comma-separated
    token must be one register or range as isadb.registers reads it; ValueError otherwise (the
    argparse type of --liveness-entry, so a typo is a usage error instead of defining nothing)."""
    out = []
    for tok in (t.strip() for t in re.split(r',\s*(?![^\[]*\])', text)):
        if not tok:
            continue
        try:
            ok = isadb.REG_TOKEN.fullmatch(tok) and isadb.registers(tok)
        except FormatError:
            ok = False
        if not ok:
            raise ValueError(f'{tok!r} is not a register or register range (e.g. s[0:1], v0, ttmp2, vcc)')
        out.append(tok)
    return out


def pass_options(args) -> dict:
    """The `options` every KERNEL_PASSES entry receives: `args.pass_options` (a dict) if set,
    then --liveness-entry / --no-liveness, then the rule selection of `suppress.options_from`
    (--enable/--disable/--severity off and .radisa.json; 'enable' and 'disable' are RuleSets)."""
    opts = dict(getattr(args, 'pass_options', None) or {})
    entry = getattr(args, 'liveness_entry', None)
    if entry:
        opts['liveness_entry'] = [r for e in entry for r in register_list(e)]
    if getattr(args, 'no_liveness', False):
        opts['liveness'] = False
    if hasattr(args, 'enable') or hasattr(args, 'config'):
        sel = suppress.options_from(args)
        enable = set(sel['enable']) | (memorder.enabled(opts) if 'enable' in opts else set())
        # Only non-empty selections: the default run keeps its options (and analysis_identity).
        for key, rules_ in (('enable', enable), ('disable', set(sel['disable']))):
            if rules_:
                opts[key] = suppress.RuleSet(rules_)
    return opts


def analysis_identity(options: dict | None = None) -> str:
    """Changes whenever the analysis can: radisa's own sources, the passes, their options.

    Results cached by `incremental` are keyed on it, so a new rule or pass is never served stale.
    """
    h = hashlib.sha256()
    for f in sorted(Path(__file__).resolve().parent.glob('*.py')):
        h.update(f.name.encode() + b'\0' + f.read_bytes())
    for fn in KERNEL_PASSES:
        h.update(f'{getattr(fn, "__module__", "")}.{getattr(fn, "__qualname__", repr(fn))}'.encode())
    h.update(json.dumps(options or {}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def all_findings(report: dict) -> list[dict]:
    """Report-level findings followed by every kernel's."""
    return report['findings'] + [f for k in report['kernels'] for f in k['findings']]


def finalize(report: dict, strict: bool) -> dict:
    """(Re)compute counts, operand-model totals, status and exit_code of `report` in place.

    FAIL (1): a finding at a failing severity ('error'; 'review' too with `strict`).
    NO_KERNELS (0): the input has no gfx1201 kernel. INCOMPLETE (3): nothing or not everything
    was analysed (no kernel, a kernel without instructions, a toolchain/source-map failure, code
    that was found but not checked (INCOMPLETE_RULES at severity 'incomplete'), or bytes that were
    not decoded: DECODE / BYTE_COVERAGE). PASS (0) otherwise.
    """
    allf = all_findings(report)
    kernels = report['kernels']
    report['counts'] = dict(Counter(f['severity'] for f in allf))
    report['operand_model'] = {'known': sum(k['operand_model']['operand_model_known'] for k in kernels),
                               'unknown': sum(k['operand_model']['operand_model_unknown'] for k in kernels)}
    failing = FAILING | ({'review'} if strict else set())
    if any(f['severity'] in failing for f in allf):
        status, code = 'FAIL', 1
    elif any(f['rule'] == 'NO_KERNELS' for f in allf):
        status, code = 'NO_KERNELS', 0
    elif (not kernels or any(not k['instructions'] for k in kernels)
          or any(f['rule'] in INCOMPLETE_RULES and f['severity'] == 'incomplete' for f in allf)
          or any(f['rule'] in COVERAGE_RULES for k in kernels for f in k['findings'])):
        status, code = 'INCOMPLETE', 3
    else:
        status, code = 'PASS', 0
    report['status'], report['exit_code'] = status, code
    return report


def analyze(args) -> dict:
    """The full report for `args` (as parsed by `add_parser`), finalized; prints and writes nothing."""
    path = Path(args.input)
    if hasattr(args, 'enable') or hasattr(args, 'config'):
        suppress.validate(args)   # a bad configuration fails before anything is compiled
    table = isadb.load(args.isa_table)
    backend = select_backend(args.llvm_library, args.objdump)
    incremental = None
    options = pass_options(args)
    with tempfile.TemporaryDirectory(prefix='radisa-check-') as tmp:
        work = Path(tmp)
        if args.keep:
            work = Path(args.keep)
            work.mkdir(parents=True, exist_ok=True)
        try:
            if path.suffix in hipsrc.SOURCE_SUFFIXES and not args.no_incremental:
                src = hipsrc.source_checks(path, lambda s: isadb.cite(table, s))
                checker = functools.partial(check_object_bytes, options=options)
                checker.identity = analysis_identity(options)
                kernels, findings, incremental = check_hip_incremental(path, table, backend, args, work, checker)
                findings = src + findings
                if kernels is None:
                    kernels = []
            elif path.suffix in hipsrc.SOURCE_SUFFIXES:
                kernels, findings = check_hip(path, table, backend, args, work)
            elif path.suffix in ('.s', '.S', '.asm'):
                kernels, findings = check_assembly(path, table, backend, args, work)
            else:
                kernels, findings = [], []
                for i, data in enumerate(_objects(path)):
                    ks, notes = check_object_bytes(data, str(path), table, backend, args.kernel, work,
                                                   timeout=args.timeout, options=options)
                    kernels += ks
                    findings += notes
        except FormatError as ex:
            if 'amdhsa.kernels' in str(ex):
                kernels, findings = [], [{'rule': 'NO_KERNELS', 'severity': 'info', 'pc': None,
                                          'message': 'this translation unit emits no gfx1201 kernel '
                                                     '(templates instantiated elsewhere, or host code only)'}]
            else:
                kernels, findings = [], [{'rule': 'CHECK_FAILED', 'severity': 'incomplete', 'pc': None,
                                          'message': str(ex)}]
        except (OSError, subprocess.SubprocessError) as ex:
            kernels, findings = [], [{'rule': 'CHECK_FAILED', 'severity': 'incomplete', 'pc': None,
                                      'message': str(ex)}]
    report = {'schema': 'radisa-check-v1', 'input': str(path), 'status': None, 'exit_code': None,
              'isa_table': table['source'], 'decoder': backend, 'counts': {},
              'operand_model': {'known': 0, 'unknown': 0},
              'findings': findings, 'kernels': kernels, 'incremental': incremental,
              'not_checked': list(NOT_CHECKED)}
    for fn in REPORT_FILTERS:
        fn(report, args)
    return finalize(report, args.strict)


def run(args) -> int:
    report = analyze(args)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True, default=_jsonable) + '\n')
    fmt = getattr(args, 'format', None) or 'text'
    if fmt == 'text':
        print(render(report, args.strict))
    else:
        sys.stdout.write(formats.emit([report], fmt, args.strict))
    return report['exit_code']


def _jsonable(x):
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _liveness_entry(text: str) -> str:
    try:
        register_list(text)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(str(ex)) from None
    return text


def add_parser(sub) -> None:
    c = sub.add_parser('check', help='lint one HIP source, assembly file or code object for gfx1201')
    c.add_argument('input')
    c.add_argument('--kernel', help='only kernels whose name contains this substring')
    c.add_argument('--json', help='also write the full report as JSON')
    c.add_argument('--strict', action='store_true', help='warnings fail too')
    c.add_argument('--cxx', help='HIP compiler (default: $RADISA_HIPCC, ROCm clang++, hipcc, clang++)')
    c.add_argument('--flag', action='append', help='extra compiler flags, e.g. --flag "-I include -DFOO"')
    c.add_argument('--compile-commands', help='compile_commands.json to take the real flags from')
    c.add_argument('--llvm-mc', help='llvm-mc for .s input')
    c.add_argument('--wave', type=int, choices=(32, 64), default=32, help='wave size for .s snippets')
    c.add_argument('--isa-table', help='alternative table built by `radisa isadb`')
    c.add_argument('--llvm-library')
    c.add_argument('--objdump')
    c.add_argument('--keep', help='keep intermediate .s/.o files in this directory')
    c.add_argument('--timeout', type=float, default=600, help='seconds for each compiler/disassembler step (default 600)')
    c.add_argument('--no-incremental', action='store_true',
                   help='HIP: compile the whole unit with clang -S instead of per-kernel llc with a cache')
    c.add_argument('--jobs', type=int, help='parallel per-kernel code generation (default: CPU count)')
    c.add_argument('--cache-dir', help='per-kernel result cache for this unit (default: $RADISA_CACHE or ~/.cache/radisa/<unit>)')
    c.add_argument('--format', choices=formats.FORMATS, default='text',
                   help='stdout format: text (default), json (the report), sarif (2.1.0) or github (workflow commands)')
    c.add_argument('--liveness-entry', action='append', metavar='REGS', type=_liveness_entry,
                   help='UNINITIALIZED_READ for a snippet: registers defined at its entry, e.g. "s[0:1],v0" (repeatable)')
    c.add_argument('--no-liveness', action='store_true', help='do not run UNINITIALIZED_READ')
    suppress.add_arguments(c)
    c.set_defaults(fn=run)
    h = sub.add_parser('hook', help='Claude Code PostToolUse hook: check the edited file, exit 2 on errors')
    h.set_defaults(fn=hook)
    b = sub.add_parser('isadb', help='rebuild the RDNA4 operand table from AMD XML (+ manual JSON)')
    b.add_argument('--xml', required=True)
    b.add_argument('--manual', help='RDNA4 ISA guide text dump (JSON); adds page citations')
    b.add_argument('--out', default=str(isadb.DATA))
    b.set_defaults(fn=build_table)
    suppress.add_rules_parser(sub)


def build_table(args) -> int:
    t = isadb.build(args.xml, args.manual)
    Path(args.out).write_text(json.dumps(t, separators=(',', ':'), sort_keys=True) + '\n')
    print(f"{len(t['instructions'])} instructions, {len(t['aliases'])} aliases, "
          f"{len(t['sections'])} manual sections -> {args.out}")
    return 0


def hook(args) -> int:
    """PostToolUse hook. Reads the hook event (JSON) on stdin.

    Environment:
      RADISA_HOOK_INCLUDE   regex the edited path must match (default: \\.(hip|s)$)
      RADISA_COMPILE_COMMANDS, RADISA_FLAGS, RADISA_HIPCC   as for `radisa check`
    Exit 2 hands the findings to the model (stderr): errors in the file (check exit 1), or a
    project configuration the check refuses to run with (check exit 2: a bad .radisa.json, a missing
    baseline). Everything else never blocks: a clean file, an incomplete analysis (exit 3, reported
    on stderr), and environment values the hook itself cannot turn into a valid command line.
    """
    import contextlib
    import io
    import os
    import sys
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    path = (event.get('tool_input') or {}).get('file_path') or ''
    if not path or not re.search(os.environ.get('RADISA_HOOK_INCLUDE', r'\.(hip|s)$'), path):
        return 0
    # `--opt=value`: a value that starts with '-' (RADISA_FLAGS=-O2) is not taken for an option.
    argv = ['check', path if not path.startswith('-') else './' + path, f'--json={os.devnull}']
    for env, opt in (('RADISA_COMPILE_COMMANDS', '--compile-commands'), ('RADISA_FLAGS', '--flag'),
                     ('RADISA_HIPCC', '--cxx')):
        if os.environ.get(env):
            argv.append(f'{opt}={os.environ[env]}')
    from .cli import main
    buf, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = main(argv)
    except SystemExit as ex:   # argparse: the command line built from the environment is unusable
        print('radisa hook: could not build a `radisa check` command from the environment '
              f'(exit {ex.code}); the file was not checked:\n' + err.getvalue(), file=sys.stderr)
        return 0
    if code == 1:
        print(buf.getvalue() + '\nFix every error above (warnings are advisory), then save again.',
              file=sys.stderr)
        return 2
    if code == 2:
        print('radisa check could not run on this file (configuration or usage error); fix it, then save '
              'again:\n' + err.getvalue() + buf.getvalue(), file=sys.stderr)
        return 2
    if code == 3:
        print('radisa check could not analyse this file:\n' + buf.getvalue(), file=sys.stderr)
    return 0


# The registration point of the rule modules.
KERNEL_PASSES += [memorder.kernel_pass,   # SYS_STORE_WAIT, opt-in (--enable SYS_STORE_WAIT)
                  liveness.kernel_pass]   # UNINITIALIZED_READ
# Order: configuration (severity/disable) -> inline suppression -> baseline, all in one filter;
# finalize() runs after every filter.
REPORT_FILTERS += [suppress.report_filter]
