"""`radisa check`: lint one kernel source or one code object against RDNA4 rules.

Inputs
  *.hip / *.cpp / *.cu   compiled for gfx1201 (device only, -g), then checked as machine
                         code; findings point at source lines
  *.s                    assembled with llvm-mc; text without .amdhsa_kernel is checked
                         as a single wave32 instruction sequence ("snippet")
  anything else          ELF code object, offload bundle, archive, fat binary or HIP
                         executable/library; every gfx1201 kernel inside is checked

Exit status: 0 clean, 1 findings at the failing severity, 3 analysis incomplete.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tempfile

from .binary import ELF, FormatError, extract_code_objects, kernel_inventory
from .decode import decode_object, select_backend
from . import hipsrc, isadb, rules, waits
from .incremental import check_hip_incremental

SEVERITY_LABEL = {'error': 'error', 'review': 'warning', 'incomplete': 'note', 'info': 'note'}
RULE_SECTION = {'VOPD_': '7.8', 'WMMA_': '7.12.1', 'PERMLANE': '7.2.8', 'DEALLOC': '3.3.3',
                'USE_AFTER_DEALLOC': '3.3.3', 'REACHABLE_CODE_END': '2.4', 'SGPR_PAIR': '5.8',
                'WATERFALL': '7.11', 'LAST_USE': '4.1.1', 'DYNAMIC_': '3.3.3'}
DROP_KERNEL_ISSUES = ('relocatable ELF',)


def _section(rule: str) -> str | None:
    for prefix, sec in RULE_SECTION.items():
        if rule.startswith(prefix):
            return sec
    return None


def check_kernel(kernel: dict, decoded: dict, table: dict, origin: dict | None = None) -> dict:
    kernel = {**kernel, 'issues': [m for m in kernel.get('issues', [])
                                   if not m.startswith(DROP_KERNEL_ISSUES)]}
    base = rules.audit_kernel(kernel, decoded)
    findings = [f for f in base['findings'] if f['rule'] not in ('LOCAL_MEMORY_HAZARD', 'CFG_INDIRECT')]
    calls = [f for f in base['findings'] if f['rule'] == 'CFG_INDIRECT']
    if calls:
        findings.append({'rule': 'CALLS', 'severity': 'info', 'pc': calls[0]['pc'],
                         'message': f'{len(calls)} call(s)/indirect branch(es); callee bodies are not '
                                    'analysed, counters are assumed drained on return (callee prologue)',
                         'call_pcs': [f['pc'] for f in calls]})
    for f in findings:
        sec = _section(f['rule'])
        if sec:
            f['source'] = isadb.cite(table, sec)
    ins = decoded['instructions']
    edges, reachable, _ = rules.cfg(ins)
    wave = kernel['metadata'].get('.wavefront_size', 32)
    wf, stats = waits.analyze(ins, edges, reachable, table, wave, origin)
    findings += wf
    md = kernel['metadata']
    spills = (md.get('.vgpr_spill_count') or 0) + (md.get('.sgpr_spill_count') or 0)
    scratch = md.get('.private_segment_fixed_size') or 0
    if spills:
        findings.append({'rule': 'REGISTER_SPILL', 'severity': 'review', 'pc': None,
                         'message': f"{md.get('.vgpr_spill_count', 0)} VGPR / {md.get('.sgpr_spill_count', 0)} "
                                    f'SGPR spills to scratch ({scratch} bytes per lane)',
                         'source': 'AMDGPU code object metadata'})
    elif scratch:
        findings.append({'rule': 'SCRATCH_USE', 'severity': 'review', 'pc': None,
                         'message': f'{scratch} bytes of scratch per lane without spills: a private '
                                    'array the compiler could not keep in registers (dynamic index?)',
                         'source': 'AMDGPU code object metadata'})
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
    return {'name': kernel['name'], 'symbol': kernel.get('symbol'), 'address': kernel.get('address', 0),
            'instructions': len(ins), 'resources': base['resources'],
            'metadata': {k: md.get(k) for k in ('.wavefront_size', '.vgpr_count', '.sgpr_count',
                                                '.group_segment_fixed_size', '.private_segment_fixed_size',
                                                '.vgpr_spill_count', '.sgpr_spill_count',
                                                '.max_flat_workgroup_size')},
            'operand_model': stats, 'findings': findings}


def _objects(path: Path) -> list[bytes]:
    objs, events = extract_code_objects(path.read_bytes(), str(path))
    out = [o['data'] for o in objs if o['target'] == 'gfx1201']
    if not out:
        targets = sorted({o['target'] for o in objs})
        raise FormatError(f'no gfx1201 code object in {path} (found: {targets or "none"})')
    return out


def check_object_bytes(data: bytes, label: str, table: dict, backend: dict, kernel_filter: str | None,
                       work: Path, asm_functions: dict | None = None,
                       timeout: float = 600) -> tuple[list[dict], list[dict]]:
    elf = ELF(data)
    ks, issues = kernel_inventory(elf)
    notes = [{'rule': 'OBJECT_EVIDENCE', 'severity': 'incomplete', 'message': m, 'pc': None}
             for m in issues]
    if kernel_filter:
        ks = [k for k in ks if kernel_filter in k['name'] or kernel_filter in k['symbol']]
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
                notes.append({'rule': 'SOURCE_MAP', 'severity': 'incomplete', 'pc': None,
                              'message': f"{k['name']}: no matching function in compiler assembly"})
            else:
                origin, why = hipsrc.source_map(d['instructions'], fn)
                if why:
                    notes.append({'rule': 'SOURCE_MAP', 'severity': 'incomplete', 'pc': None,
                                  'message': f"{k['name']}: {why}; findings carry pcs only"})
                    origin = None
        r = check_kernel(k, d, table, origin)
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


def check_assembly(path: Path, table: dict, backend: dict, args, work: Path):
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
        functions = hipsrc.parse_asm(text)
        return check_object_bytes(obj.read_bytes(), str(path), table, backend, args.kernel, work, functions,
                                  args.timeout)
    k, _ = snippet_kernel(obj, args.wave)
    decoded = decode_object(obj, [k], backend)
    d = decoded[k['symbol']]
    lines = [ln for ln in hipsrc.parse_asm('radisa_snippet:\n' + text).get('radisa_snippet', [])]
    origin, _ = hipsrc.source_map(d['instructions'], lines)
    # Snippet lines have no .loc; use the line numbers of the .s file itself.
    numbered = [i + 1 for i, ln in enumerate(text.splitlines())
                if ln.split(';', 1)[0].split('//', 1)[0].strip()
                and not ln.strip().startswith('.') and not hipsrc.LABEL.match(ln.strip())]
    if origin and len(numbered) == len(lines):
        it = iter(numbered)
        for pc in sorted(origin):
            origin[pc] = {'file': str(path), 'line': next(it), 'column': 1, 'inline_asm': False}
    r = check_kernel(k, d, table, origin or None)
    r['object'] = str(path)
    return [r], []


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
                                     args.kernel, work, functions, args.timeout)
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
    lines.append(f"radisa check: {len(report['kernels'])} kernel(s), {om['known']}/{om['known'] + om['unknown']} "
                 f"instructions in the AMD operand model; {c.get('error', 0)} error(s), "
                 f"{c.get('review', 0)} warning(s), {c.get('incomplete', 0) + c.get('info', 0)} note(s) "
                 f"-> {report['status']}")
    inc = report.get('incremental')
    if inc:
        lines[-1] += (f"; code generation: {inc['compiled']}/{inc['kernels']} kernel(s) compiled, "
                      f"{inc['reused']} from cache ({inc['mode']})")
    return '\n'.join(lines)


def run(args) -> int:
    path = Path(args.input)
    table = isadb.load(args.isa_table)
    backend = select_backend(args.llvm_library, args.objdump)
    incremental = None
    with tempfile.TemporaryDirectory(prefix='radisa-check-') as tmp:
        work = Path(tmp)
        if args.keep:
            work = Path(args.keep)
            work.mkdir(parents=True, exist_ok=True)
        try:
            if path.suffix in hipsrc.SOURCE_SUFFIXES and not args.no_incremental:
                src = hipsrc.source_checks(path, lambda s: isadb.cite(table, s))
                kernels, findings, incremental = check_hip_incremental(path, table, backend, args, work,
                                                                       check_object_bytes)
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
                                                   timeout=args.timeout)
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
    allf = findings + [f for k in kernels for f in k['findings']]
    counts = Counter(f['severity'] for f in allf)
    known = sum(k['operand_model']['operand_model_known'] for k in kernels)
    unknown = sum(k['operand_model']['operand_model_unknown'] for k in kernels)
    failing = FAILING | ({'review'} if args.strict else set())
    if any(f['severity'] in failing for f in allf):
        status, code = 'FAIL', 1
    elif any(f['rule'] == 'NO_KERNELS' for f in allf):
        status, code = 'NO_KERNELS', 0
    elif not kernels or any(f['rule'] in ('CHECK_FAILED', 'ASSEMBLER', 'SOURCE_MAP') and f['severity'] == 'incomplete'
                            for f in allf) or any(not k['instructions'] for k in kernels):
        status, code = 'INCOMPLETE', 3
    else:
        status, code = 'PASS', 0
    report = {'schema': 'radisa-check-v1', 'input': str(path), 'status': status,
              'isa_table': table['source'], 'decoder': backend, 'counts': dict(counts),
              'operand_model': {'known': known, 'unknown': unknown},
              'findings': findings, 'kernels': kernels, 'incremental': incremental,
              'not_checked': ['numerical correctness', 'inter-wave and cross-device memory ordering',
                              'global memory safety', 'performance beyond the listed resource notes']}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True, default=_jsonable) + '\n')
    print(render(report, args.strict))
    return code


def _jsonable(x):
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if isinstance(x, Path):
        return str(x)
    return str(x)


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
    c.set_defaults(fn=run)
    h = sub.add_parser('hook', help='Claude Code PostToolUse hook: check the edited file, exit 2 on errors')
    h.set_defaults(fn=hook)
    b = sub.add_parser('isadb', help='rebuild the RDNA4 operand table from AMD XML (+ manual JSON)')
    b.add_argument('--xml', required=True)
    b.add_argument('--manual', help='RDNA4 ISA guide text dump (JSON); adds page citations')
    b.add_argument('--out', default=str(isadb.DATA))
    b.set_defaults(fn=build_table)


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
    Exit 2 hands the findings to the model (stderr); everything else never blocks.
    """
    import os
    import sys
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    path = (event.get('tool_input') or {}).get('file_path') or ''
    if not path or not re.search(os.environ.get('RADISA_HOOK_INCLUDE', r'\.(hip|s)$'), path):
        return 0
    argv = ['check', path, '--json', os.devnull]
    if os.environ.get('RADISA_COMPILE_COMMANDS'):
        argv += ['--compile-commands', os.environ['RADISA_COMPILE_COMMANDS']]
    if os.environ.get('RADISA_FLAGS'):
        argv += ['--flag', os.environ['RADISA_FLAGS']]
    if os.environ.get('RADISA_HIPCC'):
        argv += ['--cxx', os.environ['RADISA_HIPCC']]
    from .cli import main
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    if code == 1:
        print(buf.getvalue() + '\nFix every error above (warnings are advisory), then save again.',
              file=sys.stderr)
        return 2
    if code == 3:
        print('radisa check could not analyse this file:\n' + buf.getvalue(), file=sys.stderr)
    return 0
