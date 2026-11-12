"""HIP front end for `radisa check`: compile one translation unit for gfx1201,
map machine instructions back to source lines, and run a few source-level checks.

Only the machine code decides MISSING_WAIT and the other ISA findings. The
source checks here are lexical hints (severity review/info), never proofs.
"""
from __future__ import annotations

import bisect
import functools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from .binary import FormatError
from . import isadb

SOURCE_SUFFIXES = {'.hip', '.cu', '.cpp', '.cc', '.cxx'}
MEMORY_MNEMONIC = re.compile(
    r'\b(ds_(?!swizzle|bpermute|permute)\w+|global_\w+|flat_\w+|scratch_\w+|buffer_\w+|'
    r'image_\w+|s_load_\w+|s_buffer_load_\w+)')


# ------------------------------------------------------------------ toolchain

def find_tool(name: str, explicit: str | None = None, near: str | None = None) -> str:
    if explicit:
        p = shutil.which(explicit) if os.sep not in explicit else explicit
        if not p or not Path(p).exists():
            raise FormatError(f'{name} not found: {explicit}')
        return p
    dirs = []
    if near:
        real = Path(shutil.which(near) or near).resolve()
        dirs += [real.parent, real.parent.parent / 'llvm' / 'bin']
    rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
    dirs += [rocm / 'llvm' / 'bin', rocm / 'bin']
    for d in dirs:
        if (d / name).exists():
            return str(d / name)
    p = shutil.which(name)
    if p:
        return p
    raise FormatError(f'{name} not found; pass it explicitly or set ROCM_PATH')


def default_compiler() -> str:
    env = os.environ.get('RADISA_HIPCC')
    if env:
        return env
    rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
    for c in (rocm / 'llvm' / 'bin' / 'clang++', rocm / 'bin' / 'hipcc'):
        if c.exists():
            return str(c)
    for c in ('hipcc', 'clang++'):
        if shutil.which(c):
            return shutil.which(c)
    raise FormatError('no HIP compiler found; pass --cxx or set RADISA_HIPCC')


DROP_WITH_VALUE = {'-o', '-MF', '-MT', '-MQ', '--offload-arch', '-x'}
DROP_EXACT = {'-c', '-S', '-MD', '-MMD', '-M', '-MM', '-E', '-fsyntax-only', '--cuda-device-only',
              '--offload-device-only', '--cuda-host-only', '--offload-host-only'}


def flags_from_compile_commands(db_path: Path, source: Path) -> tuple[str, list[str], Path]:
    """Compiler and flags CMake used for this translation unit."""
    try:
        entries = json.loads(db_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as ex:
        raise FormatError(f'cannot read {db_path}: {ex}') from ex
    src = source.resolve()
    for e in entries:
        f = Path(e['directory'], e['file']).resolve()
        if f != src:
            continue
        args = e.get('arguments') or shlex.split(e['command'])
        cc, out, skip = args[0], [], False
        for a in args[1:]:
            if skip:
                skip = False
            elif a in DROP_WITH_VALUE:
                skip = True
            elif a in DROP_EXACT or a.startswith(('--offload-arch=', '-MF', '-MT', '-MQ', '-x')):
                pass
            elif a.startswith('-o') and len(a) > 2:
                pass
            elif not a.startswith('-') and Path(e['directory'], a).resolve() == src:
                pass
            else:
                out.append(a)
        return cc, out, Path(e['directory'])
    raise FormatError(f'{source} not found in {db_path}')


def hip_command(source: Path, cxx: str | None, flags: list[str], compile_commands: Path | None,
                arch: str = 'gfx1201') -> tuple[str, list[str], Path | None]:
    """Compiler, device-only arguments (without output mode and file) and working folder."""
    cwd = None
    if compile_commands:
        cxx_cc, base, cwd = flags_from_compile_commands(compile_commands, source)
        cxx = cxx or cxx_cc
        base = base + flags
    else:
        cxx = cxx or default_compiler()
        base = list(flags)
    # Later -O flags from the project win over our default -O3.
    return cxx, ['-x', 'hip', f'--offload-arch={arch}', '--offload-device-only', '-O3',
                 '-gline-tables-only', *base], cwd


_WAVE64_ASM = re.compile(r'^\s*(?:\.amdhsa_wavefront_size32\s+0\b|\.wavefront_size:\s+64\b)', re.M)


def assembler_wave_args(asm_text: str, compile_args: list[str] = ()) -> list[str]:
    """llvm-mc arguments that select the wave size of compiler output. clang's .s names only
    `.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"`: the wave size is a subtarget feature, and the
    parser takes isWave64() from the feature bits passed on the command line; the kernel
    descriptor's `.amdhsa_wavefront_size32 0` is only checked against them, it does not set them
    (AMDGPUAsmParser::ParseDirectiveAMDHSAKernel). Without `-mattr=+wavefrontsize64` llvm-mc
    rejects every wave64 operand (VCC pair, SGPR-pair carry/mask). The wave size is read from
    the kernel descriptor / metadata in the .s; a unit without a kernel falls back to the last
    -m[no-]wavefrontsize64 of the compile arguments."""
    if _WAVE64_ASM.search(asm_text):
        return ['-mattr=+wavefrontsize64']
    if '.amdhsa_kernel' not in asm_text:
        last = [a for a in compile_args if a in ('-mwavefrontsize64', '-mno-wavefrontsize64')]
        if last and last[-1] == '-mwavefrontsize64':
            return ['-mattr=+wavefrontsize64']
    return []


def compile_hip(source: Path, workdir: Path, cxx: str | None, flags: list[str],
                compile_commands: Path | None, arch: str = 'gfx1201', timeout: float = 600) -> dict:
    cxx, args, cwd = hip_command(source, cxx, flags, compile_commands, arch)
    asm = workdir / (source.stem + '.gfx1201.s')
    cmd = [cxx, *args, '-S', str(source.resolve()), '-o', str(asm)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise FormatError(f'compiler failed to run: {ex}') from ex
    result = {'command': cmd, 'returncode': p.returncode, 'stderr': p.stderr, 'asm': asm,
              'cwd': str(cwd or Path.cwd())}
    if p.returncode or not asm.exists():
        return result
    mc = find_tool('llvm-mc', near=cxx)
    obj = workdir / (source.stem + '.gfx1201.o')
    wave = assembler_wave_args(asm.read_text(encoding='utf-8', errors='replace'), args)
    a = subprocess.run([mc, '-triple=amdgcn-amd-amdhsa', f'-mcpu={arch}', '-filetype=obj', *wave,
                        str(asm), '-o', str(obj)], capture_output=True, text=True,
                       timeout=timeout, check=False)
    result.update(assembler=mc, assembler_returncode=a.returncode, assembler_stderr=a.stderr, object=obj)
    return result


def compile_ir(source: Path, workdir: Path, cxx: str | None, flags: list[str],
               compile_commands: Path | None, arch: str = 'gfx1201', timeout: float = 600) -> dict:
    """Optimised device IR of the unit as text (clang -emit-llvm -S, same arguments as compile_hip),
    for IR-level checks such as asmclobber. {'command', 'returncode', 'stderr', 'ir': str | None}."""
    cxx, args, cwd = hip_command(source, cxx, flags, compile_commands, arch)
    out = workdir / (source.stem + '.gfx1201.ll')
    cmd = [cxx, *args, '-emit-llvm', '-S', str(source.resolve()), '-o', str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise FormatError(f'compiler failed to run: {ex}') from ex
    ir = out.read_text(encoding='utf-8', errors='replace') if not p.returncode and out.exists() else None
    return {'command': cmd, 'returncode': p.returncode, 'stderr': p.stderr, 'ir': ir}


# ------------------------------------------------------------------ .s parsing

LABEL = re.compile(r'^([A-Za-z_.$][\w.$@]*):')


def parse_asm(text: str, base: str | None = None) -> dict:
    """Per function: instruction mnemonics with the source location of their `.loc` (None before
    the first one) and inline-asm flag, for compiler output. `/* ... */` comments are blanked first
    (newlines kept), so a comment line is never taken for an instruction. Hand-written assembly is
    mapped to its own lines by check.asm_line_map instead."""
    text = re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text, flags=re.S)
    files: dict[int, str] = {}
    functions: dict[str, list[dict]] = {}
    current = None
    loc = None
    in_asm = False
    in_text = True
    in_metadata = False
    for raw in text.splitlines():
        line = raw.strip()
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
            current = None
            continue
        if not in_text:
            continue
        if line.startswith(';;#ASMSTART'):
            in_asm = True
            continue
        if line.startswith(';;#ASMEND'):
            in_asm = False
            continue
        code = re.split(r';|//', line, maxsplit=1)[0].strip()
        if not code:
            continue
        if code.startswith('.file'):
            m = re.match(r'\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?', code)
            if m:
                f = os.path.join(m[2], m[3]) if m[3] else m[2]
                files[int(m[1])] = os.path.normpath(os.path.join(base, f)) if base else f
            continue
        if code.startswith('.loc'):
            m = re.match(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)', code)
            if m and int(m[2]) > 0:
                loc = (int(m[1]), int(m[2]), int(m[3]))
            # Line 0 marks compiler-generated code; keep the last real line.
            continue
        lab = LABEL.match(code)
        if lab:
            name = lab[1]
            if name.startswith('.Lfunc_end'):
                current = None
            elif not name.startswith(('.L', '$')):
                current = name
                functions.setdefault(current, [])
                loc = None
            code = code[lab.end():].strip()
            if not code:
                continue
        if code.startswith('.') or current is None:
            continue
        mnemonic = code.split()[0]
        where = None
        if loc and loc[1] > 0:
            where = {'file': files.get(loc[0], f'file{loc[0]}'), 'line': loc[1], 'column': loc[2]}
        functions[current].append({'mnemonic': mnemonic, 'text': code, 'loc': where, 'inline_asm': in_asm})
    return functions


def _base(m: str) -> str:
    return re.sub(r'_(e32|e64|dpp|sdwa|e64_dpp)$', '', m.lower())


@functools.lru_cache(maxsize=1)
def _default_table() -> dict | None:
    try:
        return isadb.load()
    except (OSError, ValueError, FormatError):
        return None


@functools.lru_cache(maxsize=4096)
def _canonical(mnemonic: str) -> str | None:
    t = _default_table()
    return isadb.lookup(t, mnemonic)[0] if t else None


def _same(decoded: str, asm: str) -> bool:
    """Same instruction, also across the aliases the assembler accepts and the disassembler does
    not print (inline asm often keeps pre-GFX12 names: s_add_u32 is decoded as s_add_co_u32)."""
    if _base(decoded) == _base(asm) or (decoded.startswith('v_dual_') and asm.lower().startswith('v_dual_')):
        return True
    a, b = _canonical(decoded.lower()), _canonical(asm.lower())
    return a is not None and a == b


def source_map(decoded: list[dict], asm_instructions: list[dict]) -> tuple[dict, str | None]:
    """pc -> {'file','line','column','inline_asm'}; padding s_nop/s_code_end may be extra."""
    out, j = {}, 0
    for i in decoded:
        if j < len(asm_instructions) and _same(i['op'], asm_instructions[j]['mnemonic']):
            a = asm_instructions[j]
            out[i['pc']] = {**(a['loc'] or {}), 'inline_asm': a['inline_asm']}
            j += 1
        elif i['op'] in ('s_nop', 's_code_end'):
            continue
        else:
            return {}, f"assembly and object diverge at pc {i['pc']:#x} ({i['op']})"
    if j != len(asm_instructions):
        return {}, 'assembly lists instructions the object does not contain'
    return out, None


# ------------------------------------------------------------------ source checks

LEXEME = re.compile(
    r'(?<![\w])(?:u8|[uUL])?R"([^()\\\s"]{0,16})\(.*?\)\1"'      # raw string literal
    r'|"(?:[^"\\\n]|\\.)*"'                                    # string literal
    r"|(?<![\w'])'(?:[^'\\\n]|\\.)*'"                          # character literal
    r'|//[^\n]*|/\*.*?\*/', re.S)                              # comments


def _lex(raw: str) -> tuple[str, list[tuple[int, int]]]:
    """Source with comments blanked (newlines kept) and the spans of string/char literals.
    Literals are scanned first, so `//` or `/*` inside an asm string is not a comment."""
    out, strings, last = [], [], 0
    for m in LEXEME.finditer(raw):
        t = m.group(0)
        if t.startswith(('//', '/*')):
            out += [raw[last:m.start()], re.sub(r'[^\n]', ' ', t)]
            last = m.end()
        else:
            strings.append((m.start(), m.end()))
    out.append(raw[last:])
    return ''.join(out), strings


def _strip_comments(src: str) -> str:
    return _lex(src)[0]


def _line_col(src: str, pos: int) -> tuple[int, int]:
    line = src.count('\n', 0, pos) + 1
    return line, pos - (src.rfind('\n', 0, pos) + 1) + 1


# Every GNU spelling clang accepts: asm, __asm, __asm__, with volatile/inline/goto qualifiers.
ASM_STATEMENT = re.compile(
    r'(?<![\w$])(?:__asm__|__asm|asm)(?![\w$])\s*'
    r'(?:(?:volatile|__volatile__|__volatile|inline|__inline__|__inline|goto)(?![\w$])\s*)*'
    r'\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')
DIRECTIVE = re.compile(r'^[ \t]*#[ \t]*(\w*)', re.M)
DEFINE = re.compile(r'[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\()?')
IDENT = re.compile(r'(?<![\w$])[A-Za-z_]\w*(?![\w$])')
STATEMENT_KEYWORDS = {'else', 'do'}
CONTROL = {'if', 'while', 'for', 'switch'}
MAX_ALTERNATIVES = 16


class _Spans:
    """Sorted, merged half-open intervals with O(log n) membership."""

    def __init__(self, spans):
        merged: list[list[int]] = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.starts = [a for a, _ in merged]
        self.ends = [b for _, b in merged]

    def __contains__(self, pos: int) -> bool:
        i = bisect.bisect_right(self.starts, pos) - 1
        return i >= 0 and pos < self.ends[i]


def _directives(src: str) -> list[dict]:
    """Preprocessor directives (with backslash continuations): keyword, span, text; for #define
    also name, function-like and body (with its offset)."""
    out = []
    for m in DIRECTIVE.finditer(src):
        end = m.end()
        while True:
            nl = src.find('\n', end)
            if nl < 0:
                end = len(src)
                break
            if src[end:nl].rstrip(' \t\r').endswith('\\'):
                end = nl + 1
                continue
            end = nl
            break
        d = {'kind': m[1], 'start': m.start(), 'end': end, 'text': src[m.end():end].strip()}
        if m[1] == 'define':
            dm = DEFINE.match(src, m.start())
            if dm:
                d.update(name=dm[1], function=bool(dm[2]), body_start=dm.end(), body=src[dm.end():end])
        out.append(d)
    return out


def _inactive(directives: list[dict]) -> list[tuple[int, int]]:
    """Spans no compiler ever sees: `#if 0` groups and what follows `#else`/`#elif` of an `#if 1`.
    Other conditions depend on -D flags and headers and are left alone (lexical check)."""
    spans, stack = [], []          # stack: [state, start of the inactive part]
    for d in directives:
        k, cond = d['kind'], d['text']
        if k in ('if', 'ifdef', 'ifndef'):
            state = 'unknown'
            if k == 'if' and re.fullmatch(r'\(?\s*0\s*\)?', cond):
                state = 'off'
            elif k == 'if' and re.fullmatch(r'\(?\s*1\s*\)?', cond):
                state = 'on'
            stack.append([state, d['end']])
        elif k in ('else', 'elif', 'elifdef', 'elifndef') and stack:
            f = stack[-1]
            if f[0] == 'off':
                spans.append((f[1], d['start']))
                f[0] = 'on' if k == 'else' else 'unknown'
            elif f[0] in ('on', 'dead'):
                f[0], f[1] = 'dead', d['end']   # a taken branch came before: the rest is dead
        elif k == 'endif' and stack:
            f = stack.pop()
            if f[0] in ('off', 'dead'):
                spans.append((f[1], d['start']))
    return spans


def _is_asm_label(text: str, pos: int) -> bool:
    """Whether the asm keyword at `pos` is a GNU asm label of a declaration (`int x asm("x");`,
    `void f() __asm__("g");`) rather than an asm statement: a label follows a declarator, i.e. an
    identifier or a `)`/`]` that does not close an if/while/for/switch condition."""
    i = pos - 1
    while i >= 0 and text[i] in ' \t\r\n':
        i -= 1
    if i < 0:
        return False
    c = text[i]
    if c.isalnum() or c in '_$':
        j = i
        while j >= 0 and (text[j].isalnum() or text[j] in '_$'):
            j -= 1
        return text[j + 1:i + 1] not in STATEMENT_KEYWORDS
    if c not in ')]':
        return False
    depth, j = 0, i
    while j >= 0:
        if text[j] in ')]':
            depth += 1
        elif text[j] in '([':
            depth -= 1
            if depth == 0:
                break
        elif text[j] in ';{}':
            return False
        j -= 1
    if j < 0:
        return False           # e.g. the parameter list of a function-like macro: a statement
    k = j - 1
    while k >= 0 and text[k] in ' \t\r\n':
        k -= 1
    e = k
    while k >= 0 and (text[k].isalnum() or text[k] in '_$'):
        k -= 1
    return text[k + 1:e + 1] not in CONTROL


def asm_statements(src: str, strings: list[tuple[int, int]] | None = None) -> list[dict]:
    """Inline asm statements of comment-free source text: position and the alternatives of what
    is emitted there; each alternative is a list of {'template', 'macros'} (macros: the chain of
    macros it was expanded from, with the line of each definition).

    asm inside a #define body is reported where the macro is used (a macro never used emits no
    code), following macros that use such a macro. A macro defined more than once (in different
    #if branches) gives one alternative per definition, since which one is active is not known
    lexically. Not counted as uses or statements: names inside preprocessor directives and
    string literals, code in `#if 0` groups, and asm labels of declarations. Linear in the text."""
    if strings is None:
        strings = _lex(src)[1]
    newlines = [i for i, c in enumerate(src) if c == '\n']
    line_of = lambda pos: bisect.bisect_left(newlines, pos) + 1
    directives = _directives(src)
    dead = _Spans(_inactive(directives))
    excluded = _Spans([(d['start'], d['end']) for d in directives] + list(strings) +
                      [(a, b) for a, b in zip(dead.starts, dead.ends)])
    in_string = _Spans(strings)
    defs: dict[str, list[dict]] = {}
    for d in directives:
        if 'name' in d and d['start'] not in dead:
            defs.setdefault(d['name'], []).append(d)

    memo: dict[str, list[list[dict]]] = {}

    def alternatives(name: str, active: frozenset) -> list[list[dict]]:
        """Each alternative: list of {'template', 'via': [definition, ...]}."""
        if name in memo:
            return memo[name]
        out: list[list[dict]] = []
        for d in defs[name]:
            body, off = d['body'], d['body_start']
            combos = [[{'template': a[1], 'via': [d]} for a in ASM_STATEMENT.finditer(body)
                       if off + a.start() not in in_string and not _is_asm_label(body, a.start())]]
            refs = []
            for t in IDENT.finditer(body):
                n = t[0]
                if (n in defs and n != name and n not in active and n not in refs and off + t.start() not in in_string
                        and (not defs[n][0]['function'] or re.match(r'\s*\(', body[t.end():]))):
                    refs.append(n)
            for n in refs:
                sub = alternatives(n, active | {name})
                if any(sub):
                    combos = [c + [{**b, 'via': [d, *b['via']]} for b in alt] for c in combos for alt in sub]
                    combos = combos[:MAX_ALTERNATIVES]
            out += combos
        memo[name] = out
        return out

    found = []
    for m in ASM_STATEMENT.finditer(src):
        if m.start() not in excluded and not _is_asm_label(src, m.start()):
            found.append({'pos': m.start(), 'alternatives': [[{'template': m[1], 'via': []}]]})
    for t in IDENT.finditer(src):
        name = t[0]
        if name not in defs or t.start() in excluded:
            continue
        if defs[name][0]['function'] and not re.match(r'\s*\(', src[t.end():t.end() + 200]):
            continue
        alts = alternatives(name, frozenset())
        if any(alts):
            found.append({'pos': t.start(), 'alternatives': alts,
                          'definitions': [line_of(d['start']) for d in defs[name]]})
    for f in found:
        f['alternatives'] = [[{'template': b['template'],
                               'macros': [{'name': d['name'], 'line': line_of(d['start'])}
                                          for d in b['via']]} for b in alt] for alt in f['alternatives']]
    return sorted(found, key=lambda f: f['pos'])


def _no_wait_memory(template: str):
    """The first memory mnemonic of an asm template that holds no s_wait_*, else None."""
    body = ''.join(re.findall(r'"((?:[^"\\]|\\.)*)"', template))
    body = re.sub(r'\\[ntr]', ' ', body)
    mem = MEMORY_MNEMONIC.search(body)
    if mem and not re.search(r'\bs_wait_(load|store|ds|km|sample|bvh|idle)', body):
        return mem[1]
    return None


def source_checks(path: Path, cite) -> list[dict]:
    raw = path.read_text(encoding='utf-8', errors='replace')
    src, strings = _lex(raw)
    out = []

    def add(rule, severity, pos, message, section=None, **extra):
        line, col = _line_col(src, pos)
        out.append({'rule': rule, 'severity': severity, 'message': message, 'pc': None,
                    'loc': {'file': str(path), 'line': line, 'column': col},
                    'source': cite(section) if section else 'radisa HIP source check (lexical)', **extra})

    for a in asm_statements(src, strings):
        flagged = [[(b, mem) for b in alt if (mem := _no_wait_memory(b['template']))] for alt in a['alternatives']]
        hits = [x for x in flagged if x]
        if not hits:
            continue
        partial = ''
        if len(hits) < len(flagged):
            # Only some definitions of the macro issue memory without a wait.
            partial = (f'; only {len(hits)} of the {len(flagged)} definitions of the macro (lines '
                       + ', '.join(str(n) for n in a.get('definitions', [])) + ') do so, and which one '
                       'is active depends on the preprocessor configuration')
        for b, mem in hits[0]:
            via = ''
            if b['macros']:
                via = ' (expanded from ' + ' -> '.join(f"macro {m['name']} defined at line {m['line']}"
                                                         for m in b['macros']) + ')'
            add('HIP_INLINE_ASM_MEMORY', 'info' if partial else 'review', a['pos'],
                f'inline asm issues {mem} without a wait in the same asm statement{via}{partial}; the '
                'compiler does not track counters of asm memory instructions. Whether a use races it '
                'is decided on the machine code (MISSING_WAIT)', '5.7.1',
                **({'macros': b['macros']} if b['macros'] else {}))
    for m in re.finditer(r'\b__shfl(?:_xor|_up|_down)?(?:_sync)?\s*\(', src):
        depth, i, args, start = 1, m.end(), [], m.end()
        while i < len(src) and depth:
            c = src[i]
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            elif c == ',' and depth == 1:
                args.append(src[start:i]); start = i + 1
            i += 1
        args.append(src[start:i - 1])
        if args and args[-1].strip() == '64' and len(args) >= 3:
            add('HIP_WAVE64_WIDTH', 'review', m.start(),
                'shuffle width 64 on gfx1201: kernels run wave32 unless built with '
                '-mwavefrontsize64, and a width above warpSize is undefined')
    for m in re.finditer(r'\b__builtin_amdgcn_s_waitcnt\s*\(', src):
        add('HIP_LEGACY_WAITCNT', 'review', m.start(),
            'S_WAITCNT on RDNA4 is the legacy opcode, equivalent to S_WAIT_IDLE (drains every '
            'counter). Use the specific s_wait_* forms', '16.5')
    for m in re.finditer(r'\b__global__\b', src):
        tail = src[m.end():m.end() + 400]
        head = src[max(0, m.start() - 200):m.start()]
        sig = tail.split('{', 1)[0].split(';', 1)[0]
        if '(' in sig and '__launch_bounds__' not in sig and '__launch_bounds__' not in head.split(';')[-1] \
                and 'amdgpu_flat_work_group_size' not in sig:
            add('HIP_NO_LAUNCH_BOUNDS', 'info', m.start(),
                'kernel without __launch_bounds__: the compiler budgets registers for 1024 '
                'work-items per group, which can cost occupancy')
    return out
