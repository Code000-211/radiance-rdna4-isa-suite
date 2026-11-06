"""HIP front end for `radisa check`: compile one translation unit for gfx1201,
map machine instructions back to source lines, and run a few source-level checks.

Only the machine code decides MISSING_WAIT and the other ISA findings. The
source checks here are lexical hints (severity review/info), never proofs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from .binary import FormatError

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


def compile_hip(source: Path, workdir: Path, cxx: str | None, flags: list[str],
                compile_commands: Path | None, arch: str = 'gfx1201', timeout: float = 600) -> dict:
    cwd = None
    if compile_commands:
        cxx_cc, base, cwd = flags_from_compile_commands(compile_commands, source)
        cxx = cxx or cxx_cc
        base = base + flags
    else:
        cxx = cxx or default_compiler()
        base = list(flags)
    asm = workdir / (source.stem + '.gfx1201.s')
    cmd = [cxx, '-x', 'hip', f'--offload-arch={arch}', '--offload-device-only', '-O3', '-gline-tables-only',
           *base, '-S', str(source.resolve()), '-o', str(asm)]
    # Later -O flags from the project win over our default -O3.
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
    a = subprocess.run([mc, '-triple=amdgcn-amd-amdhsa', f'-mcpu={arch}', '-filetype=obj',
                        str(asm), '-o', str(obj)], capture_output=True, text=True,
                       timeout=timeout, check=False)
    result.update(assembler=mc, assembler_returncode=a.returncode, assembler_stderr=a.stderr, object=obj)
    return result


# ------------------------------------------------------------------ .s parsing

LABEL = re.compile(r'^([A-Za-z_.$][\w.$@]*):')


def parse_asm(text: str, base: str | None = None) -> dict:
    """Per function: instruction mnemonics with source location and inline-asm flag."""
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


def _same(decoded: str, asm: str) -> bool:
    return _base(decoded) == _base(asm) or (decoded.startswith('v_dual_') and asm.lower().startswith('v_dual_'))


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

def _strip_comments(src: str) -> str:
    def blank(m):
        return re.sub(r'[^\n]', ' ', m.group(0))
    return re.sub(r'//[^\n]*|/\*.*?\*/', blank, src, flags=re.S)


def _line_col(src: str, pos: int) -> tuple[int, int]:
    line = src.count('\n', 0, pos) + 1
    return line, pos - (src.rfind('\n', 0, pos) + 1) + 1


def source_checks(path: Path, cite) -> list[dict]:
    raw = path.read_text(encoding='utf-8', errors='replace')
    src = _strip_comments(raw)
    out = []

    def add(rule, severity, pos, message, section=None):
        line, col = _line_col(src, pos)
        out.append({'rule': rule, 'severity': severity, 'message': message, 'pc': None,
                    'loc': {'file': str(path), 'line': line, 'column': col},
                    'source': cite(section) if section else 'radisa HIP source check (lexical)'})

    for m in re.finditer(r'\basm\s*(?:volatile\s*)?\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)', src):
        body = ''.join(re.findall(r'"((?:[^"\\]|\\.)*)"', m[1]))
        body = re.sub(r'\\[ntr]', ' ', body)
        mem = MEMORY_MNEMONIC.search(body)
        if mem and not re.search(r'\bs_wait_(load|store|ds|km|sample|bvh|idle)', body):
            add('HIP_INLINE_ASM_MEMORY', 'review', m.start(),
                f'inline asm issues {mem[1]} without a wait in the same asm statement; the '
                'compiler does not track counters of asm memory instructions. Whether a use races it '
                'is decided on the machine code (MISSING_WAIT)', '5.7.1')
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
