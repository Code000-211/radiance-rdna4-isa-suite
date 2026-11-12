"""Inline-asm register contracts on LLVM IR: ASM_UNDECLARED_CLOBBER.

An inline asm statement promises the compiler which registers it changes: its outputs and its
clobber list. LLVM LangRef ("Inline Assembler Expressions", Output/Input/Clobber constraints):
"It is not permitted for the asm to write to any input register or memory location (unless that
input is tied to an output)", and only registers named as outputs or clobbers (`~{reg}`) may be
changed. A register written outside that contract is assumed by the register allocator to keep
its value across the statement; whether a live value sits there is an allocation accident, so the
bug comes and goes with unrelated edits.

The check reads the IR clang emits (`call ... asm sideeffect "<template>", "<constraints>"(...)`),
replaces each operand placeholder by a register reserved for it, and asks the AMD RDNA4 operand
table (isadb.operand_model) what every template line writes, including implicit writes (VCC of
VOPC/VOP2 carry forms, SCC of SALU, EXEC of V_CMPX, M0). Written registers that are neither an
output (placeholder or explicit `={v1}`) nor a clobber are reported. Severity is 'review': the
operand model is radisa's reading of the AMD XML, not the compiler's.

Clobber names clang accepts for AMDGPU are the single registers of AMDGPUTargetInfo::GCCRegNames
(clang/lib/Basic/Targets/AMDGPU.cpp): v0-v255, s0-s127, a0-a255, exec, vcc, scc, m0, flat_scratch,
exec_lo/hi, vcc_lo/hi, flat_scratch_lo/hi. Ranges ("v[4:7]") are rejected by clang 18 ("unknown
register name"); LLVM's own constraint parser (AMDGPU::parseAsmConstraintPhysReg, used by
SITargetLowering::getRegForInlineAsmConstraint) accepts `~{v[4:7]}` in IR, so both are honoured here.
"""
from __future__ import annotations

from pathlib import Path
import re

from . import isadb
from .binary import FormatError

RULE = 'ASM_UNDECLARED_CLOBBER'
SOURCE = ('LLVM LangRef, Inline Assembler Expressions (input/clobber constraints); clobber names: '
          'clang/lib/Basic/Targets/AMDGPU.cpp AMDGPUTargetInfo::GCCRegNames; writes: AMD RDNA4 ISA XML '
          'via radisa/isadb.py operand_model')

CALL_ASM = re.compile(
    r'\b(?:call|callbr)\b(?P<ret>[^"]*?)\basm\b(?P<flags>(?:\s+(?:sideeffect|alignstack|inteldialect|unwind))*)'
    r'\s+"(?P<tmpl>[^"]*)",\s*"(?P<cons>[^"]*)"\((?P<rest>.*)$')
FMF = {'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc', 'fast', 'tail', 'notail', 'musttail',
       'noundef', 'zeroext', 'signext', 'inreg', 'fastcc', 'ccc', 'coldcc'}
IMMEDIATE = {'i', 'n', 'I', 'J', 'A', 'B', 'C', 'X', 'DA', 'DB', 'E', 'F'}
SPECIAL_NAME = {106: 'vcc_lo', 107: 'vcc_hi', 124: 'null', 125: 'm0', 126: 'exec_lo', 127: 'exec_hi'}
# What to write in the clobber list for each special register (a GCCRegNames entry).
SPECIAL_CLOBBER = {106: 'vcc', 107: 'vcc', 125: 'm0', 126: 'exec', 127: 'exec'}
PLACEHOLDER = re.compile(r'\$(?:\$|\{(\d+)(?::\w*)?\}|\{:(\w+)\}|(\d+))')


# ------------------------------------------------------------------ IR text

def ir_unescape(s: str) -> str:
    """An LLVM IR string constant's text: \\XX is a hex byte, \\\\ a backslash."""
    return re.sub(r'\\([0-9A-Fa-f]{2}|\\)', lambda m: '\\' if m[1] == '\\' else chr(int(m[1], 16)), s)


def split_top(text: str, sep: str = ',') -> list[str]:
    """Split at `sep` outside (), [], {}, <> and quotes."""
    out, depth, start, quote = [], 0, 0, False
    for i, c in enumerate(text):
        if c == '"':
            quote = not quote
        elif quote:
            continue
        elif c in '([{<':
            depth += 1
        elif c in ')]}>':
            depth -= 1
        elif c == sep and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _balanced(text: str, start: int) -> int:
    """Index after the bracket group opening at text[start]."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] in '([{<':
            depth += 1
        elif text[i] in ')]}>':
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def leading_type(text: str) -> str:
    """The IR type at the start of `text` (an argument "i32 noundef %x" -> "i32")."""
    text = text.strip()
    if not text:
        return ''
    if text[0] in '<[{':
        return text[:_balanced(text, 0)]
    m = re.match(r'ptr(?:\s+addrspace\(\d+\))?|[\w.]+', text)
    return m[0] if m else ''


def type_bits(t: str) -> int:
    """Size of an IR type in bits (pointers by address space: 3 and 5 are 32-bit on AMDGPU)."""
    t = t.strip()
    m = re.fullmatch(r'i(\d+)', t)
    if m:
        return int(m[1])
    simple = {'half': 16, 'bfloat': 16, 'float': 32, 'double': 64, 'ptr': 64}
    if t in simple:
        return simple[t]
    m = re.fullmatch(r'ptr\s+addrspace\((\d+)\)', t)
    if m:
        return 32 if m[1] in ('2', '3', '5', '6') else 64
    m = re.fullmatch(r'[<\[]\s*(\d+)\s+x\s+(.*)[>\]]', t)
    if m:
        return int(m[1]) * type_bits(m[2])
    if t.startswith('{'):
        return sum(type_bits(e) for e in split_top(t[1:-1]))
    return 32


def struct_members(t: str) -> list[str]:
    t = t.strip()
    if t.startswith('{') and t.endswith('}'):
        return split_top(t[1:-1])
    return [] if t in ('void', '') else [t]


def _call_args(rest: str) -> list[str]:
    end = _balanced('(' + rest, 0) - 1
    return split_top(rest[:end - 1] if end > 0 else rest)


def parse_constraints(cons: str) -> tuple[list[dict], list[str]]:
    """Operands (numbered as the template's $N) and clobber names of an IR constraint string."""
    operands, clobbers = [], []
    for c in split_top(cons):
        if c.startswith('~'):
            clobbers.append(c[1:].strip('{}').lower())
            continue
        out = c.startswith('=')
        body = c.lstrip('=')
        indirect = body.startswith('*')
        body = body.lstrip('*&%')
        if body.startswith('!'):
            operands.append({'kind': 'label', 'output': False, 'code': body})
            continue
        code = body.split('|')[0]
        op = {'output': out, 'indirect': indirect, 'code': code, 'early': '&' in c}
        if code.startswith('{'):
            op.update(kind='reg', reg=code.strip('{}').lower())
        elif code.isdigit():
            op.update(kind='tied', tied=int(code))
        elif code in IMMEDIATE:
            op['kind'] = 'imm'
        elif code in ('v', 'VA', 'a'):
            op['kind'] = 'v'
        elif code in ('s', 'r'):
            op['kind'] = 's'
        else:
            op['kind'] = 'mem' if 'm' in code else 'other'
        operands.append(op)
    return operands, clobbers


def clobber_regs(name: str) -> set[tuple[str, int]]:
    """Registers a clobber name covers; sN above s105 names no SGPR the backend allocates."""
    m = re.fullmatch(r's(\d+)', name)
    if m and int(m[1]) > 105:
        return set()
    try:
        return isadb.registers(name)
    except FormatError:
        return set()


def reg_name(r: tuple[str, int]) -> str:
    kind, n = r
    if kind == 'scc':
        return 'scc'
    if kind == 's' and n in SPECIAL_NAME:
        return SPECIAL_NAME[n]
    if kind == 's' and 108 <= n <= 123:
        return f'ttmp{n - 108}'
    return f'{kind}{n}'


def clobber_name(r: tuple[str, int]) -> str | None:
    kind, n = r
    if kind == 'scc':
        return 'scc'
    if kind == 's' and n > 105:
        return SPECIAL_CLOBBER.get(n)
    return f'{kind}{n}'


# ------------------------------------------------------------------ template

def template_lines(tmpl: str) -> list[tuple[int, str]]:
    """(line index in the asm string, statement) after comments; labels and directives dropped."""
    text = re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m[0]), tmpl, flags=re.S)
    out = []
    for i, line in enumerate(text.split('\n')):
        line = re.split(r';|//|\$\{:comment\}', line, maxsplit=1)[0].strip()
        while (m := re.match(r'[A-Za-z_.$][\w.$]*:', line)):   # labels (a mnemonic is followed by a space)
            line = line[m.end():].strip()
        if line and not line.startswith('.'):
            out.append((i, line))
    return out


def _reserve(kind: str, width: int, taken: set) -> str:
    """A register range of `width` not named anywhere else, standing in for an operand."""
    start, top = (192, 255) if kind == 'v' else (40, 105)
    n = start
    while n + width - 1 <= top:
        if kind == 's' and width > 1 and n % 2:
            n += 1
            continue
        rng = {(kind, k) for k in range(n, n + width)}
        if not rng & taken:
            taken |= rng
            return f'{kind}{n}' if width == 1 else f'{kind}[{n}:{n + width - 1}]'
        n += 1
    raise FormatError('no register left to stand in for an asm operand')


def bind_operands(operands: list[dict], ret: str, args: list[str], literal: set) -> list[str]:
    """Text substituted for each $N; `literal` (registers the template names) is never reused."""
    outs = struct_members(ret)
    arg_types = [leading_type(a) for a in args]
    taken = set(literal)
    texts: list[str] = []
    oi = ai = 0
    for op in operands:
        if op['kind'] == 'label':
            texts.append('0')
            continue
        if op['output'] and not op['indirect']:
            t = outs[oi] if oi < len(outs) else 'i32'
            oi += 1
        else:
            t = arg_types[ai] if ai < len(arg_types) else 'i32'
            ai += 1
        width = max(1, (type_bits(t) + 31) // 32)
        if op['kind'] == 'reg':
            texts.append(op['reg'])
        elif op['kind'] == 'tied':
            texts.append(texts[op['tied']] if op['tied'] < len(texts) else '0')
        elif op['kind'] in ('v', 's'):
            texts.append(_reserve(op['kind'], width, taken))
        else:
            texts.append('0')
    return texts


def substitute(line: str, texts: list[str]) -> str:
    def rep(m):
        if m[0] == '$$':
            return '$'
        if m[2] is not None:
            return {'uid': '0', 'private': '.L', 'comment': ';'}.get(m[2], '')
        n = int(m[1] if m[1] is not None else m[3])
        return texts[n] if n < len(texts) else '0'
    return PLACEHOLDER.sub(rep, line)


# ------------------------------------------------------------------ locations

class Locations:
    """Source location of an IR instruction from its !dbg (and !srcloc for the template line)."""

    def __init__(self, metadata: dict[str, str]):
        self.md = metadata
        self._text: dict[str, str | None] = {}

    def _fields(self, n: str) -> str:
        return self.md.get(n, '')

    def _file(self, scope: str, depth: int = 0) -> str | None:
        node = self._fields(scope)
        m = re.search(r'\bfile: !(\d+)', node)
        if m:
            f = self._fields(m[1])
            name = re.search(r'filename: "((?:[^"\\]|\\.)*)"', f)
            d = re.search(r'directory: "((?:[^"\\]|\\.)*)"', f)
            if name:
                p = ir_unescape(name[1])
                return str(Path(ir_unescape(d[1])) / p) if d and not p.startswith('/') else p
        m = re.search(r'\bscope: !(\d+)', node)
        if m and depth < 20:
            return self._file(m[1], depth + 1)
        return None

    def of(self, line: str) -> dict | None:
        m = re.search(r'!dbg !(\d+)', line)
        if not m:
            return None
        node = self._fields(m[1])
        ln, col, scope = (re.search(p, node) for p in (r'\bline: (\d+)', r'\bcolumn: (\d+)', r'\bscope: !(\d+)'))
        if not ln or not scope or int(ln[1]) == 0:
            return None
        f = self._file(scope[1])
        if not f:
            return None
        return {'file': f, 'line': int(ln[1]), 'column': int(col[1]) if col else 1}

    def srclocs(self, line: str) -> list[int]:
        m = re.search(r'!srcloc !(\d+)', line)
        if not m:
            return []
        return [int(x) for x in re.findall(r'i64 (\d+)', self._fields(m[1]))]

    def template_line(self, loc: dict | None, srclocs: list[int], index: int) -> dict | None:
        """The source line of line `index` of the asm string (clang getAsmSrcLocInfo: the raw
        location of the string literal, then of the byte after each newline). Raw locations are
        offsets in clang's source manager; their difference to the first one is a file offset when
        neither is a macro location (bit 31 set). Falls back to the statement's own line."""
        if not loc or index == 0 or index >= len(srclocs) or any(s >= 1 << 31 for s in srclocs[:index + 1]):
            return loc
        if loc['file'] not in self._text:
            try:
                self._text[loc['file']] = Path(loc['file']).read_text(encoding='utf-8', errors='replace')
            except OSError:
                self._text[loc['file']] = None
        text = self._text[loc['file']]
        if text is None:
            return loc
        lines = text.split('\n')
        if loc['line'] > len(lines):
            return loc
        pos = sum(len(l) + 1 for l in lines[:loc['line'] - 1]) + loc['column'] - 1
        quote = text.find('"', pos)
        target = quote + srclocs[index] - srclocs[0] if quote >= 0 else -1
        if quote < 0 or not quote < target <= len(text):
            return loc
        return {'file': loc['file'], 'line': text.count('\n', 0, target) + 1,
                'column': target - (text.rfind('\n', 0, target) + 1) + 1}


# ------------------------------------------------------------------ the check

def check_call(line: str, table: dict, wave: int, locs: Locations | None = None) -> list[dict]:
    """Findings for one IR line holding an inline asm call (none if it holds none)."""
    m = CALL_ASM.search(line)
    if not m:
        return []
    ret = ' '.join(w for w in m['ret'].split() if w not in FMF and not w.startswith('#'))
    tmpl, cons = ir_unescape(m['tmpl']), ir_unescape(m['cons'])
    operands, clobbers = parse_constraints(cons)
    args = _call_args(m['rest'])
    plain = PLACEHOLDER.sub(' 0 ', tmpl)
    try:
        literal = isadb.registers(plain)
    except FormatError:
        literal = set()
    try:
        texts = bind_operands(operands, ret, args, literal)
    except FormatError:
        return []
    declared: set = set()
    for c in clobbers:
        declared |= clobber_regs(c)
    stand_in: dict = {}   # register -> operand number, for placeholders bound to reserved registers
    for i, (op, t) in enumerate(zip(operands, texts)):
        try:
            regs = isadb.registers(t)
        except FormatError:
            regs = set()
        if op.get('output') and not op.get('indirect') or op['kind'] == 'tied':
            declared |= regs
        elif op['kind'] in ('v', 's'):
            for r in regs:
                stand_in[r] = i
    loc = locs.of(line) if locs else None
    srclocs = locs.srclocs(line) if locs else []
    out, seen = [], set()
    for index, stmt in template_lines(tmpl):
        text = substitute(stmt, texts)
        parts = text.split(None, 1)
        op = parts[0].lower()
        try:
            model = isadb.operand_model(table, op, parts[1] if len(parts) > 1 else '', wave)
        except FormatError:
            continue
        bad = sorted(r for r in model['writes'] - declared if r not in seen)
        if not bad:
            continue
        seen |= set(bad)
        # An unmodelled instruction is assumed to write all it names: too weak to accuse its inputs.
        inputs = sorted({stand_in[r] for r in bad if r in stand_in}) if model['known'] else []
        named = [r for r in bad if r not in stand_in]
        if not model['known'] and not named:
            continue   # an unmodelled instruction naming only its operands: nothing to say
        names = ', '.join(reg_name(r) for r in named)
        what = []
        if named:
            add = sorted({c for c in (clobber_name(r) for r in named) if c})
            hint = ('add ' + ', '.join(f'"{c}"' for c in add) + ' to the clobber list'
                    + (' or make it an output' if any(r[0] in 'vs' and r[1] < 106 for r in named) else '')
                    if add else 'clang accepts no clobber name for it; do not write it in inline asm')
            what.append(f'writes {names}, which is neither an output nor in the clobber list ({hint})')
        if inputs:
            what.append('writes input operand ' + ', '.join(f'${n}' for n in inputs) +
                        ' (inputs must not be modified unless tied to an output: use "+" or a tied constraint)')
        assumed = '' if model['known'] else (
            f' ({op} is outside the AMD operand table: every register it names is assumed written)')
        f = {'rule': RULE, 'severity': 'review' if model['known'] else 'info', 'pc': None,
             'message': f'inline asm line "{stmt}" ' + '; '.join(what) + assumed +
                        '. The compiler assumes the register keeps its value across the asm statement; a '
                        'value it keeps there is overwritten (which values are kept there depends on '
                        'register allocation)',
             'source': SOURCE, 'registers': [reg_name(r) for r in named] + [f'${n}' for n in inputs],
             'asm_line': stmt,
             'constraints': cons, 'operand_model': 'xml' if model['known'] else 'assumed'}
        if inputs:
            f['input_operands'] = inputs
        where = locs.template_line(loc, srclocs, index) if locs else None
        if where:
            f['loc'] = where
        out.append(f)
    return out


def _wave(attr_text: str) -> int:
    return 64 if '+wavefrontsize64' in attr_text else 32


def check_function_ir(body: str, metadata: dict[str, str], attributes: dict[str, str], table: dict) -> list[dict]:
    """Findings for one `define ... { ... }` of a module (metadata: number -> text, attributes:
    group number -> text, as incremental.parse_module returns them)."""
    head = body.split('\n', 1)[0]
    groups = re.findall(r'#(\d+)', head.split('(', 1)[-1])
    wave = _wave(' '.join(attributes.get(g, '') for g in groups))
    locs = Locations(metadata)
    out = []
    for line in body.split('\n'):
        if ' asm ' in line:
            out += check_call(line, table, wave, locs)
    return out


def check_ir_text(ir: str, table: dict, source_path: str | Path | None = None) -> list[dict]:
    """ASM_UNDECLARED_CLOBBER findings of a whole textual IR module (llvm-dis or clang -emit-llvm -S).

    Each finding carries 'function' (symbol) and 'kernel' (amdgpu_kernel or not). Without a !dbg
    location it is placed at `source_path` (no line) when given."""
    metadata, attributes, functions = {}, {}, []
    cur: list[str] | None = None
    for line in ir.split('\n'):
        if cur is not None:
            cur.append(line)
            if line == '}':
                functions.append('\n'.join(cur))
                cur = None
        elif line.startswith('define '):
            cur = [line]
        elif line.startswith('!'):
            m = re.match(r'!(\d+) = (.*)$', line)
            if m:
                metadata[m[1]] = m[2]
        elif line.startswith('attributes #'):
            n, _, rest = line[len('attributes #'):].partition(' = ')
            attributes[n] = rest
    out = []
    for body in functions:
        head = body.split('\n', 1)[0]
        name = re.search(r'@("(?:[^"\\]|\\.)*"|[-\w.$]+)', head)
        fn = name[1].strip('"') if name else '?'
        for f in check_function_ir(body, metadata, attributes, table):
            f.update(function=fn, kernel=' amdgpu_kernel ' in head)
            if 'loc' not in f and source_path:
                f['message'] += f' (in {fn}; no debug location: compile with -g or -gline-tables-only)'
            out.append(f)
    return out


def attach(results: list[dict], findings: list[dict]) -> list[dict]:
    """Append each finding to the kernel result of its function; returns those left over
    (asm in a non-kernel function, or a kernel not among `results`)."""
    by_name = {r['name']: r for r in results}
    rest = []
    for f in findings:
        r = by_name.get(f.get('function'))
        if r is not None:
            r['findings'].append(f)
        else:
            rest.append(f)
    return rest
