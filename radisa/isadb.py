"""RDNA4 instruction table derived from AMD's machine-readable ISA XML.

`build()` turns AMD's `amdgpu_isa_rdna4.xml` (MIT licensed) into the compact
`radisa/data/rdna4_isa.json` shipped with the suite. Optionally the text dump of
the RDNA4 ISA Reference Guide (PDF) contributes *page numbers and section titles
only*, so findings can cite the manual. No manual prose is copied.

`operand_model()` maps one LLVM disassembly line onto the XML operand list of
that instruction and returns the registers it reads and writes, including the
implicit EXEC/VCC/SCC/M0 operands the XML declares. An instruction the table
cannot align is reported as unknown; callers must not treat that as harmless.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .binary import FormatError

DATA = Path(__file__).resolve().parent / 'data' / 'rdna4_isa.json'
SCHEMA = 'radisa-isadb-v1'

# Operand categories. Everything not listed is an ordinary register/constant
# operand whose registers are read from the disassembly text.
CATEGORY = {
    'OPR_SDST_EXEC': 'exec', 'OPR_EXEC': 'exec', 'OPR_VCC': 'vcc',
    'OPR_SSRC_SPECIAL_SCC': 'scc', 'OPR_SDST_M0': 'm0', 'OPR_PC': 'pc',
    'OPR_GPUMEM': 'gpumem', 'OPR_DSMEM': 'dsmem',
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manual_index(path: Path, names: set[str]) -> tuple[dict, dict, dict]:
    """Section and per-instruction PDF page numbers from the manual text dump."""
    doc = json.loads(path.read_text(encoding='utf-8'))
    sections: dict[str, dict] = {}

    def walk(items):
        for item in items:
            m = re.match(r'(\d+(?:\.\d+)*)\.\s+(.*)', item.get('title', ''))
            if m:
                sections[m[1]] = {'title': m[2].strip(), 'pdf_page': item['page_number']}
            walk(item.get('children', []))
    walk(doc.get('table_of_contents', []))
    start = sections.get('16', {}).get('pdf_page') or sections.get('16.1', {}).get('pdf_page', 0)
    pages: dict[str, int] = {}
    pattern = re.compile(r'^([A-Z][A-Z0-9_]+) \d+$', re.M)
    for page in doc['pages']:
        if page['page_number'] < start:
            continue
        for m in pattern.finditer(page['text']):
            if m[1] in names and m[1] not in pages:
                pages[m[1]] = page['page_number']
    meta = {'title': doc.get('document_title'), 'release_date': doc.get('release_date'),
            'sha256': _sha256(path), 'pages': doc.get('total_pages')}
    return sections, pages, meta


def build(xml_path: str | Path, manual_path: str | Path | None = None) -> dict:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    isa = root.find('ISA')
    if isa is None or isa.find('Instructions') is None:
        raise FormatError('not an AMD machine-readable ISA XML')
    arch = isa.findtext('Architecture/ArchitectureName')
    if arch != 'AMD RDNA 4':
        raise FormatError(f'unsupported ISA XML architecture: {arch}')
    doc = root.find('Document')
    instructions: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    for ins in isa.find('Instructions'):
        name = ins.findtext('InstructionName')
        flags = [f.tag for f in ins.find('InstructionFlags') if (f.text or '').strip() == 'TRUE']
        group = ins.find('FunctionalGroup')
        forms, seen = [], set()
        for enc in ins.find('InstructionEncodings'):
            ops = []
            for o in sorted(enc.find('Operands') if enc.find('Operands') is not None else [],
                            key=lambda o: int(o.get('Order', 0))):
                io = ('r' if o.get('Input') == 'true' else '') + ('w' if o.get('Output') == 'true' else '')
                field = o.findtext('FieldName') or ''
                typ = o.findtext('OperandType') or ''
                cat = CATEGORY.get(typ, 'reg')
                if field == 'LITERAL':
                    field = 'SRC'
                ops.append([io, 1 if o.get('IsImplicit') == 'true' else 0, cat,
                            'VDST' if field in ('VDST', 'VDATA', 'SDATA', 'SDST') and 'w' in io else ''])
            key = json.dumps(ops)
            if key in seen:
                continue
            seen.add(key)
            forms.append({'enc': enc.findtext('EncodingName'), 'opcode': int(enc.findtext('Opcode')),
                          'ops': ops})
        instructions[name] = {
            'group': group.findtext('Name') if group is not None else '',
            'subgroups': sorted({s.text for s in ins.iter('Subgroup')} - {'NOT_ASSIGNED'}),
            'flags': flags, 'forms': forms}
        for al in ins.iter('AliasedInstructionNames'):
            for a in al.iter('InstructionName'):
                if a.text and a.text.strip():
                    aliases[a.text.strip()] = name
    table = {'schema': SCHEMA,
             'source': {'xml_sha256': _sha256(xml_path),
                        'xml_release_date': doc.findtext('ReleaseDate') if doc is not None else None,
                        'xml_schema_version': doc.findtext('SchemaVersion') if doc is not None else None,
                        'xml_license': doc.findtext('License') if doc is not None else None,
                        'xml_copyright': doc.findtext('Copyright') if doc is not None else None,
                        'architecture': arch},
             'instructions': dict(sorted(instructions.items())),
             'aliases': dict(sorted(aliases.items())), 'sections': {}}
    if manual_path:
        sections, pages, meta = _manual_index(Path(manual_path), set(instructions))
        table['sections'] = sections
        table['source']['manual'] = meta
        for n, p in pages.items():
            table['instructions'][n]['pdf_page'] = p
    return table


@lru_cache(maxsize=4)
def load(path: str | None = None) -> dict:
    p = Path(path) if path else DATA
    try:
        table = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError) as ex:
        raise FormatError(f'cannot read ISA table {p}: {ex}') from ex
    if table.get('schema') != SCHEMA:
        raise FormatError(f'unsupported ISA table schema in {p}')
    return table


SUFFIXES = ('_e64_dpp', '_e32', '_e64', '_dpp', '_sdwa')


def lookup(table: dict, mnemonic: str) -> tuple[str | None, dict | None]:
    name = mnemonic.upper()
    for s in SUFFIXES:
        if name.endswith(s.upper()):
            name = name[:-len(s)]
            break
    if name in table['instructions']:
        return name, table['instructions'][name]
    alias = table['aliases'].get(name)
    if alias:
        return alias, table['instructions'][alias]
    return None, None


def cite(table: dict, section: str) -> str:
    s = table.get('sections', {}).get(section)
    if not s:
        return f'RDNA4 ISA §{section}'
    return f"RDNA4 ISA §{section} {s['title']}, PDF p.{s['pdf_page']}"


# ---------------------------------------------------------------- operand model

REG_TOKEN = re.compile(
    r'(?<![\w])(?:([vs])\[(\d+):(\d+)\]|([vs])(\d+)|ttmp\[(\d+):(\d+)\]|ttmp(\d+)|'
    r'(vcc_lo|vcc_hi|vcc|exec_lo|exec_hi|exec|m0|scc))(?![\w])')
SPECIAL = {'vcc_lo': [106], 'vcc_hi': [107], 'vcc': [106, 107], 'exec_lo': [126],
           'exec_hi': [127], 'exec': [126, 127], 'm0': [125]}


def registers(text: str) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for m in REG_TOKEN.finditer(text):
        kind, lo, hi, kind1, single, tlo, thi, tsingle, special = m.groups()
        if kind:
            a, b = int(lo), int(hi)
            if a > b or b > 4096:
                raise FormatError(f'invalid register range in {text!r}')
            out.update((kind, n) for n in range(a, b + 1))
        elif kind1:
            out.add((kind1, int(single)))
        elif tlo is not None:
            out.update(('s', 108 + n) for n in range(int(tlo), int(thi) + 1))
        elif tsingle is not None:
            out.add(('s', 108 + int(tsingle)))
        elif special == 'scc':
            out.add(('scc', 0))
        else:
            out.update(('s', n) for n in SPECIAL[special])
    return out


def implicit_registers(cat: str, wave: int) -> set[tuple[str, int]]:
    if cat == 'exec':
        return {('s', 126)} if wave == 32 else {('s', 126), ('s', 127)}
    if cat == 'vcc':
        return {('s', 106)} if wave == 32 else {('s', 106), ('s', 107)}
    if cat == 'scc':
        return {('scc', 0)}
    if cat == 'm0':
        return {('s', 125)}
    return set()


def split_operands(args: str) -> list[str]:
    out, start, depth = [], 0, 0
    for i, c in enumerate(args):
        if c in '[(':
            depth += 1
        elif c in '])':
            depth -= 1
        elif c == ',' and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
    tail = args[start:].strip()
    if tail:
        out.append(tail)
    # The last positional operand carries trailing space-separated modifiers
    # ("v[2:3], off offset:16 scope:SCOPE_SYS"). Keep only the first word of it
    # unless that word is itself a modifier.
    if out:
        words = out[-1].split()
        out[-1] = words[0] if words else ''
        if ':' in out[-1] and not out[-1].startswith(('v[', 's[', 'ttmp[')) or out[-1] in MODIFIER_WORDS:
            out.pop()
    return out


MODIFIER_WORDS = {'offen', 'idxen', 'lds', 'tfe', 'r128', 'a16', 'd16', 'unorm', 'lwe', 'gds',
                  'clamp', 'div:2', 'mul:2', 'mul:4', 'nv'}

SKIPPABLE = {'exec', 'vcc'}


def _align(positional: list[str], ops: list[list], drop_vdst: bool) -> list[tuple[str, list]] | None:
    """Align printed operands to explicit XML operands, allowing LLVM's omissions.

    LLVM leaves out the EXEC destination of V_CMPX and sometimes the VCC operand
    of VOP2/VOPC; a non-returning VMEM atomic has no VDST. Nothing else may vanish.
    """
    explicit = [o for o in ops if not o[1]]
    result: list[tuple[str, list]] = []
    if drop_vdst and explicit and explicit[0][3] == 'VDST':
        result.append(('', explicit[0]))
        explicit = explicit[1:]
    n, m = len(positional), len(explicit)
    if n > m:
        return None

    def ok_skip(o):
        return o[2] in SKIPPABLE

    i = j = 0
    skips = m - n
    while j < m:
        o = explicit[j]
        if i < n and (skips == 0 or not ok_skip(o) or _looks_like(positional[i], o)):
            result.append((positional[i], o))
            i += 1
        elif skips and ok_skip(o):
            result.append(('', o))
            skips -= 1
        else:
            return None
        j += 1
    return result if i == n else None


def _looks_like(token: str, op: list) -> bool:
    t = token.lower()
    if op[2] == 'exec':
        return 'exec' in t
    if op[2] == 'vcc':
        return 'vcc' in t or bool(re.fullmatch(r's\[\d+:\d+\]|s\d+', t))
    return True


def operand_model(table: dict, op: str, args: str, wave: int = 32) -> dict:
    """Reads/writes of one disassembled instruction, or `known=False`."""
    name, entry = lookup(table, op)
    text_regs = registers(args)
    if entry is None:
        return {'known': False, 'name': None, 'reads': text_regs, 'writes': set(),
                'reason': 'mnemonic not in AMD ISA table'}
    args = re.split(r'\s(?://|;)', ' ' + args, maxsplit=1)[0].strip()
    positional = split_operands(args)
    vmem_atomic = 'ATOMIC' in entry['subgroups'] and not name.startswith('DS_')
    drop_vdst = vmem_atomic and 'TH_ATOMIC_RETURN' not in args.upper()
    for form in entry['forms']:
        aligned = _align(positional, form['ops'], drop_vdst)
        if aligned is None:
            continue
        reads, writes = set(), set()
        returns = True
        for token, o in aligned:
            regs = registers(token) if token else set()
            if not token and o[2] in SKIPPABLE:
                regs = implicit_registers(o[2], wave)
            if not token and o[3] == 'VDST':
                returns = False
            if 'r' in o[0]:
                reads |= regs
            if 'w' in o[0]:
                writes |= regs
        mem_read = mem_write = False
        for o in form['ops']:
            if not o[1]:
                continue
            if o[2] in ('gpumem', 'dsmem'):
                mem_read |= 'r' in o[0]
                mem_write |= 'w' in o[0]
                continue
            regs = implicit_registers(o[2], wave)
            if 'r' in o[0]:
                reads |= regs
            if 'w' in o[0]:
                writes |= regs
        if entry['group'] in ('VALU', 'VMEM', 'EXPORT'):
            reads |= implicit_registers('exec', wave)
        return {'known': True, 'name': name, 'entry': entry, 'enc': form['enc'],
                'reads': reads, 'writes': writes, 'returns': returns,
                'mem_read': mem_read, 'mem_write': mem_write}
    return {'known': False, 'name': name, 'entry': entry, 'reads': text_regs, 'writes': set(),
            'reason': f'{len(positional)} printed operands match no XML form of {name}'}
