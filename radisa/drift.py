"""`radisa drift`: detect changes in the LLVM AMDGPU source the radisa rules mirror.

Several rules restate decisions of LLVM's AMDGPU backend for gfx1201 (which hazards it fixes,
which subtarget predicates are true, which features FeatureISAVersion12 enables, where
SIInsertWaitcnts puts s_nop / s_sendmsg). Nothing in the linter notices when LLVM changes such a
decision. radisa/data/llvm_pins.json lists, per rule family, each LLVM source fact a rule depends
on: a file (relative to llvm/lib/Target/AMDGPU), an anchor and an extractor, and the sha256 of
the extracted text after normalisation. `radisa drift --llvm-src DIR` re-extracts every fact
from DIR and names, for each fact whose text changed, the radisa rules to re-review.

Extractors (all work on a token stream; comments are dropped and string, character and raw string
literals are single tokens, so braces or `//` inside them never count):
  file            the whole file.
  cxx_function    the definition (not a declaration or call) of a C++ function: from its name
                  (`Class::name` or `name`, e.g. an inline predicate in a class body) through the
                  closing brace of its body. Exactly one definition must exist.
  td_def          a TableGen `def NAME`: through the matching `}` of its body, or the `;` that ends
                  it at bracket depth 0.
  statement       a statement whose first tokens are the anchor (a code fragment, matched token by
                  token, so its spacing is irrelevant): through the end of its braced body,
                  including a following `else if` / `else` chain, or through its `;`. Exactly one
                  match must exist.

Normalisation: tokens joined without whitespace, except a single space between two adjacent word
tokens. Reformatting, comments and line moves therefore do not count as drift; any change to a
token does. A drifted hash says only that the text changed, not that the behaviour did: the
listed rules need a human re-review against the new source.

Exit codes: 0 no drift; 1 at least one fact changed; 3 a file or anchor is missing, ambiguous or
unbalanced (the check is incomplete). `--update` rewrites the hashes and the provenance, and
refuses (exit 3, nothing written) while any anchor is missing.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

PINS = Path(__file__).resolve().parent / 'data' / 'llvm_pins.json'
SCHEMA = 'radisa-llvm-pins/1'
EXTRACTORS = ('file', 'cxx_function', 'td_def', 'statement')
FIELDS = ('id', 'family', 'file', 'anchor', 'extractor', 'sha256', 'rules', 'note')
SUBDIR = Path('llvm', 'lib', 'Target', 'AMDGPU')


class AnchorError(Exception):
    """The fact cannot be extracted: missing file, missing/ambiguous anchor, unbalanced text."""


_TOKEN = re.compile(r'''
    (?P<ws>\s+)
  | (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<raw>(?:u8|[uUL])?R"(?P<delim>[^()\\\s]{0,16})\(.*?\)(?P=delim)")
  | (?P<str>(?:u8|[uUL])?"(?:[^"\\\n]|\\.)*")
  | (?P<chr>(?:u8|[uUL])?'(?:[^'\\\n]|\\.)+')
  | (?P<num>\.?\d(?:'?[\w.]|[eEpP][+-])*)
  | (?P<word>[A-Za-z_$][\w$]*)
  | (?P<code>\[\{|\}\])
  | (?P<punct>.)
''', re.S | re.X)


def tokenize(text: str, tablegen: bool = False) -> list[tuple[str, int]]:
    """C++/TableGen tokens as (text, line). Comments vanish; literals stay one token each.

    With `tablegen`, a code block `[{ ... }]` becomes one opaque token, its C++ body normalised
    (so braces inside it never count); in C++ `[{` and `}]` are two tokens each. An
    unterminated block comment or literal raises AnchorError instead of silently swallowing the
    rest of the file."""
    out, line, pos = [], 1, 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        kind, tok = m.lastgroup, m.group(0)
        if kind == 'delim': kind = 'raw'
        if kind == 'punct' and tok == '/' and text.startswith('/*', pos):
            raise AnchorError(f'unterminated block comment at line {line}')
        if kind == 'punct' and tok in '"\'':
            raise AnchorError(f'unterminated literal at line {line}')
        if kind == 'code' and tablegen and tok == '[{':
            end = text.find('}]', pos + 2)
            if end < 0: raise AnchorError(f'unterminated [{{ code block at line {line}')
            tok = text[pos:end + 2]
            out.append(('[{' + normalize(tokenize(tok[2:-2])) + '}]', line))
            line += tok.count('\n'); pos = end + 2; continue
        if kind == 'code': out.extend([(tok[0], line), (tok[1], line)])
        elif kind not in ('ws', 'lc', 'bc'): out.append((tok, line))
        line += tok.count('\n'); pos = m.end()
    return out


def _wordy(t: str) -> bool:
    return t[0].isalnum() or (t[0] in '_$.' and len(t) > 1) or t[-1] in '"\''


def normalize(tokens: list[tuple[str, int]] | list[str]) -> str:
    """Tokens joined without whitespace; one space only between two adjacent word tokens."""
    out, prev = [], ''
    for t in tokens:
        t = t[0] if isinstance(t, tuple) else t
        if prev and _wordy(prev) and _wordy(t): out.append(' ')
        out.append(t); prev = t
    return ''.join(out)


OPEN = {'(': ')', '[': ']', '{': '}'}
CLOSE = {v: k for k, v in OPEN.items()}


def match_close(toks: list[tuple[str, int]], i: int) -> int:
    """Index of the bracket that closes toks[i] (which must open one), nesting-checked."""
    stack = []
    for j in range(i, len(toks)):
        t = toks[j][0]
        if t in OPEN: stack.append(t)
        elif t in CLOSE:
            if not stack or stack[-1] != CLOSE[t]:
                raise AnchorError(f'unbalanced {t!r} at line {toks[j][1]}')
            stack.pop()
            if not stack: return j
    raise AnchorError(f'{toks[i][0]!r} opened at line {toks[i][1]} is never closed')


def _find_seq(toks: list[tuple[str, int]], seq: list[str]) -> list[int]:
    n = len(seq)
    return [i for i in range(len(toks) - n + 1)
            if toks[i][0] == seq[0] and all(toks[i + k][0] == seq[k] for k in range(1, n))]


_QUALIFIERS = {'const', 'override', 'final', 'noexcept', 'volatile', '&', 'mutable'}


def _definition_body(toks: list[tuple[str, int]], i: int) -> int | None:
    """toks[i] is '(' after a function name: index of the body's '{' if this is a definition."""
    j = match_close(toks, i) + 1
    arrow = False
    while j < len(toks):
        t = toks[j][0]
        if t == '{': return j
        if t in _QUALIFIERS: j += 1; continue
        if t == '(' and toks[j - 1][0] == 'noexcept': j = match_close(toks, j) + 1; continue
        if t == '-' and j + 1 < len(toks) and toks[j + 1][0] == '>': arrow = True; j += 2; continue
        if arrow and (t[0].isalnum() or t[0] == '_' or t in ':<>*&,'): j += 1; continue
        return None
    return None


def extract_cxx_function(toks: list[tuple[str, int]], anchor: str) -> tuple[int, int]:
    seq = [t for t, _ in tokenize(anchor)]
    hits = []
    for i in _find_seq(toks, seq):
        k = i + len(seq)
        if k >= len(toks) or toks[k][0] != '(': continue
        if i and toks[i - 1][0] in ('.', '>', '~'): continue          # member call / dtor
        body = _definition_body(toks, k)
        if body is not None: hits.append((i, match_close(toks, body)))
    if not hits: raise AnchorError(f'no definition of {anchor}')
    if len(hits) > 1:
        raise AnchorError(f'{len(hits)} definitions of {anchor} (lines '
                          + ', '.join(str(toks[a][1]) for a, _ in hits) + ')')
    return hits[0]


def _statement_end(toks: list[tuple[str, int]], i: int) -> int:
    """From toks[i], the end of the statement: the first ';' at depth 0, or the '}' closing the
    first braced block at depth 0 (then any `else` chain)."""
    j = i
    while j < len(toks):
        t = toks[j][0]
        if t == ';': return j
        if t == '{':
            end = match_close(toks, j)
            if end + 1 < len(toks) and toks[end + 1][0] == 'else':
                j = end + 2; continue
            return end
        if t in ('(', '['): j = match_close(toks, j) + 1; continue
        if t in CLOSE: raise AnchorError(f'statement ends in an unmatched {t!r} at line {toks[j][1]}')
        j += 1
    raise AnchorError(f'statement starting at line {toks[i][1]} never ends')


def extract_td_def(toks: list[tuple[str, int]], anchor: str) -> tuple[int, int]:
    hits = [i for i in _find_seq(toks, ['def', anchor])
            if i + 2 < len(toks) and toks[i + 2][0] in (':', '{', ';', '<')]
    if not hits: raise AnchorError(f'no TableGen def {anchor}')
    if len(hits) > 1: raise AnchorError(f'{len(hits)} TableGen defs named {anchor}')
    j = hits[0]
    while toks[j][0] not in ('{', ';'):
        if toks[j][0] in ('(', '['): j = match_close(toks, j)
        elif toks[j][0] in CLOSE: raise AnchorError(f'unbalanced {toks[j][0]!r} at line {toks[j][1]}')
        j += 1
        if j >= len(toks): raise AnchorError(f'def {anchor} never ends')
    return hits[0], match_close(toks, j) if toks[j][0] == '{' else j


def extract_statement(toks: list[tuple[str, int]], anchor: str) -> tuple[int, int]:
    seq = [t for t, _ in tokenize(anchor)]
    if not seq: raise AnchorError('empty statement anchor')
    hits = _find_seq(toks, seq)
    if not hits: raise AnchorError(f'no statement starting with {normalize(seq)}')
    if len(hits) > 1:
        raise AnchorError(f'{len(hits)} statements start with {normalize(seq)} (lines '
                          + ', '.join(str(toks[i][1]) for i in hits) + ')')
    return hits[0], _statement_end(toks, hits[0])


def source_root(src: Path) -> Path:
    """llvm-project checkout, its llvm/ dir, or the AMDGPU directory (or a flat copy) itself."""
    for cand in (src / SUBDIR, src / Path(*SUBDIR.parts[1:])):
        if cand.is_dir(): return cand
    return src


def extract(root: Path, fact: dict, cache: dict | None = None) -> tuple[str, int]:
    """(normalised text, first line) of one fact under the AMDGPU source directory `root`."""
    cache = {} if cache is None else cache
    path = root / fact['file']
    if fact['file'] not in cache:
        if not path.is_file(): raise AnchorError(f'file {fact["file"]} not found under {root}')
        cache[fact['file']] = tokenize(path.read_text(encoding='utf-8', errors='replace'),
                                       tablegen=path.suffix == '.td')
    toks = cache[fact['file']]
    kind = fact['extractor']
    if kind == 'file': a, b = 0, len(toks) - 1
    elif kind == 'cxx_function': a, b = extract_cxx_function(toks, fact['anchor'])
    elif kind == 'td_def': a, b = extract_td_def(toks, fact['anchor'])
    elif kind == 'statement': a, b = extract_statement(toks, fact['anchor'])
    else: raise ValueError(f'unknown extractor {kind!r}')
    if not toks: return '', 1
    return normalize(toks[a:b + 1]), toks[a][1]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_pins(path: Path = PINS) -> dict:
    pins = json.loads(Path(path).read_text())
    if pins.get('schema') != SCHEMA: raise ValueError(f'{path}: schema is not {SCHEMA}')
    seen = set()
    for f in pins['facts']:
        missing = [k for k in FIELDS if k not in f]
        if missing: raise ValueError(f'{path}: fact {f.get("id")} lacks {", ".join(missing)}')
        if f['extractor'] not in EXTRACTORS: raise ValueError(f'{path}: fact {f["id"]}: unknown extractor')
        if f['id'] in seen: raise ValueError(f'{path}: duplicate fact id {f["id"]}')
        seen.add(f['id'])
    return pins


def compare(pins: dict, src: Path) -> dict:
    """Re-extract every pinned fact from `src`. Result: {status, exit_code, facts: [...]}, each
    fact {id, family, file, anchor, line, state ('same'|'changed'|'missing'), pinned, current,
    rules, reason}."""
    root, cache, out = source_root(Path(src)), {}, []
    for f in pins['facts']:
        r = {k: f[k] for k in ('id', 'family', 'file', 'anchor', 'extractor', 'rules')}
        r['pinned'] = f['sha256']
        try:
            text, line = extract(root, f, cache)
            r.update(current=digest(text), line=line,
                     state='same' if digest(text) == f['sha256'] else 'changed')
        except AnchorError as ex:
            r.update(current=None, line=None, state='missing', reason=str(ex))
        out.append(r)
    states = {r['state'] for r in out}
    status = 'INCOMPLETE' if 'missing' in states else 'DRIFT' if 'changed' in states else 'PASS'
    return {'status': status, 'exit_code': {'PASS': 0, 'DRIFT': 1, 'INCOMPLETE': 3}[status],
            'source': str(root), 'pinned_revision': pins.get('provenance', {}).get('revision'), 'facts': out}


def update(pins: dict, src: Path, revision: str) -> tuple[dict, dict]:
    """(new pins, comparison). New pins carry the current hashes and `revision` as provenance;
    the caller must not write them when comparison['status'] is INCOMPLETE."""
    result = compare(pins, src)
    new = json.loads(json.dumps(pins))
    for f, r in zip(new['facts'], result['facts']):
        if r['current']: f['sha256'] = r['current']
    prov = new.setdefault('provenance', {})
    prov['revision'] = revision
    prov['generated_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return new, result


def write_pins(pins: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(pins, indent=2, ensure_ascii=True) + '\n')


def render(result: dict, updated: bool = False) -> str:
    lines = [f'LLVM source: {result["source"]}', f'pinned revision: {result["pinned_revision"]}']
    review: dict[str, list[str]] = {}
    for r in result['facts']:
        where = f'{r["file"]}:{r["line"]}' if r['line'] else r['file']
        if r['state'] == 'changed':
            lines.append(f'CHANGED  {r["id"]}  ({r["extractor"]} {r["anchor"]} at {where})')
            lines.append('         re-review: ' + (', '.join(r['rules']) or '(no rule yet; see the pin note)'))
            for rule in r['rules']: review.setdefault(rule, []).append(r['id'])
        elif r['state'] == 'missing':
            lines.append(f'MISSING  {r["id"]}  ({r["extractor"]} {r["anchor"]}): {r["reason"]}')
            lines.append('         cannot check: ' + (', '.join(r['rules']) or '(no rule)'))
    same = sum(r['state'] == 'same' for r in result['facts'])
    lines.append(f'{same} of {len(result["facts"])} facts unchanged')
    if review and not updated:
        lines.append('rules to re-review: ' + '; '.join(f'{k} ({", ".join(v)})' for k, v in sorted(review.items())))
    lines.append(('UPDATED' if updated else result['status']) + (' (pins rewritten)' if updated else ''))
    return '\n'.join(lines)


def run(args) -> int:
    pins_path = Path(getattr(args, 'pins', None) or PINS)
    pins = load_pins(pins_path)
    src = Path(args.llvm_src)
    if not src.is_dir():
        print(f'INCOMPLETE: --llvm-src {src} is not a directory', file=sys.stderr); return 3
    if args.update:
        new, result = update(pins, src, args.revision or 'unknown')
        if result['status'] == 'INCOMPLETE':
            print(render(result)); print('pins NOT rewritten: fix or re-anchor the missing facts first')
            return 3
        write_pins(new, pins_path)
        print(render(result, updated=True)); print(f'wrote {pins_path} (revision {new["provenance"]["revision"]})')
        code = 0
    else:
        result = compare(pins, src)
        print(render(result)); code = result['exit_code']
    if getattr(args, 'json', None):
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return code


def add_parser(sub) -> None:
    d = sub.add_parser('drift', help='detect changes in the LLVM AMDGPU source facts the check rules mirror')
    d.add_argument('--llvm-src', required=True,
                   help='llvm-project checkout, its llvm/ directory, or llvm/lib/Target/AMDGPU itself')
    d.add_argument('--update', action='store_true', help='rewrite the pinned hashes from --llvm-src')
    d.add_argument('--revision', help='with --update: source revision to record (e.g. a commit hash)')
    d.add_argument('--pins', help=f'pin file (default: {PINS})')
    d.add_argument('--json', help='also write the comparison as JSON')
    d.set_defaults(fn=run)
