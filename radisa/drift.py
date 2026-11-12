"""`radisa drift`: detect changes in the LLVM AMDGPU source the radisa rules mirror.

Several rules restate decisions of LLVM's AMDGPU backend for gfx1201 (which hazards it fixes,
which subtarget predicates are true, which features FeatureISAVersion12 enables, where
SIInsertWaitcnts puts s_nop / s_sendmsg). Nothing in the linter notices when LLVM changes such a
decision. radisa/data/llvm_pins.json lists, per rule family, each LLVM source fact a rule depends
on: a file (relative to llvm/lib/Target/AMDGPU), an anchor and an extractor, and the sha256 of
the extracted text after normalisation. `radisa drift --llvm-src DIR` re-extracts every fact
from DIR and names, for each fact whose text changed, the radisa rules to re-review.

Tokens: comments are dropped; string, character and raw string literals are single tokens, so
braces or `//` inside them never count; the C++ multi-character operators (`->`, `::`, `&&`, `--`,
`<<=`, ...) are single tokens; in C++ files a preprocessor directive (`#include ...`, `#error ...`,
with its backslash continuations) is one token, its body normalised like code.

Extractors:
  file            the whole file.
  cxx_function    the definition (not a declaration or call) of a C++ function: from the start of
                  its declaration (template header, specifiers such as `static` / `inline`, return
                  type) through the closing brace of its body. The anchor is the name (`Class::name`
                  or `name`, e.g. an inline predicate in a class body). Exactly one definition must
                  exist.
  td_def          a TableGen `def NAME`: through the matching `}` of its body, or the `;` that ends
                  it at bracket depth 0.
  statement       a statement whose first tokens are the anchor (a code fragment, matched token by
                  token, so its spacing is irrelevant), parsed as a C++ statement: a block; `if`
                  with its (braced or unbraced) body and any `else if` / `else` chain; `for` /
                  `while` / `switch` with its body; `do ... while (...);`; otherwise through the `;`
                  at bracket depth 0 (so lambda bodies and braced initialisers inside it count).
                  Exactly one match must exist.

A fact may list `alt_files`: further files (relative to the same directory) searched when `file`
is absent or lacks the anchor (e.g. GCNFeatures.td upstream, AMDGPU.td in older or ROCm trees).
The anchor must then be found in exactly one of them. The hash covers the extracted text only,
so the same definition in another file hashes the same.

Normalisation: tokens joined without whitespace, except a single space between two adjacent tokens
that would otherwise lex differently (two words, `a - -b` vs `a--b`, ...). Reformatting, comments
and line moves therefore do not count as drift; any change to a token does. A drifted hash says
only that the text changed, not that the behaviour did: the listed rules need a human re-review
against the new source.

Exit codes: 0 no drift; 1 at least one fact changed; 3 a file or anchor is missing, ambiguous or
unbalanced (the check is incomplete). `--update` rewrites the hashes and the provenance (revision,
time and a note, `--note` or a generated one), and refuses (exit 3, nothing written) while any
anchor is missing: re-anchor such a fact by editing its file / alt_files / anchor in the pin file.
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
FIELDS = ('id', 'family', 'file', 'anchor', 'extractor', 'sha256', 'rules', 'note')  # + optional alt_files
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
  | (?P<op>>>=|<<=|<=>|->\*|\.\.\.|::|->|\+\+|--|&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%&|^]=|\#\#|\.\*)
  | (?P<punct>.)
''', re.S | re.X)


def tokenize(text: str, tablegen: bool = False) -> list[tuple[str, int]]:
    """C++/TableGen tokens as (text, line). Comments vanish; literals stay one token each.

    With `tablegen`, a code block `[{ ... }]` becomes one opaque token, its C++ body normalised
    (so braces inside it never count); in C++ `[{` and `}]` are two tokens each. Without it, a
    `#` that starts a line begins a preprocessor directive, one token through the end of the
    line (and its backslash continuations). An unterminated block comment or literal raises AnchorError
    instead of silently swallowing the rest of the file."""
    out, line, pos = [], 1, 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        kind, tok = m.lastgroup, m.group(0)
        if kind == 'delim': kind = 'raw'
        if kind == 'punct' and tok == '#' and not tablegen and not text[text.rfind('\n', 0, pos) + 1:pos].strip():
            end = pos
            while True:
                end = text.find('\n', end)
                if end < 0: end = len(text); break
                if text[pos:end].rstrip('\r').endswith('\\'): end += 1; continue
                break
            tok = text[pos:end]
            out.append((_directive(tok), line))
            line += tok.count('\n'); pos = end; continue
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


def _directive(raw: str) -> str:
    """One preprocessor directive as a token: `#` + its body normalised like code, or, if the body
    does not lex as code (`#error don't`), with whitespace runs collapsed."""
    body = re.sub(r'\\\r?\n', ' ', raw[1:])
    try: return '#' + normalize(tokenize(body))
    except AnchorError: return '#' + ' '.join(body.split())


def _glues(left: str, t: str) -> bool:
    """True if `left` directly followed by `t` would not lex as `left` then `t`. A TableGen code
    block token never glues: its first `}]` ends it."""
    if left.startswith('[{') and left.endswith('}]') and len(left) > 3: return False
    return _TOKEN.match(left + t).end() != len(left)


def normalize(tokens: list[tuple[str, int]] | list[str]) -> str:
    """Tokens joined without whitespace; a single space only where the two tokens (or the last
    two and the next) would otherwise lex differently, so distinct token sequences give distinct
    text (`a - -b` -> `a- -b`, `a--b` -> `a--b`; `int x` keeps its space)."""
    out, prev, prev2 = [], '', ''
    for t in tokens:
        t = t[0] if isinstance(t, tuple) else t
        if prev and (_glues(prev, t) or (prev2 and _glues(prev2, prev + t))):
            out.append(' '); prev2 = ''
        else:
            prev2 = prev
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


_QUALIFIERS = {'const', 'override', 'final', 'noexcept', 'volatile', '&', '&&', 'mutable'}
_DECL_TOKENS = {'::', '<', '>', '>>', '*', '&', '&&', ',', '[', ']', '...'}


def _definition_body(toks: list[tuple[str, int]], i: int) -> int | None:
    """toks[i] is '(' after a function name: index of the body's '{' if this is a definition."""
    j = match_close(toks, i) + 1
    arrow = False
    while j < len(toks):
        t = toks[j][0]
        if t == '{': return j
        if t in _QUALIFIERS: j += 1; continue
        if t == '(' and toks[j - 1][0] == 'noexcept': j = match_close(toks, j) + 1; continue
        if t == '->': arrow = True; j += 1; continue
        if arrow and (t[0].isalnum() or t[0] == '_' or t in _DECL_TOKENS): j += 1; continue
        return None
    return None


def _decl_start(toks: list[tuple[str, int]], i: int) -> int:
    """Walk back from the function name toks[i] over the declaration's specifiers, return type,
    attributes and template header, to the token after the previous `;` `{` `}` `)` `:`,
    literal or preprocessor directive."""
    while i > 0:
        t = toks[i - 1][0]
        if not (t[0].isalnum() or t[0] in '_$' or t in _DECL_TOKENS): break
        i -= 1
    return i


def extract_cxx_function(toks: list[tuple[str, int]], anchor: str) -> tuple[int, int]:
    seq = [t for t, _ in tokenize(anchor)]
    hits = []
    for i in _find_seq(toks, seq):
        k = i + len(seq)
        if k >= len(toks) or toks[k][0] != '(': continue
        if i and toks[i - 1][0] in ('.', '->', '.*', '->*', '~'): continue   # member call / dtor
        body = _definition_body(toks, k)
        if body is not None: hits.append((_decl_start(toks, i), match_close(toks, body)))
    if not hits: raise AnchorError(f'no definition of {anchor}')
    if len(hits) > 1:
        raise AnchorError(f'{len(hits)} definitions of {anchor} (lines '
                          + ', '.join(str(toks[a][1]) for a, _ in hits) + ')')
    return hits[0]


def _expect(toks: list[tuple[str, int]], j: int, want: str) -> int:
    if toks[j][0] != want: raise AnchorError(f'expected {want!r} at line {toks[j][1]}, got {toks[j][0]!r}')
    return j


def _stmt(toks: list[tuple[str, int]], j: int) -> int:
    """Index of the last token of the C++ statement that starts at toks[j]."""
    t = toks[j][0]
    if t == '{': return match_close(toks, j)
    if t == 'if':
        k = j + 1
        if toks[k][0] == 'constexpr': k += 1
        if toks[k][0] == '!' or toks[k][0] == 'consteval':                    # if !consteval { }
            while toks[k][0] in ('!', 'consteval'): k += 1
            end = _stmt(toks, k)
        else:
            end = _stmt(toks, match_close(toks, _expect(toks, k, '(')) + 1)
        if end + 1 < len(toks) and toks[end + 1][0] == 'else': end = _stmt(toks, end + 2)
        return end
    if t in ('for', 'while', 'switch'):
        return _stmt(toks, match_close(toks, _expect(toks, j + 1, '(')) + 1)
    if t == 'else': return _stmt(toks, j + 1)
    if t == 'do':
        k = _expect(toks, _stmt(toks, j + 1) + 1, 'while')
        return _expect(toks, match_close(toks, _expect(toks, k + 1, '(')) + 1, ';')
    if t == 'try':
        end = match_close(toks, _expect(toks, j + 1, '{'))
        while end + 1 < len(toks) and toks[end + 1][0] == 'catch':
            end = match_close(toks, _expect(toks, match_close(toks, _expect(toks, end + 2, '(')) + 1, '{'))
        return end
    while True:                                   # expression / declaration statement
        t = toks[j][0]
        if t == ';': return j
        if t in OPEN: j = match_close(toks, j) + 1; continue
        if t in CLOSE: raise AnchorError(f'statement ends in an unmatched {t!r} at line {toks[j][1]}')
        j += 1


def _statement_end(toks: list[tuple[str, int]], i: int) -> int:
    """From toks[i], the index of the statement's last token (see `statement` in the module doc)."""
    try: return _stmt(toks, i)
    except IndexError: raise AnchorError(f'statement starting at line {toks[i][1]} never ends') from None


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


def _tokens(root: Path, name: str, cache: dict) -> list[tuple[str, int]]:
    if name not in cache:
        path = root / name
        if not path.is_file(): cache[name] = AnchorError(f'file {name} not found under {root}')
        else:
            try: cache[name] = tokenize(path.read_text(encoding='utf-8', errors='replace'),
                                        tablegen=path.suffix == '.td')
            except AnchorError as ex: cache[name] = AnchorError(f'{name}: {ex}')
    if isinstance(cache[name], AnchorError): raise cache[name]
    return cache[name]


def _extract_one(toks: list[tuple[str, int]], fact: dict) -> tuple[str, int]:
    kind = fact['extractor']
    if kind == 'file': a, b = 0, len(toks) - 1
    elif kind == 'cxx_function': a, b = extract_cxx_function(toks, fact['anchor'])
    elif kind == 'td_def': a, b = extract_td_def(toks, fact['anchor'])
    elif kind == 'statement': a, b = extract_statement(toks, fact['anchor'])
    else: raise ValueError(f'unknown extractor {kind!r}')
    if not toks: return '', 1
    return normalize(toks[a:b + 1]), toks[a][1]


def extract_located(root: Path, fact: dict, cache: dict | None = None) -> tuple[str, int, str]:
    """(normalised text, first line, file it was found in) of one fact under the AMDGPU source
    directory `root`, searching `file` then each of `alt_files`; the anchor must be found in
    exactly one of them."""
    cache = {} if cache is None else cache
    found, errors = [], []
    for name in [fact['file'], *fact.get('alt_files', [])]:
        try: found.append((*_extract_one(_tokens(root, name, cache), fact), name))
        except AnchorError as ex: errors.append(str(ex) if str(ex).startswith(('file ', name)) else f'{name}: {ex}')
    if len(found) > 1:
        raise AnchorError(f'{fact["anchor"]} found in {len(found)} files ('
                          + ', '.join(f'{n}:{l}' for _, l, n in found) + ')')
    if not found: raise AnchorError('; '.join(errors) if len(errors) > 1 else errors[0].removeprefix(f'{fact["file"]}: '))
    return found[0]


def extract(root: Path, fact: dict, cache: dict | None = None) -> tuple[str, int]:
    """(normalised text, first line) of one fact under the AMDGPU source directory `root`."""
    return extract_located(root, fact, cache)[:2]


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
        alt = f.get('alt_files', [])
        if not (isinstance(alt, list) and all(isinstance(a, str) and a for a in alt)):
            raise ValueError(f'{path}: fact {f["id"]}: alt_files must be a list of file names')
        seen.add(f['id'])
    return pins


def compare(pins: dict, src: Path) -> dict:
    """Re-extract every pinned fact from `src`. Result: {status, exit_code, facts: [...]}, each
    fact {id, family, file, anchor, line, found_in (the file the anchor was found in), state
    ('same'|'changed'|'missing'), pinned, current, rules, reason}."""
    root, cache, out = source_root(Path(src)), {}, []
    for f in pins['facts']:
        r = {k: f[k] for k in ('id', 'family', 'file', 'anchor', 'extractor', 'rules')}
        r['pinned'] = f['sha256']
        try:
            text, line, found_in = extract_located(root, f, cache)
            r.update(current=digest(text), line=line, found_in=found_in,
                     state='same' if digest(text) == f['sha256'] else 'changed')
        except AnchorError as ex:
            r.update(current=None, line=None, found_in=None, state='missing', reason=str(ex))
        out.append(r)
    states = {r['state'] for r in out}
    status = 'INCOMPLETE' if 'missing' in states else 'DRIFT' if 'changed' in states else 'PASS'
    return {'status': status, 'exit_code': {'PASS': 0, 'DRIFT': 1, 'INCOMPLETE': 3}[status],
            'source': str(root), 'pinned_revision': pins.get('provenance', {}).get('revision'), 'facts': out}


def update(pins: dict, src: Path, revision: str, note: str | None = None) -> tuple[dict, dict]:
    """(new pins, comparison). New pins carry the current hashes and a fresh provenance:
    `revision`, the time, and `note` (default: a generated statement of where the hashes came
    from; the old note is never kept, it described the old hashes). The caller must not write
    them when comparison['status'] is INCOMPLETE."""
    result = compare(pins, src)
    new = json.loads(json.dumps(pins))
    for f, r in zip(new['facts'], result['facts']):
        if r['current']: f['sha256'] = r['current']
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    changed = [r['id'] for r in result['facts'] if r['state'] == 'changed']
    new['provenance'] = {'revision': revision, 'generated_utc': now, 'note': note or (
        f'Generated by `radisa drift --update` from an --llvm-src tree at revision {revision} on {now}. '
        + (f'Facts whose hash changed in that update (their rules need a re-review against this '
           f'source): {", ".join(changed)}.' if changed else 'No fact changed in that update.'))}
    return new, result


def write_pins(pins: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(pins, indent=2, ensure_ascii=True) + '\n')


def render(result: dict, updated: bool = False) -> str:
    lines = [f'LLVM source: {result["source"]}', f'pinned revision: {result["pinned_revision"]}']
    review: dict[str, list[str]] = {}
    for r in result['facts']:
        where = f'{r.get("found_in") or r["file"]}:{r["line"]}' if r['line'] else r['file']
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
        new, result = update(pins, src, args.revision or 'unknown', getattr(args, 'note', None))
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
    d.add_argument('--note', help='with --update: provenance note to record (default: a generated one)')
    d.add_argument('--pins', help=f'pin file (default: {PINS})')
    d.add_argument('--json', help='also write the comparison as JSON')
    d.set_defaults(fn=run)
