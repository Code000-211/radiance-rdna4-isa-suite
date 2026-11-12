"""Per-kernel code generation for `radisa check` on HIP sources.

Problem: a libr4d unit holds tens to hundreds of kernels; compiling all of them for
every edit costs 10 s to minutes, although an edit usually changes few kernels.

The clang pipeline is frontend -> optimiser -> code generation. The optimiser must
see the whole unit (whether a helper is inlined depends on how many kernels call
it), but after it every Radiance kernel is one self-contained function. So:

  1. clang -emit-llvm -O3 once for the unit (the optimised module, as the build has it)
  2. a fingerprint per kernel: its IR, the globals, declarations, attributes and
     metadata it references, without debug locations; plus the codegen options
  3. llvm-extract + llc only for kernels whose fingerprint is new, in parallel
     (opt + clang where the toolchain ships neither, as ROCm's LLVM 24 does)
  4. the findings of unchanged kernels come from the cache, their source lines
     moved along with the edit (difflib over the old and new file text)

The optimised IR also feeds asmclobber (ASM_UNDECLARED_CLOBBER): its findings are added to the
kernel's result before caching, so a cached kernel reports them too. They depend only on the
kernel's IR, which the fingerprint covers.

Step 3 produces the same machine code as compiling the unit (241/241 kernels of
seven libr4d units compared). When the optimised module still has non-kernel
functions, module-level code generation (LDS layout across kernels) could differ:
then the whole module goes through llc and nothing is cached.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from .binary import FormatError
from . import __version__, asmclobber, hipsrc

SCHEMA = 'radisa-incremental-v3'   # v3: kernel results carry ASM_UNDECLARED_CLOBBER (asmclobber)
LLC_OPT = {'-O0': '-O0', '-O1': '-O1', '-O2': '-O2', '-O3': '-O3', '-Os': '-O2', '-Oz': '-O2'}


def _run(cmd: list[str], cwd=None, timeout: float = 600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise FormatError(f'{Path(cmd[0]).name} failed to run: {ex}') from ex


def codegen_options(cxx: str, args: list[str], source: Path, cwd) -> dict:
    """The options clang's own code generation gets that the IR does not carry (-mllvm, -O)."""
    p = _run([cxx, *args, '-###', '-c', str(source.resolve()), '-o', os.devnull], cwd)
    for line in p.stderr.splitlines():
        if '"-cc1"' in line and '"-fcuda-is-device"' in line:
            toks = shlex.split(line)
            mllvm = [toks[i + 1] for i, t in enumerate(toks[:-1]) if t == '-mllvm']
            opt = [t for t in toks if t in LLC_OPT]
            return {'mllvm': mllvm, 'opt': LLC_OPT[opt[-1]] if opt else '-O3'}
    raise FormatError('could not read the device compile command from `clang -###`')


class Codegen:
    """One kernel out of the optimised module, and machine code from a module.

    llvm-extract and llc where the toolchain has them. ROCm's LLVM 24 packages ship neither
    (nor does the compiler Radiance builds from them), so the fallback uses what they do ship:
    `opt -passes=internalize,globaldce` keeps the kernel, every global variable and what they
    reach, and clang compiles the IR with its optimiser off (-disable-llvm-passes), which
    leaves llc's work: code generation at the same -O level with the same -mllvm options."""

    def __init__(self, cxx: str, opts: dict):
        self.cxx, self.opt_level = cxx, opts['opt']
        self.mllvm = [o if o.startswith('-') else '-' + o for o in opts['mllvm']]
        self.tools = {t: hipsrc.find_tool(t, near=cxx) for t in ('llvm-dis', 'llvm-mc')}
        try:
            self.tools.update({t: hipsrc.find_tool(t, near=cxx) for t in ('llvm-extract', 'llc')})
            self.mode = 'llc'
        except FormatError:
            self.tools['opt'] = hipsrc.find_tool('opt', near=cxx)
            self.mode = 'clang'
        tool = self.tools['llc'] if self.mode == 'llc' else cxx
        # What a cached result depends on besides the kernel's IR.
        self.identity = [self.mode, self.opt_level, self.mllvm, _run([tool, '--version']).stdout]

    def compile(self, bc: Path, asm: Path, timeout: float) -> None:
        if self.mode == 'llc':
            cmd = [self.tools['llc'], self.opt_level, '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx1201', *self.mllvm]
        else:
            cmd = [self.cxx, '--target=amdgcn-amd-amdhsa', '-mcpu=gfx1201', self.opt_level, '-x', 'ir',
                   '-Xclang', '-disable-llvm-passes', '-S']
            for o in self.mllvm:
                cmd += ['-mllvm', o]
        r = _run([*cmd, str(bc), '-o', str(asm)], timeout=timeout)
        if r.returncode or not asm.exists():
            raise FormatError(f'{Path(cmd[0]).name} failed on {bc.name}: ' + r.stderr.strip())

    def extract(self, kernel: str, variables: list[str], bc: Path, out: Path, timeout: float) -> None:
        if self.mode == 'llc':
            cmd = [self.tools['llvm-extract'], f'--func={kernel}']
        else:
            keep = out.with_suffix('.keep')
            keep.write_text('\n'.join([kernel, *variables]) + '\n', encoding='utf-8')
            cmd = [self.tools['opt'], '-passes=internalize,globaldce', f'-internalize-public-api-file={keep}']
        x = _run([*cmd, str(bc), '-o', str(out)], timeout=timeout)
        if x.returncode or not out.exists():
            raise FormatError(f'{Path(cmd[0]).name} {kernel}: ' + x.stderr.strip())


# ------------------------------------------------------------------ module text

# Debug locations, and the source offset of an inline-asm string (used only to place asm diagnostics).
DBG = re.compile(r',\s*!(?:dbg|srcloc) !\d+|\s*!dbg !\d+')
NAME = re.compile(r'@("(?:[^"\\]|\\.)*"|[-\w.$]+)')
META_REF = re.compile(r'!(\d+)\b')
ATTR_REF = re.compile(r'#(\d+)\b')


def parse_module(text: str) -> dict:
    header, globals_, functions, declares, attributes, metadata = [], {}, {}, {}, {}, {}
    kernels, cur, body = [], None, []
    for line in text.split('\n'):
        if cur is not None:
            body.append(line)
            if line == '}':
                functions[cur] = '\n'.join(body)
                cur = None
            continue
        if line.startswith('define '):
            cur = NAME.search(line)[1]
            body = [line]
            if ' amdgpu_kernel ' in line:
                kernels.append(cur)
        elif line.startswith('declare '):
            declares[NAME.search(line)[1]] = line
        elif line.startswith('@'):
            m = NAME.match(line)
            if m:
                globals_[m[1]] = line
        elif line.startswith('attributes #'):
            n, _, rest = line[len('attributes #'):].partition(' = ')
            attributes[n] = rest
        elif line.startswith('!'):
            m = re.match(r'!(\d+) = (.*)$', line)
            if m:
                metadata[m[1]] = m[2]
            else:
                header.append(line)   # named metadata (!llvm.module.flags = !{...})
        elif line.startswith(('target ', 'source_filename')):
            if not line.startswith('source_filename'):
                header.append(line)
    return {'header': header, 'globals': globals_, 'functions': functions, 'declares': declares,
            'attributes': attributes, 'metadata': metadata, 'kernels': kernels}


class Fingerprints:
    """Fingerprints that do not move when only line numbers or metadata numbering move."""

    def __init__(self, mod: dict, salt: str):
        self.mod = mod
        self.salt = salt
        self._canon: dict[str, str] = {}
        self.header = self._meta_canon('\n'.join(mod['header']))

    def _node(self, n: str, active: set) -> str:
        if n in self._canon:
            return self._canon[n]
        content = self.mod['metadata'].get(n, '?')
        if re.match(r'(distinct )?!DI', content):
            value = '<di>'
        elif n in active:
            return '<self>'
        else:
            active.add(n)
            value = META_REF.sub(lambda m: self._node(m[1], active), content)
            active.discard(n)
        self._canon[n] = value
        return value

    def _meta_canon(self, text: str) -> str:
        return META_REF.sub(lambda m: self._node(m[1], set()), text)

    def of(self, kernel: str) -> str:
        mod, seen, parts, todo = self.mod, set(), [], [kernel]
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            text = mod['functions'].get(name) or mod['globals'].get(name) or mod['declares'].get(name)
            if text is None:
                continue
            text = '\n'.join(l for l in DBG.sub('', text).split('\n') if not l.lstrip().startswith('#dbg_'))
            parts.append(text)
            todo += [m for m in NAME.findall(text) if m != name]
        attrs = sorted({a for p in parts for a in ATTR_REF.findall(p)})
        parts += [mod['attributes'].get(a, '?') for a in attrs]
        h = hashlib.sha256()
        for chunk in (self.salt, self.header, *(self._meta_canon(p) for p in parts)):
            h.update(chunk.encode())
            h.update(b'\0')
        return h.hexdigest()

    def non_kernel_functions(self) -> list[str]:
        return [f for f in self.mod['functions'] if f not in self.mod['kernels']]


def unquote(name: str) -> str:
    return name[1:-1].replace('\\22', '"') if name.startswith('"') else name


# ------------------------------------------------------------------ line remapping

def _line_map(old: list[str], new: list[str]) -> dict[int, int]:
    out = {}
    for tag, i1, i2, j1, _ in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                out[i1 + k + 1] = j1 + k + 1
    return out


def remap(result: dict, snapshots: dict[str, str], cache_dir: Path) -> dict | None:
    """The cached kernel result with its source lines moved to the current files; None if a line vanished."""
    maps: dict[str, dict | None] = {}

    def mapper(file: str):
        if file not in maps:
            old_sha = snapshots.get(file)
            try:
                new_text = Path(file).read_text(encoding='utf-8', errors='replace')
            except OSError:
                maps[file] = None
                return None
            if old_sha is None:
                maps[file] = None
            elif hashlib.sha1(new_text.encode()).hexdigest() == old_sha:
                maps[file] = {}
            else:
                try:
                    old_text = (cache_dir / 'src' / old_sha).read_text(encoding='utf-8')
                except OSError:
                    maps[file] = None
                    return None
                maps[file] = _line_map(old_text.splitlines(), new_text.splitlines())
                maps[file]['__changed__'] = 1
        return maps[file]

    def move(loc: dict) -> bool:
        m = mapper(loc['file'])
        if m is None:
            return False
        if not m:
            return True
        if loc['line'] not in m:
            return False
        loc['line'] = m[loc['line']]
        return True

    result = json.loads(json.dumps(result))
    for f in result['findings']:
        if f.get('loc') and not move(f['loc']):
            return None
        po = f.get('producer_origin')
        if po and po.get('line'):
            old = po['line']
            if not move(po):
                return None
            name = Path(po['file']).name
            f['message'] = f['message'].replace(f'({name}:{old})', f"({name}:{po['line']})")
    return result


def files_of(result: dict) -> set[str]:
    out = set()
    for f in result['findings']:
        for loc in (f.get('loc'), f.get('producer_origin')):
            if loc and loc.get('file'):
                out.add(loc['file'])
    return out


# ------------------------------------------------------------------ driver

def _plain(x):
    return sorted(x) if isinstance(x, (set, frozenset)) else str(x)


def kernel_asm_findings(kernel: str, mod: dict, table: dict) -> list[dict]:
    """asmclobber findings of one kernel of the optimised module (helpers are inlined into it)."""
    out = asmclobber.check_function_ir(mod['functions'][kernel], mod['metadata'], mod['attributes'], table)
    for f in out:
        f.update(function=unquote(kernel), kernel=True)
    return out


def default_cache_root() -> Path:
    env = os.environ.get('RADISA_CACHE')
    if env:
        return Path(env)
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return Path(base) / 'radisa'


def check_hip_incremental(path: Path, table: dict, backend: dict, args, work: Path, check_object_bytes):
    """Returns (kernel results, notes, stats) like check.check_hip, compiling only changed kernels."""
    flags = []
    for f in args.flag or []:
        flags += f.split()
    cc = Path(args.compile_commands) if args.compile_commands else None
    cxx, cargs, cwd = hipsrc.hip_command(path, args.cxx, flags, cc)
    base_cwd = str(cwd or Path.cwd())
    bc = work / 'module.bc'
    p = _run([cxx, *cargs, '-emit-llvm', '-c', str(path.resolve()), '-o', str(bc)], cwd, args.timeout)
    if p.returncode or not bc.exists():
        return None, [{'rule': 'COMPILE', 'severity': 'error', 'pc': None,
                       'message': 'gfx1201 device compile failed:\n' + p.stderr.strip(),
                       'command': shlex.join([cxx, *cargs])}], {}
    notes = []
    if p.stderr.strip():
        notes.append({'rule': 'COMPILER_DIAGNOSTICS', 'severity': 'info', 'pc': None,
                      'message': p.stderr.strip()})
    gen = Codegen(cxx, codegen_options(cxx, cargs, path, cwd))
    tools = gen.tools
    d = _run([tools['llvm-dis'], str(bc), '-o', '-'], timeout=args.timeout)
    if d.returncode:
        raise FormatError('llvm-dis failed: ' + d.stderr.strip())
    mod = parse_module(d.stdout)
    # `identity` (check.analysis_identity): cached findings follow every change to the analysis.
    salt = json.dumps([SCHEMA, __version__, table['source']['xml_sha256'], gen.identity,
                       getattr(check_object_bytes, 'identity', None)])
    fp = Fingerprints(mod, salt)
    kernels = [k for k in mod['kernels']
               if not args.kernel or args.kernel in unquote(k)]
    if not mod['kernels']:
        return [], notes + [{'rule': 'NO_KERNELS', 'severity': 'info', 'pc': None,
                             'message': 'this translation unit emits no gfx1201 kernel '
                                        '(templates instantiated elsewhere, or host code only)'}], {}
    extra = fp.non_kernel_functions()
    variables = [unquote(g) for g in mod['globals']]
    jobs = max(1, args.jobs or os.cpu_count() or 1)

    def assemble(asm: Path, obj: Path):
        a = _run([tools['llvm-mc'], '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-filetype=obj',
                  str(asm), '-o', str(obj)], timeout=args.timeout)
        if a.returncode:
            raise FormatError('llvm-mc could not assemble the generated code: ' + a.stderr.strip())

    if extra:
        # Module-level code generation applies: whole module, no per-kernel cache.
        asm, obj = work / 'module.s', work / 'module.o'
        gen.compile(bc, asm, args.timeout)
        assemble(asm, obj)
        functions = hipsrc.parse_asm(asm.read_text(encoding='utf-8', errors='replace'), base_cwd)
        results, n2 = check_object_bytes(obj.read_bytes(), str(path), table, backend, args.kernel,
                                         work, functions, args.timeout)
        names = {r['name'] for r in results}
        n2 += asmclobber.attach(results, [f for f in asmclobber.check_ir_text(d.stdout, table, str(path))
                                          if not f['kernel'] or f['function'] in names])   # --kernel filter
        nk = sum(1 for r in results if not r.get('callee'))   # functions are checked too, not counted
        return results, notes + n2, {'mode': 'whole-module', 'codegen': gen.mode, 'kernels': nk,
                                     'compiled': nk, 'reused': 0,
                                     'reason': f'{len(extra)} non-kernel function(s) after optimisation'}

    key = hashlib.sha1(json.dumps([str(path.resolve()), cxx, cargs, base_cwd]).encode()).hexdigest()[:16]
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_root() / key
    cache_file = cache_dir / 'cache.json'
    try:
        cache = json.loads(cache_file.read_text())
        if cache.get('schema') != SCHEMA:
            cache = {}
    except (OSError, ValueError):
        cache = {}
    entries = cache.get('kernels', {})
    prints = {k: fp.of(k) for k in kernels}
    results, todo, reused = {}, [], 0
    for k in kernels:
        e = entries.get(k)
        if e and e['fingerprint'] == prints[k]:
            moved = remap(e['result'], e.get('files', {}), cache_dir)
            if moved is not None:
                results[k] = (moved, e.get('notes', []))
                reused += 1
                continue
        todo.append(k)

    def one(i_k):
        i, k = i_k
        kdir = work / f'k{i}'
        kdir.mkdir()
        kbc, asm, obj = kdir / 'k.bc', kdir / 'k.s', kdir / 'k.o'
        gen.extract(unquote(k), variables, bc, kbc, args.timeout)
        gen.compile(kbc, asm, args.timeout)
        assemble(asm, obj)
        functions = hipsrc.parse_asm(asm.read_text(encoding='utf-8', errors='replace'), base_cwd)
        rs, n = check_object_bytes(obj.read_bytes(), str(path), table, backend, None, kdir, functions,
                                   args.timeout)
        if len(rs) != 1:
            raise FormatError(f'{unquote(k)}: expected one kernel in its object, found {len(rs)}')
        rs[0]['findings'] += kernel_asm_findings(k, mod, table)
        # The same shape as a cached result: plain JSON values.
        return k, json.loads(json.dumps(rs[0], default=_plain)), json.loads(json.dumps(n, default=_plain))

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for k, r, n in pool.map(one, enumerate(todo)):
            results[k] = (r, n)
            snap = {}
            for f in files_of(r):
                try:
                    text = Path(f).read_text(encoding='utf-8', errors='replace')
                except OSError:
                    continue
                sha = hashlib.sha1(text.encode()).hexdigest()
                (cache_dir / 'src').mkdir(parents=True, exist_ok=True)
                (cache_dir / 'src' / sha).write_text(text, encoding='utf-8')
                snap[f] = sha
            entries[k] = {'fingerprint': prints[k], 'result': r, 'notes': n, 'files': snap}
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        live = {k: entries[k] for k in mod['kernels'] if k in entries}
        tmp = cache_file.with_suffix('.tmp')
        tmp.write_text(json.dumps({'schema': SCHEMA, 'source': str(path.resolve()), 'kernels': live},
                                  default=_plain))
        tmp.replace(cache_file)
    except OSError as ex:
        notes.append({'rule': 'CACHE', 'severity': 'info', 'pc': None, 'message': f'cache not written: {ex}'})
    ordered = [results[k][0] for k in kernels]
    for k in kernels:
        notes += results[k][1]
    return ordered, notes, {'mode': 'per-kernel', 'codegen': gen.mode, 'kernels': len(kernels), 'compiled': len(todo),
                            'reused': reused, 'jobs': jobs, 'cache': str(cache_dir)}
