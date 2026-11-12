"""`radisa check-all`: lint many translation units in one run.

Every selected file goes through exactly what `radisa check FILE` does (`check.analyze`), with an
argparse.Namespace produced by the real `check` parser: every option `radisa check` has is mirrored
onto `check-all` automatically (its `--jobs` as `--check-jobs`; `--json` is replaced by `--out`).
Each file runs in its own process (its own process group, so a timeout also stops the compiler it
started). A file that crashes the checker, kills its process or exceeds `--timeout-per-file`
becomes an INCOMPLETE entry for that file; the run itself continues.

Exit status (the same meaning as `radisa check`): 1 if any file FAILs, else 3 if any file is
INCOMPLETE, else 0. `--fail-on warning` makes warnings fail (`--strict` for every file);
`--fail-on incomplete` additionally turns INCOMPLETE files into exit 1. An invalid option,
configuration or missing `--baseline` is checked once, before any file, and exits 2 as in `check`.
SIGINT / SIGTERM stop every running check (its whole process group) and exit 130 / 143.

Baselines: `--write-baseline B` writes ONE file covering every checked file; each entry carries
its file (`input`, relative to B's directory). Entries of files not checked in this run (not
selected, or crashed) are kept. `--baseline B` (or a configuration's baseline) gives each file
only its own entries, so a finding is hidden only in the file it was recorded for and
BASELINE_STALE names only that file's entries. A single-file baseline written by `radisa check`
applies to the file named in its `input` (or to the only file checked).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
from multiprocessing.connection import wait as mp_wait
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from . import __version__, check, formats, hipsrc, suppress
from .formats import emit  # noqa: F401  (re-exported: batch.emit(reports, fmt))

SCHEMA = 'radisa-check-all-v1'
DEFAULT_INCLUDE = r'\.hip$'
WALK_SKIP = {'.git', '__pycache__', '.cache', 'node_modules'}
# `radisa check` options that check-all owns itself or replaces.
NOT_MIRRORED = {'input', 'json', 'compile_commands', 'help'}
RENAMED = {'jobs': '--check-jobs'}
PREFIX = 'check__'


# ------------------------------------------------------------------ the real `check` parser

def check_parser() -> argparse.ArgumentParser:
    """The `radisa check` sub-parser exactly as `radisa check` builds it."""
    p = argparse.ArgumentParser(prog='radisa')
    sub = p.add_subparsers(dest='command')
    check.add_parser(sub)
    return sub.choices['check']


def _mirror(dst: argparse.ArgumentParser) -> None:
    """Copy every optional argument of the check parser onto `dst` (dest prefixed, default suppressed)."""
    g = dst.add_argument_group('options passed to every `radisa check`')
    taken = set(dst._option_string_actions)
    for act in check_parser()._actions:
        if not act.option_strings or act.dest in NOT_MIRRORED or isinstance(act, argparse._HelpAction):
            continue
        new = copy.copy(act)
        new.dest = PREFIX + act.dest
        new.default = argparse.SUPPRESS
        if new.metavar is None and new.choices is None and new.nargs != 0:
            new.metavar = act.dest.upper()
        strings = [RENAMED[act.dest]] if act.dest in RENAMED else [s for s in act.option_strings if s not in taken]
        if not strings:
            strings = ['--check-' + act.dest.replace('_', '-')]
        new.option_strings = strings
        g._add_action(new)


def file_namespace(path: Path, args, compile_commands: str | None = None,
                   index: int = 0) -> argparse.Namespace:
    """The Namespace `radisa check PATH <mirrored options>` would produce."""
    ns = check_parser().parse_args([str(path)])
    for k, v in vars(args).items():
        if k.startswith(PREFIX):
            setattr(ns, k[len(PREFIX):], v)
    ns.json = None
    ns.compile_commands = compile_commands
    ns.strict = bool(getattr(ns, 'strict', False)) or args.fail_on in ('warning', 'incomplete')
    if getattr(ns, 'jobs', None) is None and hasattr(ns, 'jobs'):
        ns.jobs = max(1, (os.cpu_count() or 1) // max(1, args.jobs))
    unit = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', path.name)}-{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]}"
    if getattr(ns, 'keep', None):
        ns.keep = str(Path(ns.keep) / unit)
    if getattr(ns, 'cache_dir', None):  # one cache per unit, as without --cache-dir
        ns.cache_dir = str(Path(ns.cache_dir) / unit)
    for k in ('fn', 'command'):
        if hasattr(ns, k):
            delattr(ns, k)
    return ns


# ------------------------------------------------------------------ input selection

def _git(cwd: Path, *argv: str) -> bytes:
    p = subprocess.run(['git', '-C', str(cwd), *argv], capture_output=True, check=False)
    if p.returncode:
        err = p.stderr.decode('utf-8', 'replace').strip()
        raise ValueError(f"git {' '.join(argv)} failed in {cwd}: {err}")
    return p.stdout


def _z(out: bytes) -> list[str]:
    """NUL-separated `git ... -z` output as file-system names (no C quoting, any byte allowed)."""
    return [os.fsdecode(x) for x in out.split(b'\0') if x]


def changed_files(paths: list[Path], ref: str) -> set[Path]:
    """Files changed relative to `ref` (git diff --name-only, working tree) plus untracked files,
    for every git repository the `paths` (the selected files) live in; resolved absolute paths.
    `-z`: names with non-ASCII or special characters come back verbatim (core.quotePath would
    C-quote them otherwise, git-config(1))."""
    dirs = {p if p.is_dir() else p.parent for p in paths}
    tops = {Path(os.fsdecode(_git(d, 'rev-parse', '--show-toplevel')).strip()).resolve() for d in dirs}
    out = set()
    for top in tops:
        names = _z(_git(top, 'diff', '--name-only', '-z', ref, '--'))
        names += _z(_git(top, 'ls-files', '-z', '--others', '--exclude-standard'))
        out |= {(top / n).resolve() for n in names}
    return out


def compile_db_files(db: Path) -> dict[Path, str]:
    """Every source file of compile_commands.json, resolved."""
    entries = json.loads(db.read_text())
    out = {}
    for e in entries:
        f = Path(e['file'])
        if not f.is_absolute():
            f = Path(e.get('directory', '.')) / f
        out[f.resolve()] = str(db)
    return out


def select(args) -> tuple[list[Path], dict[Path, str]]:
    """(files to check in order, file -> compile_commands to use)."""
    inc = re.compile(args.include)
    exc = re.compile(args.exclude) if args.exclude else None
    files: dict[Path, None] = {}
    ccmap: dict[Path, str] = {}
    if args.compile_commands:
        db = Path(args.compile_commands)
        ccmap = compile_db_files(db)
        for f in sorted(ccmap):
            if inc.search(str(f)):
                files[f] = None
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            raise ValueError(f'input does not exist: {raw}')
        if p.is_file():
            files[p.resolve()] = None
            continue
        for d, dirs, names in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in WALK_SKIP)
            for n in sorted(names):
                f = (Path(d) / n).resolve()
                if inc.search(str(f)):
                    files[f] = None
    sel = [f for f in files if not (exc and exc.search(str(f)))]
    if args.changed_since:
        # The git roots come from the selected files only: a compile db or a directory argument
        # may live outside the repository (an out-of-tree build directory).
        changed = changed_files(sel, args.changed_since) if sel else set()
        sel = [f for f in sel if f in changed]
    return sel, {f: ccmap[f] for f in sel if f in ccmap and f.suffix in hipsrc.SOURCE_SUFFIXES}


# ------------------------------------------------------------------ per-file work

def synthetic_report(path: str, rule: str, message: str, strict: bool, **extra) -> dict:
    """A finalized report for a file that was not (fully) analysed: always INCOMPLETE."""
    f = {'rule': rule, 'severity': 'incomplete', 'pc': None, 'message': message,
         'source': 'radisa check-all', **extra}
    report = {'schema': 'radisa-check-v1', 'input': path, 'status': None, 'exit_code': None,
              'isa_table': None, 'decoder': None, 'counts': {}, 'operand_model': {'known': 0, 'unknown': 0},
              'findings': [f], 'kernels': [], 'incremental': None, 'not_checked': list(check.NOT_CHECKED)}
    return check.finalize(report, strict)


def kernel_names(path: Path, ns) -> list[tuple[str, str]] | None:
    """(name, symbol) of every kernel in the unit before any code generation; None if it cannot
    be told cheaply. .s: `.amdhsa_kernel` directives (a snippet without one is one kernel);
    objects: the code-object metadata; HIP: kernels of the optimised IR (one `clang -emit-llvm`
    run, only done with --max-kernels-per-file)."""
    from .binary import ELF, kernel_inventory
    from .incremental import parse_module
    if path.suffix in ('.s', '.S', '.asm'):
        text = path.read_text(encoding='utf-8', errors='replace')
        names = re.findall(r'^\s*\.amdhsa_kernel\s+(\S+)', text, re.M)
        return [(n, n + '.kd') for n in names] or [('<snippet>', '<snippet>')]
    if path.suffix in hipsrc.SOURCE_SUFFIXES:
        flags = [x for f in (ns.flag or []) for x in f.split()]
        with tempfile.TemporaryDirectory(prefix='radisa-count-') as tmp:
            r = hipsrc.compile_ir(path, Path(tmp), ns.cxx, flags,
                                  Path(ns.compile_commands) if ns.compile_commands else None,
                                  timeout=ns.timeout)
        return [(n, n) for n in parse_module(r['ir'])['kernels']] if r.get('ir') else None
    return [(k['name'], k['symbol']) for d in check._objects(path) for k in kernel_inventory(ELF(d))[0]]


def count_kernels(path: Path, ns) -> int | None:
    """Kernels the check of `ns` analyses: with `--kernel S` only those whose name or symbol
    contains S, as check.check_object_bytes filters them."""
    names = kernel_names(path, ns)
    if names is None:
        return None
    sub = getattr(ns, 'kernel', None)
    return len([1 for n, s in names if not sub or sub in n or sub in s])


def check_one(path: Path, ns, max_kernels: int | None) -> dict:
    """check.analyze(ns), or the INCOMPLETE skip entry for a unit above `max_kernels`."""
    if max_kernels:
        try:
            n = count_kernels(path, ns)
        except Exception:  # the real analysis reports why
            n = None
        if n is not None and n > max_kernels:
            sel = f' matching --kernel {ns.kernel}' if getattr(ns, 'kernel', None) else ''
            return synthetic_report(str(path), 'SKIPPED_TOO_MANY_KERNELS',
                                    f'not checked: {n} kernels{sel} exceed --max-kernels-per-file {max_kernels}; '
                                    f'check this unit on its own (radisa check {path.name}, optionally --kernel NAME)',
                                    ns.strict, kernels=n)
    return check.analyze(ns)


def _child(conn, path: Path, ns, max_kernels: int | None) -> None:
    try:
        os.setsid()   # its own process group: _kill stops the compilers it starts too
    except (OSError, AttributeError):  # AttributeError: no os.setsid (Windows)
        pass
    try:  # the parent's SIGTERM handler (run) is inherited by fork; the worker just dies
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except (OSError, ValueError, AttributeError):
        pass
    try:
        report = check_one(path, ns, max_kernels)
    except BaseException as ex:  # noqa: BLE001 - a crash is a result, never the end of the run
        report = synthetic_report(str(path), 'CHECK_CRASHED', f'{type(ex).__name__}: {ex}', ns.strict,
                                  traceback=traceback.format_exc())
    try:
        conn.send(report)
    except Exception:
        conn.send(json.loads(json.dumps(report, default=formats._plain)))
    conn.close()


def _kill(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        proc.kill()
    proc.join(5)


def run_files(jobs: list[tuple[Path, argparse.Namespace]], n: int, timeout: float | None,
              max_kernels: int | None, progress=None) -> list[dict]:
    """Check every (path, namespace) in at most `n` processes at a time; reports in input order,
    each with `elapsed_s`."""
    ctx = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn')
    todo = list(enumerate(jobs))
    running: dict = {}
    reports: list[dict | None] = [None] * len(jobs)
    done = 0

    def finish(i: int, report: dict, t0: float) -> None:
        nonlocal done
        report['elapsed_s'] = round(time.monotonic() - t0, 3)
        reports[i] = report
        done += 1
        if progress:
            progress(done, len(jobs), report)

    try:
        _loop(ctx, todo, running, n, timeout, max_kernels, finish)
    finally:  # normal end: nothing left; KeyboardInterrupt / SIGTERM: stop every running check
        for r, (proc, *_) in list(running.items()):
            _kill(proc)
            r.close()
        running.clear()
    return reports  # type: ignore[return-value]


def _loop(ctx, todo: list, running: dict, n: int, timeout: float | None, max_kernels: int | None, finish) -> None:
    while todo or running:
        while todo and len(running) < max(1, n):
            i, (path, ns) = todo.pop(0)
            r, w = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_child, args=(w, path, ns, max_kernels), daemon=False)
            proc.start()
            running[r] = (proc, i, time.monotonic(), path, ns)
            w.close()
        now = time.monotonic()
        wait_for = 1.0
        if timeout:
            wait_for = max(0.0, min([t0 + timeout - now for _, _, t0, _, _ in running.values()] + [1.0]))
        for r in mp_wait(list(running), wait_for):
            proc, i, t0, path, ns = running.pop(r)
            try:
                report = r.recv()
            except (EOFError, OSError):
                proc.join(5)
                report = synthetic_report(str(path), 'CHECK_CRASHED',
                                          f'the checker process ended without a result (exit code {proc.exitcode})',
                                          ns.strict)
            r.close()
            proc.join(5)
            finish(i, report, t0)
        if timeout:
            now = time.monotonic()
            for r in [r for r, v in running.items() if now - v[2] >= timeout]:
                proc, i, t0, path, ns = running.pop(r)
                _kill(proc)
                r.close()
                finish(i, synthetic_report(str(path), 'CHECK_TIMEOUT',
                                           f'not finished after --timeout-per-file {timeout:g} s; the file was not '
                                           'checked (its compiler and checker processes were stopped)', ns.strict), t0)


# ------------------------------------------------------------------ validation and baselines

def validate(jobs: list[tuple[Path, argparse.Namespace]]) -> None:
    """What `check.analyze` validates first (configuration, --severity, a missing --baseline),
    for every file, in the parent: ValueError -> exit 2 as in `radisa check`, not one
    CHECK_CRASHED per file. Runs on copies (suppress.settings caches on the Namespace)."""
    for _, ns in jobs:
        if hasattr(ns, 'enable') or hasattr(ns, 'config'):
            suppress.validate(copy.copy(ns))


def _read_baseline_doc(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as ex:
        raise ValueError(f'baseline {path}: {ex}') from ex
    if not isinstance(doc, dict) or doc.get('schema') != suppress.SCHEMA or not isinstance(doc.get('entries'), list):
        raise ValueError(f'baseline {path}: not a {suppress.SCHEMA} file')
    return doc


def _rel(f: Path, bdir: Path) -> str:
    try:
        return Path(os.path.relpath(f, bdir.resolve())).as_posix()
    except ValueError:  # another drive (Windows)
        return Path(f).as_posix()


def baseline_entries(path: Path, files: list[Path]) -> dict[Path | None, list[dict]]:
    """The entries of baseline `path` per checked file (resolved); None: entries of no file in
    `files`. Per-entry `input` is relative to the baseline's directory (check-all baselines);
    a `radisa check` baseline names one file in its top-level `input` (as given on that command
    line: tried against the current and the baseline's directory); with one file checked it
    applies to that file, as `radisa check FILE --baseline B` does."""
    doc = _read_baseline_doc(path)
    bdir = path.resolve().parent
    known = set(files)
    out: dict[Path | None, list[dict]] = {}
    for e in doc['entries']:
        if not isinstance(e, dict) or 'fingerprint' not in e:
            continue
        if e.get('input'):
            cands = [bdir / e['input']]
        elif doc.get('input'):
            cands = [Path.cwd() / doc['input'], bdir / doc['input']]
        else:
            cands = []
        hit = next((c.resolve() for c in cands if c.resolve() in known), None)
        if hit is None and not e.get('input') and len(files) == 1:
            hit = files[0]
        out.setdefault(hit, []).append(e)
    return out


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def prepare_baselines(jobs: list[tuple[Path, argparse.Namespace]], tmp: Path) -> dict:
    """Point every per-file Namespace at a baseline of its own under `tmp` (see the module
    docstring). Returns {'write': B or None, 'files': {i: per-file temp to write},
    'retarget': {temp path: user path}} for finish_baselines and the reports."""
    plan: dict = {'write': None, 'files': {}, 'retarget': {}}
    files = [f for f, _ in jobs]
    loaded: dict[Path, dict] = {}
    for i, (f, ns) in enumerate(jobs):
        if getattr(ns, 'write_baseline', None):
            plan['write'] = Path(ns.write_baseline)
            t = tmp / f'write-{i}.json'
            plan['files'][i] = t
            plan['retarget'][str(t)] = str(ns.write_baseline)
            ns.write_baseline = str(t)
            continue
        b = suppress.settings(copy.copy(ns))['baseline'] if hasattr(ns, 'baseline') else None
        if b is None:
            continue
        key = b.resolve()
        if key not in loaded:
            loaded[key] = baseline_entries(b, files)
        t = tmp / f'baseline-{i}.json'
        # Per-entry 'input' is relative to B's directory, not tmp's: drop it, the top-level
        # 'input' names the file these entries belong to (suppress.read_baseline).
        _write_json(t, {'schema': suppress.SCHEMA, 'radisa': __version__, 'input': str(f),
                        'entries': [{k: v for k, v in e.items() if k != 'input'}
                                    for e in loaded[key].get(f, [])]})
        plan['retarget'][str(t)] = str(b)
        ns.baseline = str(t)
    return plan


def finish_baselines(plan: dict, jobs: list[tuple[Path, argparse.Namespace]]) -> int:
    """Merge the per-file baselines written by the workers into the one `--write-baseline` file;
    the entries of files without a new per-file baseline (not checked, crashed) are kept from the
    existing file. Returns the number of entries written."""
    target: Path | None = plan['write']
    if target is None:
        return 0
    bdir = target.resolve().parent
    rows, written = [], set()
    for i, t in plan['files'].items():
        if not t.is_file():
            continue
        f = jobs[i][0]
        written.add(f)
        for e in _read_baseline_doc(t)['entries']:
            rows.append({**e, 'input': _rel(f, bdir)})
    if target.is_file():
        try:
            old = baseline_entries(target, [f for f, _ in jobs])
        except ValueError:
            old = {}
        for f, es in old.items():
            if f is None or f not in written:
                rows += [e if e.get('input') or f is None else {**e, 'input': _rel(f, bdir)} for e in es]
    rows.sort(key=lambda r: (str(r.get('input', '')), r.get('rule', ''), r.get('kernel', ''), r['fingerprint']))
    _write_json(target, {'schema': suppress.SCHEMA, 'radisa': __version__, 'input': None,
                         'inputs': sorted({r['input'] for r in rows if r.get('input')}), 'entries': rows})
    return len(rows)


def retarget(value, paths: dict[str, str]):
    """`value` with every per-file temporary baseline path replaced by the user's path."""
    if isinstance(value, str):
        for a, b in paths.items():
            if a in value:
                value = value.replace(a, b)
        return value
    if isinstance(value, dict):
        return {k: retarget(v, paths) for k, v in value.items()}
    if isinstance(value, list):
        return [retarget(v, paths) for v in value]
    return value


# ------------------------------------------------------------------ summary and exit status

def exit_code(reports: list[dict], fail_on: str = 'error') -> int:
    st = [r['status'] for r in reports]
    if 'FAIL' in st:
        return 1
    if 'INCOMPLETE' in st:
        return 1 if fail_on == 'incomplete' else 3
    return 0


def summary(reports: list[dict], fail_on: str = 'error', slowest: int = 5) -> dict:
    from collections import Counter
    sev = Counter(f['severity'] for r in reports for _, f in formats.iter_findings(r))
    st = Counter(r['status'] for r in reports)
    slow = sorted(reports, key=lambda r: -r.get('elapsed_s', 0))[:slowest]
    return {'files': len(reports), 'kernels': sum(1 for r in reports for k in r['kernels'] if not k.get('callee')),
            'functions': sum(1 for r in reports for k in r['kernels'] if k.get('callee')),
            'status': {k: st.get(k, 0) for k in ('PASS', 'FAIL', 'INCOMPLETE', 'NO_KERNELS')},
            'errors': sev.get('error', 0), 'warnings': sev.get('review', 0),
            'notes': sev.get('incomplete', 0) + sev.get('info', 0),
            'slowest': [{'input': r['input'], 'seconds': r.get('elapsed_s', 0), 'status': r['status']} for r in slow],
            'fail_on': fail_on, 'exit_code': exit_code(reports, fail_on)}


def render_summary(s: dict) -> str:
    st = s['status']
    lines = ['radisa check-all summary',
             f"  files       {s['files']}",
             f"  kernels     {s['kernels']}" + (f" (+ {s['functions']} function(s))" if s.get('functions') else ''),
             f"  PASS {st['PASS']}  FAIL {st['FAIL']}  INCOMPLETE {st['INCOMPLETE']}  NO_KERNELS {st['NO_KERNELS']}",
             f"  findings    {s['errors']} error(s), {s['warnings']} warning(s), {s['notes']} note(s)"]
    if s['slowest']:
        lines.append('  slowest files')
        for x in s['slowest']:
            lines.append(f"    {x['seconds']:9.2f} s  {x['status']:<10} {x['input']}")
    lines.append(f"  -> exit {s['exit_code']} (--fail-on {s['fail_on']})")
    return '\n'.join(lines)


# ------------------------------------------------------------------ CLI

def add_parser(sub) -> None:
    c = sub.add_parser('check-all', help='lint many HIP sources / assembly files / code objects in one run',
                       description=__doc__.split('\n\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument('paths', nargs='*', help='files (always checked) and directories (files matching --include)')
    c.add_argument('--compile-commands', help='compile_commands.json: check its entries matching --include; '
                                              'HIP files listed there are compiled with their flags')
    c.add_argument('--include', default=DEFAULT_INCLUDE,
                   help=f'regex for files taken from --compile-commands and directories (default {DEFAULT_INCLUDE})')
    c.add_argument('--exclude', help='regex; matching files are dropped')
    c.add_argument('--changed-since', metavar='GITREF',
                   help='only files changed relative to GITREF (git diff --name-only) or untracked')
    c.add_argument('--jobs', type=int, default=1, help='files checked in parallel (default 1)')
    c.add_argument('--timeout-per-file', type=float, help='seconds; a slower file becomes INCOMPLETE')
    c.add_argument('--max-kernels-per-file', type=int,
                   help='skip (INCOMPLETE, named in the report) units with more kernels than this')
    c.add_argument('--format', choices=formats.FORMATS, default='text')
    c.add_argument('--out', help='write the --format output here instead of stdout')
    c.add_argument('--fail-on', choices=('error', 'warning', 'incomplete'), default='error',
                   help='error: exit 1 on errors, 3 on INCOMPLETE; warning: warnings fail too; '
                        'incomplete: warnings and INCOMPLETE files exit 1')
    c.add_argument('--quiet', action='store_true', help='no per-file progress on stderr')
    _mirror(c)
    c.set_defaults(fn=run)


def run(args) -> int:
    if args.jobs < 1:
        raise ValueError('--jobs must be at least 1')
    if not args.paths and not args.compile_commands:
        raise ValueError('check-all needs paths and/or --compile-commands')
    files, ccmap = select(args)
    jobs = [(f, file_namespace(f, args, ccmap.get(f), i)) for i, f in enumerate(files)]
    validate(jobs)

    def progress(i, n, r):
        if not args.quiet:
            print(f"[{i}/{n}] {r['status']:<10} {r.get('elapsed_s', 0):7.2f} s  {r['input']}", file=sys.stderr)

    got: list[int] = []

    def on_signal(signum, frame):
        got.append(signum)
        raise KeyboardInterrupt

    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM,):
            try:
                handlers[sig] = signal.signal(sig, on_signal)
            except (OSError, ValueError):
                pass
    tmp = Path(tempfile.mkdtemp(prefix='radisa-check-all-'))
    try:
        plan = prepare_baselines(jobs, tmp)
        reports = run_files(jobs, args.jobs, args.timeout_per_file, args.max_kernels_per_file, progress)
        finish_baselines(plan, jobs)
        reports = [retarget(r, plan['retarget']) for r in reports]
    except KeyboardInterrupt:
        signum = got[0] if got else signal.SIGINT
        print(f'radisa check-all: interrupted by signal {int(signum)}; every running check was stopped',
              file=sys.stderr)
        return 128 + int(signum)
    finally:
        for sig, h in handlers.items():
            signal.signal(sig, h)
        shutil.rmtree(tmp, ignore_errors=True)
    s = summary(reports, args.fail_on)
    strict = args.fail_on != 'error'
    if args.format == 'json':
        body = json.dumps({'schema': SCHEMA, 'summary': s, 'files': reports}, indent=2, sort_keys=True,
                          default=formats._plain) + '\n'
    elif args.format == 'text':
        body = emit(reports, 'text', strict) if reports else ''
    else:
        body = emit(reports, args.format, strict)
    text_summary = render_summary(s) if reports else 'radisa check-all: no files selected -> exit 0'
    if args.out:
        Path(args.out).write_text(body)
    elif body:
        sys.stdout.write(body)
    # The summary goes to stdout only for the text format; machine formats keep stdout parseable.
    print(text_summary, file=sys.stdout if args.format == 'text' and not args.out else sys.stderr)
    return s['exit_code']


def main(argv=None) -> int:
    """Stand-alone entry (`python3 -m radisa.batch ...`) until `radisa check-all` is wired into cli."""
    p = argparse.ArgumentParser(prog='radisa')
    sub = p.add_subparsers(dest='command', required=True)
    add_parser(sub)
    args = p.parse_args(['check-all', *(sys.argv[1:] if argv is None else argv)])
    try:
        return args.fn(args)
    except (OSError, ValueError, KeyError, TypeError) as ex:
        print(f'INCOMPLETE: {type(ex).__name__}: {ex}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
