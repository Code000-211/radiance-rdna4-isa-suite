"""`radisa check-all`: lint many translation units in one run.

Every selected file goes through exactly what `radisa check FILE` does (`check.analyze`), with an
argparse.Namespace produced by the real `check` parser: every option `radisa check` has is mirrored
onto `check-all` automatically (its `--jobs` as `--check-jobs`; `--json` is replaced by `--out`).
Each file runs in its own process (its own process group, so a timeout also stops the compiler it
started). A file that crashes the checker, kills its process or exceeds `--timeout-per-file`
becomes an INCOMPLETE entry for that file; the run itself continues.

Exit status (the same meaning as `radisa check`): 1 if any file FAILs, else 3 if any file is
INCOMPLETE, else 0. `--fail-on warning` makes warnings fail (`--strict` for every file);
`--fail-on incomplete` additionally turns INCOMPLETE files into exit 1.
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
import signal
import subprocess
import sys
import time
import traceback

from . import check, formats, hipsrc
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

def _git(cwd: Path, *argv: str) -> str:
    p = subprocess.run(['git', '-C', str(cwd), *argv], capture_output=True, text=True, check=False)
    if p.returncode:
        raise ValueError(f"git {' '.join(argv)} failed in {cwd}: {p.stderr.strip()}")
    return p.stdout


def changed_files(paths: list[Path], ref: str) -> set[Path]:
    """Files changed relative to `ref` (git diff --name-only, working tree) plus untracked files,
    for every git repository the `paths` live in; resolved absolute paths."""
    roots, out = {}, set()
    for p in paths:
        d = p if p.is_dir() else p.parent
        roots.setdefault(d, None)
    tops = set()
    for d in roots:
        tops.add(Path(_git(d, 'rev-parse', '--show-toplevel').strip()).resolve())
    for top in tops:
        names = _git(top, 'diff', '--name-only', ref, '--').splitlines()
        names += _git(top, 'ls-files', '--others', '--exclude-standard').splitlines()
        out |= {(top / n).resolve() for n in names if n}
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
        anchors = [Path(x).resolve() for x in args.paths] + list(sel)
        if args.compile_commands:
            anchors.append(Path(args.compile_commands).resolve())
        changed = changed_files(anchors, args.changed_since) if anchors else set()
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


def count_kernels(path: Path, ns) -> int | None:
    """Kernels in the unit before any code generation; None if it cannot be told cheaply.
    .s: `.amdhsa_kernel` directives; objects: the code-object metadata; HIP: kernels of the
    optimised IR (one `clang -emit-llvm` run, only done with --max-kernels-per-file)."""
    from .binary import ELF, kernel_inventory
    from .incremental import parse_module
    import tempfile
    if path.suffix in ('.s', '.S', '.asm'):
        text = path.read_text(encoding='utf-8', errors='replace')
        return max(1, len(re.findall(r'^\s*\.amdhsa_kernel\s', text, re.M)))
    if path.suffix in hipsrc.SOURCE_SUFFIXES:
        flags = [x for f in (ns.flag or []) for x in f.split()]
        with tempfile.TemporaryDirectory(prefix='radisa-count-') as tmp:
            r = hipsrc.compile_ir(path, Path(tmp), ns.cxx, flags,
                                  Path(ns.compile_commands) if ns.compile_commands else None,
                                  timeout=ns.timeout)
        return len(parse_module(r['ir'])['kernels']) if r.get('ir') else None
    return sum(len(kernel_inventory(ELF(d))[0]) for d in check._objects(path))


def check_one(path: Path, ns, max_kernels: int | None) -> dict:
    """check.analyze(ns), or the INCOMPLETE skip entry for a unit above `max_kernels`."""
    if max_kernels:
        try:
            n = count_kernels(path, ns)
        except Exception:  # the real analysis reports why
            n = None
        if n is not None and n > max_kernels:
            return synthetic_report(str(path), 'SKIPPED_TOO_MANY_KERNELS',
                                    f'not checked: {n} kernels exceed --max-kernels-per-file {max_kernels}; '
                                    f'check this unit on its own (radisa check {path.name}, optionally --kernel NAME)',
                                    ns.strict, kernels=n)
    return check.analyze(ns)


def _child(conn, path: Path, ns, max_kernels: int | None) -> None:
    try:
        os.setsid()
    except OSError:
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

    while todo or running:
        while todo and len(running) < max(1, n):
            i, (path, ns) = todo.pop(0)
            r, w = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_child, args=(w, path, ns, max_kernels), daemon=False)
            proc.start()
            w.close()
            running[r] = (proc, i, time.monotonic(), path, ns)
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
    return reports  # type: ignore[return-value]


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

    def progress(i, n, r):
        if not args.quiet:
            print(f"[{i}/{n}] {r['status']:<10} {r.get('elapsed_s', 0):7.2f} s  {r['input']}", file=sys.stderr)

    reports = run_files(jobs, args.jobs, args.timeout_per_file, args.max_kernels_per_file, progress)
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
