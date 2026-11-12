"""`radisa check-all` (radisa/batch.py): input selection, the per-file Namespace from the real
`check` parser, process isolation (crash, timeout), the large-unit skip, summary and exit codes."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

from support import CHECK_FIX, ROOT, needs_llvm_mc
from radisa import batch, check, formats


def parse(*argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog='radisa')
    sub = p.add_subparsers(dest='command', required=True)
    batch.add_parser(sub)
    return p.parse_args(['check-all', *argv])


def run(*argv):
    args = parse(*argv)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = batch.run(args)
    return code, out.getvalue(), err.getvalue()


def fake_analyze(ns):
    """Stands in for check.analyze in the forked workers: status from the file name."""
    name = Path(ns.input).name
    if 'crash' in name:
        raise RuntimeError('boom')
    if 'die' in name:
        os._exit(9)
    if 'slow' in name:
        time.sleep(30)
    sev = 'error' if 'bad' in name else 'review' if 'warn' in name else None
    fs = [{'rule': 'MISSING_WAIT', 'severity': sev, 'pc': 0, 'message': 'x', 'source': 'cite',
           'loc': {'file': ns.input, 'line': 2, 'column': 1}}] if sev else []
    r = {'schema': 'radisa-check-v1', 'input': ns.input, 'status': None, 'exit_code': None, 'isa_table': None,
         'decoder': None, 'counts': {}, 'operand_model': {}, 'incremental': None, 'not_checked': [],
         'findings': [], 'kernels': [{'name': 'k', 'object': ns.input, 'instructions': [1], 'findings': fs,
                                      'operand_model': {'operand_model_known': 1, 'operand_model_unknown': 0}}],
         'ns': {k: v for k, v in vars(ns).items() if isinstance(v, (str, int, float, bool, list, type(None)))}}
    return check.finalize(r, ns.strict)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix='radisa-batch-test-'))
        self.addCleanup(shutil.rmtree, self.d, True)

    def touch(self, *names, text='s_endpgm\n'):
        out = []
        for n in names:
            p = self.d / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            out.append(p)
        return out


class SelectionTest(Tmp):
    def test_compile_commands_include_exclude_and_paths(self):
        a, b, c, s = self.touch('src/a.hip', 'src/b.hip', 'src/c.cpp', 'asm/x.s')
        gen = self.touch('build/gen.hip')[0]
        db = self.d / 'compile_commands.json'
        db.write_text(json.dumps([{'directory': str(self.d / 'src'), 'file': 'a.hip', 'command': 'clang++ -c a.hip'},
                                  {'directory': str(self.d), 'file': str(b), 'command': 'clang++ -c b.hip'},
                                  {'directory': str(self.d), 'file': str(c), 'command': 'clang++ -c c.cpp'},
                                  {'directory': str(self.d), 'file': str(gen), 'command': 'clang++'}]))
        args = parse('--compile-commands', str(db), '--exclude', '/build/', str(s))
        files, cc = batch.select(args)
        self.assertEqual(files, [a.resolve(), b.resolve(), s.resolve()])
        self.assertEqual(cc, {a.resolve(): str(db), b.resolve(): str(db)})  # .s gets no compile db
        # Directories are walked with --include; explicit files are always taken.
        files, _ = batch.select(parse(str(self.d), '--include', r'\.(hip|s)$', '--exclude', 'gen'))
        self.assertEqual(files, sorted([a.resolve(), b.resolve(), s.resolve()]))

    def test_missing_input_is_an_error(self):
        with self.assertRaises(ValueError):
            batch.select(parse(str(self.d / 'nope.s')))

    @unittest.skipUnless(shutil.which('git'), 'git not installed')
    def test_changed_since(self):
        g = ['git', '-C', str(self.d), '-c', 'user.email=t@t', '-c', 'user.name=t']
        subprocess.run(['git', 'init', '-q', str(self.d)], check=True)
        a, b, c = self.touch('a.s', 'b.s', 'sub/c.s')
        subprocess.run([*g, 'add', '.'], check=True)
        subprocess.run([*g, 'commit', '-qm', 'base'], check=True)
        b.write_text('s_nop 0\ns_endpgm\n')           # modified
        new = self.touch('sub/new.s')[0]              # untracked
        files, _ = batch.select(parse(str(self.d), '--include', r'\.s$', '--changed-since', 'HEAD'))
        self.assertEqual(files, [b.resolve(), new.resolve()])
        with self.assertRaises(ValueError):
            batch.select(parse(str(self.d), '--include', r'\.s$', '--changed-since', 'no-such-ref'))


class NamespaceTest(Tmp):
    def test_mirrors_every_check_option(self):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        batch.add_parser(sub)
        mirrored = {a.dest[len(batch.PREFIX):] for a in sub.choices['check-all']._actions
                    if a.dest.startswith(batch.PREFIX)}
        real = {a.dest for a in batch.check_parser()._actions if a.option_strings} - batch.NOT_MIRRORED
        self.assertEqual(mirrored, real)

    def test_values_reach_the_check_namespace(self):
        f = self.touch('a.hip')[0]
        args = parse('--cxx', '/opt/x/clang++', '--flag=-DA', '--flag=-DB', '--kernel', 'gemm',
                     '--check-jobs', '3', '--jobs', '2', '--timeout', '7', '--keep', str(self.d / 'keep'),
                     '--cache-dir', str(self.d / 'cache'), '--no-incremental', str(f))
        ns = batch.file_namespace(f, args, 'cc.json')
        ref = check.add_parser  # noqa: F841 - the real parser is the reference
        self.assertEqual((ns.input, ns.cxx, ns.flag, ns.kernel, ns.jobs, ns.timeout, ns.no_incremental),
                         (str(f), '/opt/x/clang++', ['-DA', '-DB'], 'gemm', 3, 7.0, True))
        self.assertEqual(ns.compile_commands, 'cc.json')
        self.assertIsNone(ns.json)
        self.assertFalse(ns.strict)
        self.assertTrue(ns.keep.startswith(str(self.d / 'keep') + os.sep))   # one subdirectory per unit
        self.assertTrue(ns.cache_dir.startswith(str(self.d / 'cache') + os.sep))
        self.assertNotEqual(batch.file_namespace(self.touch('b.hip')[0], args).cache_dir, ns.cache_dir)
        # defaults are the check parser's own; nested parallelism is divided
        ns = batch.file_namespace(f, parse('--jobs', '1000', str(f)))
        self.assertEqual((ns.timeout, ns.wave, ns.jobs, ns.keep, ns.cache_dir), (600, 32, 1, None, None))
        self.assertTrue(batch.file_namespace(f, parse('--fail-on', 'warning', str(f))).strict)

    def test_new_check_flags_are_picked_up(self):
        orig = check.add_parser

        def extended(sub):
            orig(sub)
            sub.choices['check'].add_argument('--pass-option', action='append', help='new')
        f = self.touch('a.s')[0]
        with mock.patch.object(check, 'add_parser', extended):
            ns = batch.file_namespace(f, parse('--pass-option', 'x=1', str(f)))
            self.assertEqual(ns.pass_option, ['x=1'])
            self.assertIsNone(batch.file_namespace(f, parse(str(f))).pass_option)


class IsolationTest(Tmp):
    """Workers are forked, so patching check.analyze here reaches them."""

    def setUp(self):
        super().setUp()
        p = mock.patch.object(check, 'analyze', fake_analyze)
        p.start()
        self.addCleanup(p.stop)

    def test_crash_timeout_and_exit_codes(self):
        fs = self.touch('ok.s', 'bad.s', 'crash.s', 'die.s', 'slow.s', 'warn.s')
        code, out, err = run(*map(str, fs), '--jobs', '3', '--timeout-per-file', '2', '--format', 'json')
        self.assertEqual(code, 1)
        doc = json.loads(out)
        st = {Path(r['input']).name: r['status'] for r in doc['files']}
        self.assertEqual(st, {'ok.s': 'PASS', 'bad.s': 'FAIL', 'crash.s': 'INCOMPLETE', 'die.s': 'INCOMPLETE',
                              'slow.s': 'INCOMPLETE', 'warn.s': 'PASS'})
        rules = {Path(r['input']).name: [f['rule'] for f in r['findings']] for r in doc['files']}
        self.assertEqual(rules['crash.s'], ['CHECK_CRASHED'])
        self.assertIn('RuntimeError: boom', doc['files'][2]['findings'][0]['message'])
        self.assertEqual(rules['die.s'], ['CHECK_CRASHED'])
        self.assertEqual(rules['slow.s'], ['CHECK_TIMEOUT'])
        self.assertLess(doc['files'][4]['elapsed_s'], 15)
        self.assertEqual([Path(r['input']).name for r in doc['files']], [f.name for f in fs])  # input order
        s = doc['summary']
        self.assertEqual(s['status'], {'PASS': 2, 'FAIL': 1, 'INCOMPLETE': 3, 'NO_KERNELS': 0})
        self.assertEqual((s['files'], s['kernels'], s['errors'], s['warnings']), (6, 3, 1, 1))
        self.assertIn('radisa check-all summary', err)        # machine formats keep stdout clean

    def test_fail_on(self):
        ok, warn, crash = self.touch('ok.s', 'warn.s', 'crash.s')
        self.assertEqual(run(str(ok), str(warn))[0], 0)
        self.assertEqual(run(str(ok), str(warn), '--fail-on', 'warning')[0], 1)
        self.assertEqual(run(str(ok), str(crash))[0], 3)
        self.assertEqual(run(str(ok), str(crash), '--fail-on', 'warning')[0], 3)
        self.assertEqual(run(str(ok), str(crash), '--fail-on', 'incomplete')[0], 1)

    def test_text_summary_and_out_file(self):
        ok, bad = self.touch('ok.s', 'bad.s')
        code, out, _ = run(str(ok), str(bad), '--quiet')
        self.assertEqual(code, 1)
        self.assertIn('[MISSING_WAIT]', out)
        self.assertIn('PASS 1  FAIL 1  INCOMPLETE 0  NO_KERNELS 0', out)
        self.assertIn('slowest files', out)
        sarif = self.d / 'o.sarif'
        code, out, err = run(str(ok), str(bad), '--format', 'sarif', '--out', str(sarif), '--quiet')
        self.assertEqual(out, '')
        log = json.loads(sarif.read_text())
        self.assertEqual(log['version'], '2.1.0')
        self.assertEqual([r['ruleId'] for r in log['runs'][0]['results']], ['MISSING_WAIT'])
        code, out, _ = run(str(ok), str(bad), '--format', 'github', '--quiet')
        self.assertEqual(out.splitlines()[0].split(' ')[0], '::error')

    def test_no_files_selected(self):
        code, out, _ = run(str(self.d), '--include', r'\.nothing$')
        self.assertEqual(code, 0)
        self.assertIn('no files selected', out)

    def test_max_kernels_skip_is_explicit(self):
        two = self.touch('two.s', text='.amdhsa_kernel a\n.end_amdhsa_kernel\n.amdhsa_kernel b\n.end_amdhsa_kernel\n')[0]
        one = self.touch('ok.s')[0]
        code, out, _ = run(str(two), str(one), '--max-kernels-per-file', '1', '--format', 'json')
        self.assertEqual(code, 3)
        files = json.loads(out)['files']
        self.assertEqual(files[0]['status'], 'INCOMPLETE')
        f = files[0]['findings'][0]
        self.assertEqual((f['rule'], f['severity'], f['kernels']), ('SKIPPED_TOO_MANY_KERNELS', 'incomplete', 2))
        self.assertIn('--max-kernels-per-file 1', f['message'])
        self.assertEqual(files[1]['status'], 'PASS')
        self.assertEqual(run(str(two), '--max-kernels-per-file', '2')[0], 0)


class RealCheckTest(unittest.TestCase):
    """End to end with the real check.analyze on checked-in snippets (needs llvm-mc)."""

    @needs_llvm_mc
    def test_fixtures(self):
        clean = [CHECK_FIX / 'clean_waits.s', CHECK_FIX / 'clean_alu_waits.s', CHECK_FIX / 'd16_halves.s']
        code, out, _ = run(*map(str, clean), '--jobs', '2', '--format', 'json')
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)['summary']['status']['PASS'], 3)
        # Same verdict as the single-file `radisa check`.
        code, out, _ = run(str(CHECK_FIX / 'missing_waits.s'), str(CHECK_FIX / 'undecodable.s'), '--format', 'json')
        doc = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual([r['status'] for r in doc['files']], ['FAIL', 'INCOMPLETE'])
        single = check.analyze(batch.check_parser().parse_args([str(CHECK_FIX / 'missing_waits.s')]))
        self.assertEqual([f['rule'] for f in formats.iter_findings(single) for f in [f[1]]],
                         [f['rule'] for _, f in formats.iter_findings(doc['files'][0])])

    @needs_llvm_mc
    def test_check_tree_sh(self):
        env = dict(os.environ, PYTHONPATH='')
        p = subprocess.run(['bash', str(ROOT / 'check-tree.sh'), str(CHECK_FIX / 'clean_waits.s'), '--quiet'],
                           capture_output=True, text=True, env=env, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        p = subprocess.run(['bash', str(ROOT / 'check-tree.sh'), '--each', str(CHECK_FIX / 'undecodable.s'),
                            str(CHECK_FIX / 'missing_waits.s'), str(CHECK_FIX / 'clean_waits.s')],
                           capture_output=True, text=True, env=env, check=False)
        self.assertEqual(p.returncode, 1)
        p = subprocess.run(['bash', str(ROOT / 'check-tree.sh'), '--each', str(CHECK_FIX / 'clean_waits.s'),
                            str(CHECK_FIX / 'undecodable.s')], capture_output=True, text=True, env=env, check=False)
        self.assertEqual(p.returncode, 3)


class HipCountTest(Tmp):
    @unittest.skipUnless(shutil.which('clang++'), 'clang++ not installed')
    def test_hip_kernels_counted_from_ir(self):
        src = self.touch('two.hip', text='#define __global__ __attribute__((global))\n'
                                         '__global__ void a(float *p) { p[0] = 1; }\n'
                                         '__global__ void b(float *p) { p[1] = 2; }\n')[0]
        ns = batch.file_namespace(src, parse('--flag=-nogpuinc -nogpulib', '--cxx', shutil.which('clang++'), str(src)))
        try:
            n = batch.count_kernels(src, ns)
        except Exception as ex:  # a clang without the AMDGPU target
            self.skipTest(f'clang++ cannot compile for gfx1201: {ex}')
        if n is None:
            self.skipTest('clang++ cannot compile for gfx1201')
        self.assertEqual(n, 2)
        r = batch.check_one(src, ns, 1)
        self.assertEqual((r['status'], r['findings'][0]['rule']), ('INCOMPLETE', 'SKIPPED_TOO_MANY_KERNELS'))


class RepoFilesTest(unittest.TestCase):
    def test_hook_and_workflow_files(self):
        hooks = (ROOT / '.pre-commit-hooks.yaml').read_text()
        for needle in ('- id: radisa-check', 'entry: check-tree.sh --each', "files: '\\.(hip|s)$'",
                       'language: script'):
            self.assertIn(needle, hooks)
        wf = (ROOT / '.github/workflows/radisa.yml').read_text()
        uses = {line.split('uses:')[1].strip() for line in wf.splitlines() if 'uses:' in line}
        self.assertEqual(uses, {'actions/checkout@v4', 'actions/upload-artifact@v4'})
        self.assertNotIn('RAD_ISA_REQUIRE_LLVM: ', wf)
        self.assertIn('python3 -m unittest discover -s tests', wf)
        self.assertIn('LLVM 24', wf)
        for f in ('clean_waits.s', 'clean_alu_waits.s', 'd16_halves.s', 'minimal_kernel.s'):
            self.assertTrue((CHECK_FIX / f).exists(), f)
            self.assertIn(f'fixtures/check/{f}', wf)
        self.assertTrue(os.access(ROOT / 'check-tree.sh', os.X_OK))


if __name__ == '__main__':
    unittest.main()
