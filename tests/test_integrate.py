"""The wiring of the rule modules into `radisa check` and the CLI (radisa/check.py, radisa/cli.py).

Each module has its own tests; these check only that the pieces are connected: registered passes
and filters, the CLI flags that reach them, callee linting in the object path, the inline-asm
clobber check in both HIP paths, and the subcommands.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from support import CHECK_FIX, all_findings, check_args, needs_llvm_mc, needs_minimal_kernel, run_check

from radisa import check, cli, liveness, memorder, suppress

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which('clang++')


def rules_of(report, severities=('error', 'review')):
    return sorted(f['rule'] for f in all_findings(report) if f['severity'] in severities)


class Registration(unittest.TestCase):
    def test_passes_and_filters(self):
        self.assertIn(memorder.kernel_pass, check.KERNEL_PASSES)
        self.assertIn(liveness.kernel_pass, check.KERNEL_PASSES)
        self.assertEqual(check.REPORT_FILTERS[-1], suppress.report_filter)

    def test_subcommands(self):
        for argv in (['rules'], ['check-all', '--help'], ['drift', '--help'], ['check', '--help']):
            with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()) as out:
                try:
                    code = cli.main(argv)
                except SystemExit as ex:   # --help
                    code = ex.code
            self.assertEqual(code, 0, out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                cli.main(['check', '--help'])
        for flag in ('--format', '--enable', '--disable', '--severity', '--config', '--no-config', '--baseline',
                     '--write-baseline', '--liveness-entry', '--no-liveness'):
            self.assertIn(flag, out.getvalue())

    def test_pass_options(self):
        a = check_args('x.s', '--no-config', '--enable', 'sys_store_wait', '--disable', 'CLAUSE_MIXED',
                       '--liveness-entry', 's[0:1], v0', '--liveness-entry', 'v1', '--no-liveness')
        o = check.pass_options(a)
        self.assertEqual((sorted(o['enable']), sorted(o['disable'])), (['SYS_STORE_WAIT'], ['CLAUSE_MIXED']))
        self.assertEqual((o['liveness_entry'], o['liveness']), (['s[0:1]', 'v0', 'v1'], False))
        # The cache identity is the same in every process (RuleSet prints sorted).
        self.assertEqual(check.analysis_identity(o), check.analysis_identity(check.pass_options(a)))
        # Without any selection the options are what they were before the integration.
        self.assertEqual(check.pass_options(check_args('x.s', '--no-config')), {})
        b = check_args('x.s', '--no-config', '--enable', 'CLAUSE_MIXED')
        b.pass_options = {'enable': 'SYS_STORE_WAIT'}
        self.assertEqual(sorted(check.pass_options(b)['enable']), ['CLAUSE_MIXED', 'SYS_STORE_WAIT'])


@needs_llvm_mc
class CheckFlags(unittest.TestCase):
    def test_enable_reaches_opt_in_pass(self):
        self.assertEqual(run_check(CHECK_FIX / 'memorder_sys_store.s', '--no-config')[0], 0)
        code, r = run_check(CHECK_FIX / 'memorder_sys_store.s', '--no-config', '--enable', 'SYS_STORE_WAIT')
        self.assertEqual((code, rules_of(r)), (0, ['SYS_STORE_WAIT']))
        code, r = run_check(CHECK_FIX / 'memorder_sys_store_clean.s', '--no-config', '--enable', 'SYS_STORE_WAIT')
        self.assertEqual((code, rules_of(r)), (0, []))

    def test_config_file_enables_and_regrades(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'store.s')
            shutil.copy(CHECK_FIX / 'memorder_sys_store.s', src)
            Path(d, '.radisa.json').write_text(json.dumps({'enable': ['SYS_STORE_WAIT'],
                                                           'severity': {'SYS_STORE_WAIT': 'error'}}))
            code, r = run_check(src)
        self.assertEqual((code, [(f['rule'], f['severity']) for f in all_findings(r)]),
                         (1, [('SYS_STORE_WAIT', 'error')]))

    def test_liveness_entry_for_snippets(self):
        snippet = CHECK_FIX / 'liveness_snippet.s'
        self.assertNotIn('UNINITIALIZED_READ', rules_of(run_check(snippet, '--no-config')[1]))
        code, r = run_check(snippet, '--no-config', '--liveness-entry', 'v0')
        self.assertIn('UNINITIALIZED_READ', rules_of(r))

    @needs_minimal_kernel
    def test_no_liveness(self):
        code, r = run_check(CHECK_FIX / 'liveness_uninit.s', '--no-config')
        self.assertIn('UNINITIALIZED_READ', rules_of(r))
        code, r = run_check(CHECK_FIX / 'liveness_uninit.s', '--no-config', '--no-liveness')
        self.assertNotIn('UNINITIALIZED_READ', rules_of(r))

    def test_hazard_sources_keep_llvm_citation(self):
        _, r = run_check(CHECK_FIX / 'hazard_dealloc.s', '--no-config')
        [f] = [f for f in all_findings(r) if f['rule'] == 'DEALLOC_NO_NOP']
        self.assertIn('SIInsertWaitcnts', f['source'])

    def test_formats(self):
        for fmt, probe in (('json', lambda s: json.loads(s)['status'] == 'FAIL'),
                           ('sarif', lambda s: json.loads(s)['version'] == '2.1.0'),
                           ('github', lambda s: s.startswith('::error ')),
                           ('text', lambda s: '-> FAIL' in s)):
            with self.subTest(fmt=fmt), contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli.main(['check', str(CHECK_FIX / 'missing_waits.s'), '--no-config', '--format', fmt])
            self.assertEqual(code, 1)
            self.assertTrue(probe(out.getvalue()), out.getvalue()[:300])

    def test_hook_honours_config(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'w.s')
            shutil.copy(CHECK_FIX / 'missing_waits.s', src)
            event = json.dumps({'tool_input': {'file_path': str(src)}})
            env = {**os.environ, 'PYTHONPATH': str(ROOT)}

            def hook():
                return subprocess.run([sys.executable, '-m', 'radisa', 'hook'], input=event, text=True, env=env,
                                      capture_output=True, cwd=d).returncode
            self.assertEqual(hook(), 2)
            Path(d, '.radisa.json').write_text(json.dumps({'severity': {'MISSING_WAIT': 'warning'}}))
            self.assertEqual(hook(), 0)


@needs_minimal_kernel
class Callees(unittest.TestCase):
    def test_functions_are_linted(self):
        code, r = run_check(CHECK_FIX / 'callee_missing_waits.s', '--no-config')
        self.assertEqual(code, 1)
        self.assertEqual([(k['name'], bool(k.get('callee'))) for k in r['kernels']], [('k', False), ('helper', True)])
        helper = r['kernels'][1]
        self.assertEqual(sorted(f['rule'] for f in helper['findings']), ['CALLEE_RETURN_OUTSTANDING', 'MISSING_WAIT'])
        [calls] = [f for f in r['kernels'][0]['findings'] if f['rule'] == 'CALLS']
        self.assertIn('(helper) are checked on their own', calls['message'])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cli.main(['check', str(CHECK_FIX / 'callee_missing_waits.s'), '--no-config'])
        self.assertIn('1 kernel(s) + 1 function(s)', out.getvalue())

    def test_kernel_filter(self):
        code, r = run_check(CHECK_FIX / 'callee_missing_waits.s', '--no-config', '--kernel', 'k')
        self.assertEqual([k['name'] for k in r['kernels']], ['k'])
        [calls] = [f for f in all_findings(r) if f['rule'] == 'CALLS']
        self.assertIn('not checked in this run', calls['message'])

    def test_compiler_output_is_clean(self):
        code, r = run_check(CHECK_FIX / 'callee_compiled.s', '--no-config')
        self.assertEqual((code, rules_of(r)), (0, []))
        self.assertEqual(sum(1 for k in r['kernels'] if k.get('callee')), 1)


@unittest.skipUnless(CXX or os.environ.get('RAD_ISA_REQUIRE_LLVM') == '1', 'no clang++')
class HipPaths(unittest.TestCase):
    """Incremental (fresh and cached) and --no-incremental report the same kernel findings."""

    def key(self, r):
        return sorted((k['name'], f['rule'], f['severity'], (f.get('loc') or {}).get('line'), f['message'])
                      for k in r['kernels'] for f in k['findings'])

    def test_paths_agree(self):
        for name in ('asm_clobber.hip', 'inline_asm_lds.hip'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                argv = ('--no-config', '--cxx', CXX, '--flag', '-nogpuinc -nogpulib', '--cache-dir', d)
                _, fresh = run_check(CHECK_FIX / name, *argv)
                _, cached = run_check(CHECK_FIX / name, *argv)
                _, whole = run_check(CHECK_FIX / name, *argv, '--no-incremental')
                self.assertEqual(cached['incremental']['compiled'], 0)
                self.assertEqual(self.key(fresh), self.key(cached))
                self.assertEqual(self.key(fresh), self.key(whole))
                if name == 'asm_clobber.hip':
                    self.assertEqual(sum(f['rule'] == 'ASM_UNDECLARED_CLOBBER' for f in all_findings(whole)), 6)


if __name__ == '__main__':
    unittest.main()
