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

from support import (CHECK_FIX, all_findings, check_args, check_text, kernel_asm, needs_hip, needs_llvm_mc,
                     needs_minimal_kernel, run_check, seq)

from radisa import check, cli, isadb, liveness, memorder, suppress

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
        # The cache identity is the same in every process (RuleSet prints sorted): frozenset order
        # only changes across processes with another hash seed, so compare against such processes.
        code = ('import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; from support import check_args; '
                'from radisa import check; a = check_args("x.s", "--no-config", "--enable", "sys_store_wait", '
                '"--enable", "CLAUSE_MIXED", "--enable", "VOPD_ANTIDEP", "--disable", "PERMLANE_AFTER_CMPX", '
                '"--disable", "WMMA_HAZARD", "--disable", "MISSING_WAIT"); '
                'print(check.analysis_identity(check.pass_options(a)))')
        ids = {subprocess.run([sys.executable, '-c', code, str(ROOT), str(ROOT / 'tests')], capture_output=True,
                              text=True, check=True, env={**os.environ, 'PYTHONHASHSEED': seed}).stdout
               for seed in ('1', '2', '3', '4')}
        self.assertEqual(len(ids), 1, ids)
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
        # (How many MISSING_WAITs the entry rule gives is waits.py's business, tested there.)
        self.assertEqual(sorted({f['rule'] for f in helper['findings']}), ['CALLEE_RETURN_OUTSTANDING', 'MISSING_WAIT'])
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


TABLE = isadb.load()


def fake_kernel(*spec):
    """check_kernel on rules-only instructions (no toolchain)."""
    ins = seq(*spec)
    size = sum(i['size'] for i in ins)
    k = {'name': 'x', 'symbol': 'x', 'metadata': {'.wavefront_size': 32}, 'issues': [], 'descriptor': {},
         'size': size, 'code_hex': ''.join(i['raw'] for i in ins)}
    return check.check_kernel(k, {'instructions': ins, 'issues': [], 'decoded_bytes': size}, TABLE)


class UnknownOperandReads(unittest.TestCase):
    """A MISSING_ALU_WAIT that rests on the guessed read of an unknown instruction is 'review'."""

    def grade(self, *spec):
        return [(f['severity'], f.get('operand_model')) for f in fake_kernel(*spec)['findings']
                if f['rule'] == 'MISSING_ALU_WAIT']

    def test_unknown_consumer(self):
        # VALU reads s10, SALU writes s10, an unknown SALU names s10: only its guessed read makes the hazard.
        self.assertEqual(self.grade(('v_mov_b32', 'v0, s10'), ('s_mov_b32', 's10, 1'), ('s_made_up_b32', 's10, -1'),
                                    ('s_endpgm',)), [('review', 'assumed')])

    def test_unknown_earlier_valu_read(self):
        # The VALU read that makes the SGPR pair tracked is the guess of an unknown VALU instruction.
        self.assertEqual(self.grade(('v_made_up_b32', 'v0, s10'), ('s_mov_b32', 's10, 1'),
                                    ('s_add_co_u32', 's6, s10, 1'), ('s_endpgm',)), [('review', 'assumed')])

    def test_known_sequence_stays_error(self):
        self.assertEqual(self.grade(('v_mov_b32', 'v0, s10'), ('s_made_up_b32', 's20, -1'), ('s_mov_b32', 's10, 1'),
                                    ('s_add_co_u32', 's6, s10, 1'), ('s_endpgm',)), [('error', None)])

    @needs_llvm_mc
    def test_unknown_write_only_instruction(self):
        # s_get_barrier_state is outside the AMD table; llvm-mc assembles it for gfx1201.
        code, r = check_text('v_mov_b32 v0, s10\ns_mov_b32 s10, 1\ns_get_barrier_state s10, -1\ns_endpgm\n',
                             '--no-config')
        self.assertEqual((code, rules_of(r)), (0, ['MISSING_ALU_WAIT', 'OPERAND_MODEL_UNKNOWN']))
        code, r = check_text('v_mov_b32 v0, s10\ns_mov_b32 s10, 1\ns_add_co_u32 s6, s10, 1\ns_endpgm\n', '--no-config')
        self.assertEqual((code, rules_of(r, ('error',))), (1, ['MISSING_ALU_WAIT']))


class Coverage(unittest.TestCase):
    """INCOMPLETE when code was found and not analysed; not for notes about analysed code."""
    CALL = 's_getpc_b64 s[4:5]\ns_add_co_u32 s4, s4, f@rel32@lo+4\ns_add_co_ci_u32 s5, s5, f@rel32@hi+12\n' \
           's_swappc_b64 s[30:31], s[4:5]\ns_endpgm'
    F = '\t.globl f\n\t.p2align 2\n\t.type f,@function\nf:\n\ts_setpc_b64 s[30:31]\n'

    def finalize(self, rule, severity='incomplete'):
        k = {'findings': [{'rule': rule, 'severity': severity, 'pc': 0, 'message': ''}], 'instructions': 3,
             'operand_model': {'operand_model_known': 3, 'operand_model_unknown': 0}}
        return check.finalize({'findings': [], 'kernels': [k]}, False)['status']

    def test_unanalysed_code_is_incomplete(self):
        for rule in ('OBJECT_EVIDENCE', 'CFG_TARGET', 'CFG_ENCODING', 'VOPD_IDENTITY', 'WMMA_OPERANDS'):
            self.assertEqual(self.finalize(rule), 'INCOMPLETE', rule)

    def test_notes_about_analysed_code_pass(self):
        for rule in ('KERNEL_EVIDENCE', 'CFG_FALLTHROUGH', 'DYNAMIC_VGPR_MODE', 'CALLS', 'SCRATCH_USE'):
            self.assertEqual(self.finalize(rule, 'incomplete' if rule != 'CALLS' else 'info'), 'PASS', rule)

    @needs_minimal_kernel
    def test_function_without_size(self):
        code, r = check_text(kernel_asm(self.CALL, extra_functions=self.F), '--no-config')
        self.assertEqual((code, r['status']), (3, 'INCOMPLETE'))
        [calls] = [f for f in all_findings(r) if f['rule'] == 'CALLS']
        self.assertIn('function(s) f of this object are NOT checked', calls['message'])
        self.assertNotIn('holds no function body', calls['message'])
        # The same function with its .size is checked, and the run passes.
        sized = self.F + '.Lf_end:\n\t.size f, .Lf_end-f\n'
        code, r = check_text(kernel_asm(self.CALL, extra_functions=sized), '--no-config')
        self.assertEqual((code, r['status'], [k['name'] for k in r['kernels']]), (0, 'PASS', ['k', 'f']))

    @needs_minimal_kernel
    def test_compiler_output_is_not_incomplete(self):
        code, r = run_check(CHECK_FIX / 'callee_compiled.s', '--no-config')
        self.assertEqual((code, r['status']), (0, 'PASS'))


@unittest.skipUnless(CXX or os.environ.get('RAD_ISA_REQUIRE_LLVM') == '1', 'no clang++')
class CallStack(unittest.TestCase):
    @needs_hip(CXX, CHECK_FIX / 'callee_scratch.hip')
    def test_call_stack_is_not_a_private_array(self):
        _, r = run_check(CHECK_FIX / 'callee_scratch.hip', '--no-config', '--cxx', CXX,
                         '--flag', '-nogpuinc -nogpulib -O1', '--no-incremental')
        [k] = [k for k in r['kernels'] if not k.get('callee')]
        self.assertGreater(k['metadata']['.private_segment_fixed_size'], 0)
        [s] = [f for f in k['findings'] if f['rule'] == 'SCRATCH_USE']
        self.assertEqual(s['severity'], 'info')
        self.assertIn('call stack', s['message'])
        # Compiler output with a size for every function is never incomplete.
        self.assertEqual([f for f in all_findings(r) if f['severity'] == 'incomplete'], [])
        self.assertNotEqual(r['status'], 'INCOMPLETE')

    def test_scratch_without_calls_stays_review(self):
        ins = seq(('v_mov_b32', 'v1, v2'), ('s_endpgm',))
        k = {'name': 'x', 'symbol': 'x', 'issues': [], 'descriptor': {}, 'size': 8,
             'metadata': {'.wavefront_size': 32, '.private_segment_fixed_size': 16}}
        r = check.check_kernel(k, {'instructions': ins, 'issues': [], 'decoded_bytes': 8}, TABLE)
        [s] = [f for f in r['findings'] if f['rule'] == 'SCRATCH_USE']
        self.assertEqual(s['severity'], 'review')


@needs_llvm_mc
class AssemblyLines(unittest.TestCase):
    """Hand-written .s findings carry the line of the .s file, so `; radisa: ignore[...]` applies."""
    PERM = 'v_cmpx_gt_u32_e32 1, v0\n  v_permlane16_b32 v2, v3, s0, s1 ; radisa: ignore[PERMLANE_AFTER_CMPX] reviewed\n' \
           '  s_endpgm\n'

    def test_snippet_layouts(self):
        for name, text, line in (('plain', '  ' + self.PERM, 2), ('label', 'start:\n  ' + self.PERM, 3),
                                 ('label on the line', '.Lx: ' + self.PERM, 2),
                                 ('block comment', '/* header\n   comment */\n  ' + self.PERM, 4),
                                 ('inline block comment', '/* a */ ' + self.PERM, 2)):
            with self.subTest(name):
                code, r = check_text(text, '--no-config')
                self.assertEqual((code, r['status'], rules_of(r)), (0, 'PASS', []))
                self.assertEqual([f['loc']['line'] for f in r['suppressed']], [line])

    def test_unmappable_snippet_says_so(self):
        text = '.macro twice\n  v_nop\n  v_nop\n.endm\n  ' + self.PERM.replace('\n', '\n  twice\n', 1)
        code, r = check_text(text, '--no-config')
        self.assertEqual((code, r['status']), (1, 'FAIL'))   # the error is not hidden, nor incomplete
        [m] = [f for f in all_findings(r) if f['rule'] == 'SOURCE_MAP']
        self.assertEqual(m['severity'], 'info')
        [e] = [f for f in all_findings(r) if f['rule'] == 'PERMLANE_AFTER_CMPX']
        self.assertNotIn('loc', e)

    @needs_minimal_kernel
    def test_kernel_form(self):
        body = ('s_load_b64 s[4:5], s[0:1], 0x0\ns_wait_kmcnt 0\nglobal_load_b32 v1, v0, s[4:5]\n'
                'v_add_f32 v2, v1, v1 ; radisa: ignore[MISSING_WAIT] reviewed: v2 is discarded\ns_endpgm')
        code, r = check_text(kernel_asm(body), '--no-config')
        self.assertEqual((code, r['status'], rules_of(r)), (0, 'PASS', []))
        [f] = r['suppressed']
        self.assertEqual((f['rule'], f['loc']['line'], f['loc']['column']), ('MISSING_WAIT', 11, 1))
        code, r = check_text(kernel_asm(body.replace(' ; radisa: ignore[MISSING_WAIT] reviewed: v2 is discarded', '')),
                             '--no-config')
        [f] = [f for f in all_findings(r) if f['rule'] == 'MISSING_WAIT']
        self.assertEqual((code, f['loc']['line']), (1, 11))

    def test_line_map(self):
        text = '/* x */ start: .Ly: v_nop ; c\n\t.text\n// v_bogus\nloop:\n  s_endpgm // e\n'
        got = check.asm_line_map(text, 'f.s', True)['<snippet>']
        self.assertEqual([(i['mnemonic'], i['loc']['line'], i['loc']['column']) for i in got],
                         [('v_nop', 1, 1), ('s_endpgm', 5, 1)])
        k = '\t.type k,@function\nk:\n  v_nop\nloop:\n  s_endpgm\n\t.type g,@function\ng:\n  s_setpc_b64 s[30:31]\n'
        got = check.asm_line_map(k, 'f.s', False)
        self.assertEqual({n: [i['loc']['line'] for i in v] for n, v in got.items()}, {'k': [3, 5], 'g': [8]})


class LivenessEntry(unittest.TestCase):
    def test_bad_register_is_a_usage_error(self):
        for bad in ('bogus', 's[0;1]', 'v0, x1'):
            with self.subTest(bad), contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as ex:
                check_args('x.s', '--liveness-entry', bad)
            self.assertEqual(ex.exception.code, 2)
            self.assertIn('not a register', err.getvalue())
        a = check_args('x.s', '--no-config', '--liveness-entry', 's[0:1], v0,ttmp2', '--liveness-entry', 'vcc')
        self.assertEqual(check.pass_options(a)['liveness_entry'], ['s[0:1]', 'v0', 'ttmp2', 'vcc'])


@needs_llvm_mc
class Hook(unittest.TestCase):
    def run_hook(self, d, src, **env):
        event = json.dumps({'tool_input': {'file_path': str(src)}})
        return subprocess.run([sys.executable, '-m', 'radisa', 'hook'], input=event, text=True, capture_output=True,
                              cwd=d, env={**os.environ, 'PYTHONPATH': str(ROOT), **env})

    def test_flags_that_look_like_options_do_not_block(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'ok.s')
            src.write_text('v_mov_b32 v1, 0\ns_endpgm\n')
            for flags in ('-O2', '-O2 -g'):
                p = self.run_hook(d, src, RADISA_FLAGS=flags)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertNotIn('usage:', p.stderr)
            shutil.copy(CHECK_FIX / 'missing_waits.s', src)
            self.assertEqual(self.run_hook(d, src, RADISA_FLAGS='-O2').returncode, 2)

    def test_bad_project_configuration_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'ok.s')
            src.write_text('v_mov_b32 v1, 0\ns_endpgm\n')
            self.assertEqual(self.run_hook(d, src).returncode, 0)
            for cfg in ({'severity': {'MISSING_WAIT': 'warn'}}, {'baseline': 'missing.json'}):
                with self.subTest(cfg=cfg):
                    Path(d, '.radisa.json').write_text(json.dumps(cfg))
                    p = self.run_hook(d, src)
                    self.assertEqual(p.returncode, 2)
                    self.assertIn('configuration or usage error', p.stderr)


if __name__ == '__main__':
    unittest.main()
