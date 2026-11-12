"""`radisa check` plumbing: analyze/finalize, extension points, coverage status, test helpers."""
import io
import os
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from radisa import check, isadb, sgpr
from support import (CHECK_FIX, all_findings, check_args, check_text, kernel_asm, needs_llvm_mc,
                     needs_minimal_kernel, run_check, seq, _need)

TABLE = isadb.load()


def fake_kernel(*spec, descriptor=None):
    """check_kernel on rules-only instructions (no toolchain): the result dict of one kernel."""
    ins = seq(*spec)
    size = sum(i['size'] for i in ins)
    k = {'name': 'x', 'symbol': 'x', 'metadata': {'.wavefront_size': 32}, 'issues': [],
         'descriptor': descriptor or {}, 'size': size, 'code_hex': ''.join(i['raw'] for i in ins)}
    return check.check_kernel(k, {'instructions': ins, 'issues': [], 'decoded_bytes': size}, TABLE)


def rules_of(findings):
    return [(f['rule'], f['severity']) for f in findings]


class Registered:
    """Temporarily add entries to a module-level list (KERNEL_PASSES, REPORT_FILTERS)."""
    def __init__(self, target, *fns):
        self.target, self.fns = target, fns

    def __enter__(self):
        self.target.extend(self.fns)

    def __exit__(self, *exc):
        for fn in self.fns:
            self.target.remove(fn)


class FinalizeTests(unittest.TestCase):
    def report(self, *findings, kernel_findings=(), instructions=3):
        k = {'findings': list(kernel_findings), 'instructions': instructions,
             'operand_model': {'operand_model_known': instructions, 'operand_model_unknown': 0}}
        return {'findings': list(findings), 'kernels': [k]}

    def test_status_and_exit_code(self):
        f = lambda rule, sev: {'rule': rule, 'severity': sev, 'pc': None, 'message': ''}
        cases = [((), (), 'PASS', 0),
                 ((f('X', 'error'),), (), 'FAIL', 1),
                 ((), (f('X', 'review'),), 'PASS', 0),
                 ((), (f('DECODE', 'incomplete'),), 'INCOMPLETE', 3),
                 ((), (f('BYTE_COVERAGE', 'incomplete'),), 'INCOMPLETE', 3),
                 ((), (f('BYTE_COVERAGE', 'incomplete'), f('MISSING_WAIT', 'error')), 'FAIL', 1),
                 ((f('SOURCE_MAP', 'incomplete'),), (), 'INCOMPLETE', 3),
                 ((f('NO_KERNELS', 'info'),), (), 'NO_KERNELS', 0)]
        for top, kf, status, code in cases:
            r = check.finalize(self.report(*top, kernel_findings=kf), False)
            self.assertEqual((r['status'], r['exit_code']), (status, code), (top, kf))
        r = check.finalize(self.report(kernel_findings=[f('X', 'review')]), True)
        self.assertEqual((r['status'], r['exit_code'], r['counts']), ('FAIL', 1, {'review': 1}))
        self.assertEqual(check.finalize({'findings': [], 'kernels': []}, False)['status'], 'INCOMPLETE')

    def test_finalize_can_be_rerun(self):
        r = check.finalize(self.report(kernel_findings=[{'rule': 'X', 'severity': 'error'}]), False)
        self.assertEqual(r['status'], 'FAIL')
        r['kernels'][0]['findings'][0]['severity'] = 'review'
        self.assertEqual((check.finalize(r, False)['status'], r['counts']), ('PASS', {'review': 1}))


class UnknownOperandModelTests(unittest.TestCase):
    def test_unknown_instruction_is_reported(self):
        r = fake_kernel(('v_made_up_b32', 'v1, v2'), ('v_made_up_b32', 'v3, v2'), ('s_endpgm',))
        [f] = [f for f in r['findings'] if f['rule'] == 'OPERAND_MODEL_UNKNOWN']
        self.assertEqual((f['severity'], f['mnemonics'], f['pcs']), ('review', {'v_made_up_b32': 2}, [0, 4]))
        self.assertIn('v_made_up_b32 (2x)', f['message'])
        self.assertEqual(r['operand_model']['operand_model_unknown'], 2)

    def test_known_instructions_only(self):
        r = fake_kernel(('v_mov_b32', 'v1, v2'), ('s_endpgm',))
        self.assertEqual([f for f in r['findings'] if f['rule'] == 'OPERAND_MODEL_UNKNOWN'], [])

    def test_unknown_writes_are_conservative(self):
        # s4 is read by a vector instruction, then named by an unknown vector instruction: it may
        # write s4, so the later vector read needs the wait; graded 'review', since it rests on a guess.
        r = fake_kernel(('v_mov_b32', 'v0, s4'), ('v_made_up_b32', 's4, v1'), ('v_mov_b32', 'v2, s4'),
                        ('s_endpgm',))
        got = {f['rule']: f for f in r['findings']}
        self.assertEqual(got['MISSING_ALU_WAIT']['severity'], 'review')
        self.assertEqual(got['MISSING_ALU_WAIT']['operand_model'], 'assumed')
        self.assertEqual(got['MISSING_ALU_WAIT']['required_wait'], 's_wait_alu depctr_va_sdst(0)')
        # A known producer keeps its hazard an error, unknown instructions elsewhere or not.
        r = fake_kernel(('v_mov_b32', 'v0, s4'), ('v_made_up_b32', 'v5, v1'), ('s_mov_b32', 's4, 1'),
                        ('s_add_co_u32', 's6, s4, 1'), ('s_endpgm',))
        self.assertIn(('MISSING_ALU_WAIT', 'error'), rules_of(r['findings']))

    def test_classify_unknown(self):
        m = isadb.operand_model(TABLE, 's_made_up_b32', 's4, s5')
        self.assertEqual(sgpr.classify({'op': 's_made_up_b32', 'args': 's4, s5'}, m), ('salu', [4, 5], [4, 5]))
        m = isadb.operand_model(TABLE, 'x_made_up', 's4')
        self.assertEqual(sgpr.classify({'op': 'x_made_up', 'args': 's4'}, m), ('', [4], [4]))


class KernelPassTests(unittest.TestCase):
    def test_pass_sees_context_and_reports(self):
        seen = []

        def probe(ctx):
            seen.append(ctx)
            i = ctx['instructions'][1]
            return [{'rule': 'PROBE', 'severity': 'review', 'pc': i['pc'], 'message': 'x', 'source': 'test'}]
        with Registered(check.KERNEL_PASSES, probe):
            r = fake_kernel(('v_mov_b32', 'v1, v2'), ('v_add_f32', 'v3, v1, v1'), ('s_endpgm',))
        [ctx] = seen
        self.assertEqual(set(ctx), {'kernel', 'instructions', 'edges', 'reachable', 'table', 'wave', 'origin',
                                    'models', 'options'})
        self.assertEqual(sorted(ctx['models']), [0, 4, 8])
        self.assertTrue(ctx['models'][4]['known'])
        self.assertEqual((ctx['wave'], ctx['options'], ctx['table'] is TABLE), (32, {}, True))
        self.assertEqual(ctx['reachable'], {0, 4, 8})
        [f] = [f for f in r['findings'] if f['rule'] == 'PROBE']
        self.assertEqual(f['instruction'], 'v_add_f32 v3, v1, v1')

    def test_no_pass_no_finding(self):
        registered = list(check.KERNEL_PASSES)
        check.KERNEL_PASSES.clear()
        try:
            self.assertEqual(fake_kernel(('v_mov_b32', 'v1, v2'), ('s_endpgm',))['findings'], [])
        finally:
            check.KERNEL_PASSES[:] = registered

    def test_registered_passes(self):
        from radisa import liveness, memorder, suppress
        self.assertEqual(check.KERNEL_PASSES, [memorder.kernel_pass, liveness.kernel_pass])
        self.assertEqual(check.REPORT_FILTERS, [suppress.report_filter])

    def test_identity_follows_passes(self):
        before = check.analysis_identity()
        with Registered(check.KERNEL_PASSES, lambda ctx: []):
            self.assertNotEqual(check.analysis_identity(), before)
        self.assertEqual(check.analysis_identity(), before)
        self.assertNotEqual(check.analysis_identity({'x': 1}), before)


@needs_llvm_mc
class AnalyzeTests(unittest.TestCase):
    def test_analyze_prints_nothing(self):
        out = io.StringIO()
        with redirect_stdout(out):
            r = check.analyze(check_args(CHECK_FIX / 'missing_waits.s'))
        self.assertEqual(out.getvalue(), '')
        self.assertEqual((r['status'], r['exit_code']), ('FAIL', 1))
        code, j = run_check(CHECK_FIX / 'missing_waits.s')
        self.assertEqual((code, j['exit_code'], j['counts']), (1, 1, r['counts']))

    def test_options_reach_passes(self):
        seen = []
        with Registered(check.KERNEL_PASSES, lambda ctx: seen.append(ctx['options']) or []):
            args = check_args(CHECK_FIX / 'clean_waits.s')
            args.pass_options = {'rule': 'on'}
            self.assertEqual(check.analyze(args)['status'], 'PASS')
        self.assertEqual(seen, [{'rule': 'on'}])

    def test_report_filter_regrades_before_status(self):
        def soften(report, args):
            for f in all_findings(report):
                if f['rule'] == 'MISSING_WAIT':
                    f['severity'] = 'review'
        with Registered(check.REPORT_FILTERS, soften):
            code, r = run_check(CHECK_FIX / 'missing_waits.s')
        self.assertEqual((code, r['status'], r['counts']), (0, 'PASS', {'review': 3}))
        with Registered(check.REPORT_FILTERS, soften):
            self.assertEqual(run_check(CHECK_FIX / 'missing_waits.s', '--strict')[0], 1)

    def test_partly_decoded_snippet_is_incomplete(self):
        code, r = run_check(CHECK_FIX / 'undecodable.s')
        self.assertEqual((code, r['status']), (3, 'INCOMPLETE'))
        self.assertEqual({f['rule'] for f in all_findings(r)} & {'DECODE', 'BYTE_COVERAGE'}, {'DECODE', 'BYTE_COVERAGE'})
        # An error found before the undecodable bytes still fails.
        code, r = check_text('global_load_b32 v1, v[2:3], off\nv_add_f32 v2, v1, v1\n.long 0xffffffff\ns_endpgm\n')
        self.assertEqual((code, r['status']), (1, 'FAIL'))
        # The same code without the garbage dword is analysed whole.
        code, r = check_text('v_add_f32 v2, v1, v1\nglobal_load_b32 v1, v[4:5], off\ns_wait_loadcnt 0x0\n'
                             'v_add_f32 v2, v1, v1\ns_endpgm\n')
        self.assertEqual((code, r['status'], all_findings(r)), (0, 'PASS', []))


@needs_minimal_kernel
class KernelAsmTests(unittest.TestCase):
    def test_fixture_is_a_clean_kernel(self):
        code, r = run_check(CHECK_FIX / 'minimal_kernel.s')
        self.assertEqual((code, r['status'], all_findings(r)), (0, 'PASS', []))
        [k] = r['kernels']
        self.assertEqual((k['name'], k['instructions'], k['metadata']['.vgpr_count']), ('minimal', 8, 8))

    def test_descriptor_parameters(self):
        body = 'v_mov_b32_e32 v{} , 0\ns_endpgm'
        code, r = check_text(kernel_asm(body.format(15), name='wide', vgprs=16, lds=512))
        self.assertEqual((code, all_findings(r)), (0, []))
        k = r['kernels'][0]
        self.assertEqual((k['name'], k['metadata']['.group_segment_fixed_size'], k['resources']['descriptor_vgprs']),
                         ('wide', 512, 16))
        # A descriptor-dependent rule fires on the helper's kernel.
        code, r = check_text(kernel_asm(body.format(8), vgprs=8))
        self.assertEqual((code, [f['rule'] for f in all_findings(r)]), (1, ['VGPR_OUT_OF_RANGE']))

    def test_extra_functions_are_not_kernels(self):
        callee = '\t.p2align 2\n\t.type helper,@function\nhelper:\n\ts_setpc_b64 s[30:31]\n.Lhelper_end:\n' \
                 '\t.size helper, .Lhelper_end-helper\n'
        code, r = check_text(kernel_asm('s_endpgm', extra_functions=callee))
        # A non-kernel function is linted on its own (callee semantics), marked 'callee'.
        self.assertEqual((code, [(k['name'], bool(k.get('callee'))) for k in r['kernels']]),
                         (0, [('k', False), ('helper', True)]))
        # Nothing in the helper waits, so the caller's counters may still be pending at its return.
        self.assertEqual([(f['rule'], f['severity']) for f in all_findings(r)],
                         [('CALLEE_RETURN_OUTSTANDING', 'review')])


class SkipHelperTests(unittest.TestCase):
    def test_require_llvm_turns_skips_into_runs(self):
        def t(self):
            pass
        with mock.patch.dict(os.environ, {'RAD_ISA_REQUIRE_LLVM': '1'}):
            self.assertIs(_need('missing', 'x')(t), t)
        with mock.patch.dict(os.environ, {'RAD_ISA_REQUIRE_LLVM': ''}):
            self.assertTrue(getattr(_need('missing', 'x')(t), '__unittest_skip__', False))
            self.assertIs(_need('', 'x')(t), t)


if __name__ == '__main__':
    unittest.main()
