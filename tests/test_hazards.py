"""radisa.hazards: LLVM-fixed gfx1201 hazards over the whole direct CFG, DEALLOC_NO_NOP, s_clause lint."""
import os, shutil, tempfile, unittest
from pathlib import Path
from radisa import hazards, isadb
from radisa.rules import cfg, local_checks
from support import (seq, CHECK_FIX, run_check, all_findings, needs_llvm_mc, needs_minimal_kernel, kernel_asm,
                     check_text, needs_hip)


def gfx1201_compiler():
    for c in (os.environ.get('RADISA_TEST_CXX'), '/usr/lib/llvm-20/bin/clang++', '/opt/rocm/llvm/bin/clang++',
              shutil.which('clang++')):
        if c and Path(c).exists():
            return c
    return None

W1 = ('v_wmma_f32_16x16x16_f16', 'v[8:15], v[0:3], v[4:7], v[8:15]')
W_A = ('v_wmma_f32_16x16x16_f16', 'v[16:23], v[8:11], v[4:7], v[16:23]')   # A overlaps W1's D
W_C = ('v_wmma_f32_16x16x16_f16', 'v[8:15], v[0:3], v[4:7], v[8:15]')     # C overlaps W1's D only
CMPX = ('v_cmpx_gt_u32', 'v0, v1')
PERM = ('v_permlane16_b32', 'v2, v3, s0, s1')


def run(*spec):
    x = seq(*spec)
    e, r, _ = cfg(x)
    return hazards.analyze(x, e, r)[0]


def rules(*spec):
    return [f['rule'] for f in run(*spec)]


class WmmaTests(unittest.TestCase):
    def test_adjacent(self):
        f = run(W1, W_A, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['severity'], x['pc'], x['producer']) for x in f],
                         [('WMMA_D_TO_AB', 'error', 4, 0)])
        self.assertIn('fixWMMAHazards', f[0]['source'])
        self.assertEqual(f[0]['registers'], ['v8', 'v9', 'v10', 'v11'])

    def test_across_branch(self):
        # W1; s_cbranch_scc0 -> W_A (skips the VALU on the fall-through path)
        self.assertIn('WMMA_D_TO_AB', rules(W1, ('s_cbranch_scc0', '1', 1), ('v_mov_b32', 'v24, v25'),
                                            W_A, ('s_endpgm',)))

    def test_valu_on_every_path(self):
        self.assertEqual(rules(W1, ('s_cbranch_scc0', '2', 2), ('v_nop',), ('s_branch', '1', 1),
                               ('v_add_f32', 'v24, v25, v26'), W_A, ('s_endpgm',)), [])

    def test_back_edge(self):
        loop = rules(('v_wmma_f32_16x16x16_f16', 'v[16:23], v[0:3], v[8:11], v[16:23]'), W1,
                     ('s_sub_co_u32', 's0, s0, 1'), ('s_cbranch_scc1', '-4', -4), ('s_endpgm',))
        self.assertEqual(loop, ['WMMA_D_TO_AB'])

    def test_c_overlap_allowed(self):
        self.assertEqual(rules(W1, W_C, ('s_endpgm',)), [])

    def test_salu_and_snop_do_not_separate(self):
        self.assertIn('WMMA_D_TO_AB', rules(W1, ('s_nop', '0'), ('s_mov_b32', 's0, 0'), W_A, ('s_endpgm',)))

    def test_vnop_and_any_valu_separate(self):
        self.assertEqual(rules(W1, ('v_nop',), W_A, ('s_endpgm',)), [])
        self.assertEqual(rules(W1, ('v_mov_b32', 'v30, v31'), W_A, ('s_endpgm',)), [])

    def test_intervening_wmma_expires_earlier_one(self):
        # A non-overlapping WMMA is a VALU: the earlier D is no longer a hazard (LLVM IsExpired).
        self.assertEqual(rules(W1, ('v_wmma_f32_16x16x16_f16', 'v[32:39], v[0:3], v[4:7], v[32:39]'), W_A,
                               ('s_endpgm',)), [])

    def test_swmmac_index(self):
        f = run(W1, ('v_swmmac_f32_16x16x32_f16', 'v[16:23], v[0:3], v[24:31], v8'), ('s_endpgm',))
        self.assertEqual([x['rule'] for x in f], ['WMMA_D_TO_AB'])
        self.assertIn('index (v8)', f[0]['message'])

    def test_swmmac_no_overlap(self):
        self.assertEqual(rules(W1, ('v_swmmac_f32_16x16x32_f16', 'v[16:23], v[0:3], v[24:31], v40'),
                               ('s_endpgm',)), [])

    def test_unreachable_not_reported(self):
        self.assertEqual(rules(('s_endpgm',), W1, W_A, ('s_endpgm',)), [])


class PermlaneTests(unittest.TestCase):
    def test_adjacent(self):
        f = run(CMPX, PERM, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['severity'], x['producer']) for x in f], [('PERMLANE_AFTER_CMPX', 'error', 0)])
        self.assertIn('v_mov_b32 v3, v3', f[0]['message'])
        self.assertIn('v_cmpx_gt_u32 at pc 0x0', f[0]['message'])

    def test_message_lists_at_most_six_producers(self):
        f = run(*scale(10, 2))
        self.assertEqual(len(f), 1)
        self.assertIn('v_cmpx_gt_u32 at pc 0x28 and 4 more', f[0]['message'])
        self.assertNotIn('pc 0x30', f[0]['message'])
        self.assertEqual(f[0]['producers'], list(range(0, 80, 8)))

    def test_snop_vnop_salu_do_not_clear(self):
        self.assertIn('PERMLANE_AFTER_CMPX', rules(CMPX, ('s_nop', '0'), PERM, ('s_endpgm',)))
        self.assertIn('PERMLANE_AFTER_CMPX', rules(CMPX, ('v_nop',), PERM, ('s_endpgm',)))
        self.assertIn('PERMLANE_AFTER_CMPX', rules(CMPX, ('s_mov_b32', 's4, 0'), ('v_nop',), PERM, ('s_endpgm',)))

    def test_valu_clears(self):
        self.assertEqual(rules(CMPX, ('v_mov_b32', 'v3, v3'), PERM, ('s_endpgm',)), [])

    def test_across_branch(self):
        self.assertIn('PERMLANE_AFTER_CMPX', rules(CMPX, ('s_cbranch_execz', '1', 1),
                                                   ('v_add_f32', 'v4, v5, v6'), ('v_nop',), PERM, ('s_endpgm',)))

    def test_vop3_compare_into_exec(self):
        self.assertIn('PERMLANE_AFTER_CMPX', rules(('v_cmp_eq_u32_e64', 'exec_lo, v0, v1'), PERM, ('s_endpgm',)))

    def test_compare_into_vcc_is_no_producer(self):
        self.assertEqual(rules(('v_cmp_eq_u32', 'vcc_lo, v0, v1'), PERM, ('s_endpgm',)), [])

    def test_other_permlanes(self):
        # The VAR forms take vdst, src0, src1 (llvm-mc rejects 'v2, v3': too few operands).
        for p in (('v_permlanex16_b32', 'v2, v3, s0, s1'), ('v_permlane16_var_b32', 'v2, v3, v4'),
                  ('v_permlanex16_var_b32', 'v2, v3, v4')):
            self.assertIn('PERMLANE_AFTER_CMPX', rules(CMPX, p, ('s_endpgm',)), p)


def scale(p, l):
    """p blocks `v_cmpx; s_cbranch_scc1 JOIN` converging at JOIN, then l s_nop and a permlane."""
    spec = []
    for k in range(p):
        spec += [CMPX, ('s_cbranch_scc1', '', 2 * p - (2 * k + 1) - 1)]
    return spec + [('s_nop', '0')] * l + [PERM, ('s_endpgm',)]


class ScaleTests(unittest.TestCase):
    def test_many_converging_producers_is_fast(self):
        # Reviewer input shape: per-instruction producer sets took 12 s for 400 producers x 4000 s_nop
        # (quadratic); per-block bitmasks are linear in the blocks and edges.
        import time
        x = seq(*scale(2000, 20000))
        e, r, _ = cfg(x)
        t = time.process_time()
        f, _ = hazards.analyze(x, e, r)
        self.assertLess(time.process_time() - t, 10)
        self.assertEqual([(y['rule'], len(y['producers'])) for y in f], [('PERMLANE_AFTER_CMPX', 2000)])

    def test_wmma_through_chained_joins(self):
        # W1 reaches W_A along s_nop-only blocks joined by both arms of every branch.
        spec = [W1]
        for _ in range(50):
            spec += [('s_cbranch_scc0', '1', 1), ('s_nop', '0')]
        f = run(*spec, W_A, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['producer']) for x in f], [('WMMA_D_TO_AB', 0)])


class CfgTests(unittest.TestCase):
    """rules.cfg edges the hazard dataflow walks: no fall-through after s_setpc_b64 (as LLVM's MBB
    predecessor lists), long branches resolved, noreturn calls and llvm.trap end the path."""

    @staticmethod
    def long_branch(offset, filler=(), reg='s[2:3]', lo='s2', hi='s3'):
        return [('s_getpc_b64', reg), *filler, ('s_add_co_u32', f'{lo}, {lo}, {offset:#x}'),
                ('s_add_co_ci_u32', f'{hi}, {hi}, {-1 if offset < 0 else 0}'), *filler, ('s_setpc_b64', reg)]

    def test_return_does_not_fall_through(self):
        # Reviewer reproducer q1_setpc.s: L2 is reached only from the s_cbranch_scc0.
        x = (('s_cbranch_scc0', '2', 2), W1, ('s_setpc_b64', 's[30:31]'), W_A, ('s_setpc_b64', 's[30:31]'))
        self.assertEqual(rules(*x), [])
        e, _, f = cfg(seq(*x))
        self.assertEqual(e[8], [])
        self.assertEqual([y['rule'] for y in f], ['CFG_INDIRECT', 'CFG_INDIRECT'])

    def test_long_branch_does_not_fall_through(self):
        # Reviewer reproducer q2_longbranch.s with the target inside the code: pc 12 + 0x14 = 32.
        x = [('s_cbranch_scc0', '5', 5), CMPX, *self.long_branch(0x14), PERM, ('s_endpgm',), ('s_endpgm',)]
        self.assertEqual(rules(*x), [])
        e, r, f = cfg(seq(*x))
        self.assertEqual((e[20], f, 32 in r), ([32], [], True))

    def test_long_branch_target_is_followed(self):
        # The jump itself is a real path: the permlane at the target (pc 8 + 0x10 = 24) is reported.
        f = run(CMPX, *self.long_branch(0x10), ('s_endpgm',), PERM, ('s_endpgm',))
        self.assertEqual([(y['rule'], y['pc'], y['producer']) for y in f], [('PERMLANE_AFTER_CMPX', 24, 0)])

    def test_long_branch_backwards_with_wait_alu(self):
        # LLVM 20+ puts s_wait_alu after s_getpc_b64 and after the add pair (ApplyHazardWorkarounds).
        x = seq(('s_nop', '0'), *self.long_branch(-8, filler=[('s_wait_alu', '0xfffe')]))
        e, r, f = cfg(x)
        self.assertEqual((e[x[-1]['pc']], f), ([0], []))

    def test_unrecognised_setpc_has_no_successor(self):
        for x in (self.long_branch(0x4, reg='s[4:5]'),             # adds to another pair
                  self.long_branch(0x4)[1:],                        # no s_getpc_b64
                  [('s_mov_b64', 's[2:3]', 0), ('s_setpc_b64', 's[2:3]')]):
            e, r, f = cfg(seq(*x, ('v_mov_b32', 'v0, 1'), ('s_endpgm',)))
            last = len(x) * 4 - 4
            self.assertEqual((e[last], [y['rule'] for y in f]), ([], ['CFG_INDIRECT']), x)

    def test_long_branch_out_of_range(self):
        e, r, f = cfg(seq(*self.long_branch(0x100), ('s_endpgm',)))
        self.assertEqual([(y['rule'], y['target']) for y in f], [('CFG_TARGET', 0x104)])

    def test_noreturn_call_and_trap(self):
        # clang -O2 for `[[noreturn]] die()` (store; __builtin_trap()) called last in a kernel.
        for x in [(('v_mov_b32', 'v0, 1'), ('s_swappc_b64', 's[30:31], s[0:1]')),
                  (('v_mov_b32', 'v0, 1'), ('s_swappc_b64', 's[30:31], s[0:1]'), ('s_code_end',)),
                  (('v_mov_b32', 'v0, 1'), ('s_trap', '2', 2)),
                  (('s_trap', '2', 2), ('s_code_end',))]:
            e, r, f = cfg(seq(*x))
            self.assertNotIn('CFG_FALLTHROUGH', [y['rule'] for y in f], x)
            self.assertNotIn('REACHABLE_CODE_END', [y['rule'] for y in f], x)
        f = cfg(seq(('s_swappc_b64', 's[30:31], s[0:1]')))[2]
        self.assertEqual([(y['rule'], y.get('noreturn')) for y in f], [('CFG_INDIRECT', True)])

    def test_real_fall_through_still_reported(self):
        # llvm.debugtrap (s_trap 3) continues; a returning call returns to the next instruction;
        # a missing s_endpgm and a conditional branch at the end still fall through.
        for x in [(('s_trap', '3', 3),), (('s_swappc_b64', 's[30:31], s[0:1]'), ('v_mov_b32', 'v0, 1')),
                  (('v_mov_b32', 'v0, 1'),), (('s_cbranch_scc0', '0', 0),)]:
            f = cfg(seq(*x))[2]
            self.assertIn(('CFG_FALLTHROUGH', 'error'), [(y['rule'], y['severity']) for y in f], x)
        self.assertEqual(cfg(seq(('s_trap', '3', 3), ('s_code_end',)))[2][0]['rule'], 'REACHABLE_CODE_END')
        # The hazard dataflow still follows a returning call.
        self.assertEqual(rules(CMPX, ('s_swappc_b64', 's[30:31], s[0:1]'), PERM, ('s_endpgm',)),
                         ['PERMLANE_AFTER_CMPX'])


class DeallocTests(unittest.TestCase):
    MSG = ('s_sendmsg', 'sendmsg(MSG_DEALLOC_VGPRS)', 3)

    def test_missing_nop(self):
        f = run(('v_mov_b32', 'v1, v2'), self.MSG, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['severity']) for x in f], [('DEALLOC_NO_NOP', 'review')])
        self.assertIn('requiresNopBeforeDeallocVGPRs', f[0]['source'])
        self.assertIn('not documented', f[0]['message'])

    def test_with_nop(self):
        self.assertEqual(rules(('v_mov_b32', 'v1, v2'), ('s_nop', '0'), self.MSG, ('s_endpgm',)), [])

    def test_branch_predecessor_without_nop(self):
        # s_branch jumps straight to the message; the fall-through predecessor has its s_nop.
        self.assertIn('DEALLOC_NO_NOP', rules(('s_cbranch_scc0', '2', 2), ('v_mov_b32', 'v1, v2'),
                                              ('s_nop', '0'), self.MSG, ('s_endpgm',)))

    def test_other_message(self):
        self.assertEqual(rules(('s_sendmsg', 'sendmsg(MSG_INTERRUPT)', 1), ('s_endpgm',)), [])


class ClauseTests(unittest.TestCase):
    LOAD = ('global_load_b32', 'v1, v0, s[0:1]')

    def test_too_long(self):
        f = run(('s_clause', '0x20', 0x20), *[self.LOAD] * 33, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['severity'], x['clause_length']) for x in f],
                         [('CLAUSE_TOO_LONG', 'review', 33)])

    def test_32_is_allowed(self):
        self.assertEqual(rules(('s_clause', '0x1f', 0x1f), *[self.LOAD] * 32, ('s_endpgm',)), [])

    def test_mixed_load_store(self):
        f = run(('s_clause', '0x1', 1), self.LOAD, ('global_store_b32', 'v0, v5, s[0:1]'), ('s_endpgm',))
        self.assertEqual([x['rule'] for x in f], ['CLAUSE_MIXED'])
        self.assertEqual(f[0]['kinds'], ['vmem_load', 'vmem_store'])

    def test_global_and_flat_differ(self):
        self.assertEqual(rules(('s_clause', '0x1', 1), self.LOAD, ('flat_load_b32', 'v2, v[4:5]'), ('s_endpgm',)),
                         ['CLAUSE_MIXED'])

    def test_illegal_inside(self):
        f = run(('s_clause', '0x2', 2), self.LOAD, ('v_add_f32', 'v2, v3, v3'), self.LOAD, ('s_endpgm',))
        self.assertEqual([x['rule'] for x in f], ['CLAUSE_MIXED'])
        self.assertEqual(f[0]['illegal_pcs'], [8])
        self.assertEqual(rules(('s_clause', '0x1', 1), ('ds_load_b32', 'v1, v0'), ('ds_load_b32', 'v2, v0'),
                               ('s_endpgm',)), ['CLAUSE_MIXED'])

    def test_snop_inside_and_smem(self):
        self.assertEqual(rules(('s_clause', '0x2', 2), ('s_load_b64', 's[4:5], s[0:1], 0x0'), ('s_nop', '0'),
                               ('s_load_b64', 's[6:7], s[0:1], 0x8'), ('s_endpgm',)), [])

    def test_buffer_and_global_share_a_type(self):
        self.assertEqual(rules(('s_clause', '0x1', 1), self.LOAD,
                               ('buffer_load_b32', 'v2, off, s[4:7], 0'), ('s_endpgm',)), [])

    def test_uncertain_kinds_are_not_reported(self):
        # s_prefetch_data: the table does not say whether LLVM treats it as a load.
        self.assertEqual(rules(('s_clause', '0x1', 1), ('s_load_b32', 's4, s[0:1], 0x0'),
                               ('s_prefetch_data', 's[0:1], 0x0, null, 0'), ('s_endpgm',)), [])

    def test_branch_into_body(self):
        f = run(('s_cbranch_scc0', '2', 2), ('s_clause', '0x1', 1), self.LOAD, self.LOAD, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['targets']) for x in f], [('CLAUSE_BRANCH', [12])])

    def test_branch_to_clause_start_is_fine(self):
        self.assertEqual(rules(('s_cbranch_scc0', '0', 0), ('s_clause', '0x1', 1), self.LOAD, self.LOAD,
                               ('s_endpgm',)), [])


class IntegrationTests(unittest.TestCase):
    def test_local_checks_carries_the_rules_once(self):
        x = seq(W1, W_A, CMPX, PERM, ('s_endpgm',))
        e, r, _ = cfg(x)
        got = [f['rule'] for f in local_checks(x, e, r)[0]]
        self.assertEqual(sorted(got), ['PERMLANE_AFTER_CMPX', 'WMMA_D_TO_AB'])

    def test_models_argument(self):
        from radisa.waits import operand_models
        x = seq(CMPX, PERM, ('s_endpgm',))
        e, r, _ = cfg(x)
        t = isadb.load()
        self.assertEqual([f['rule'] for f in hazards.analyze(x, e, r, operand_models(x, t), t)[0]],
                         ['PERMLANE_AFTER_CMPX'])


def fixture_rules(name):
    code, rep = run_check(CHECK_FIX / name)
    return code, sorted(f['rule'] for f in all_findings(rep))


class FixtureTests(unittest.TestCase):
    @needs_llvm_mc
    def test_failing_fixtures(self):
        for name, code, want in [
                ('hazard_wmma_xblock.s', 1, ['WMMA_D_TO_AB']),
                ('hazard_wmma_loop.s', 1, ['WMMA_D_TO_AB']),
                ('hazard_swmmac_index.s', 1, ['WMMA_D_TO_AB']),
                ('hazard_permlane_xblock.s', 1, ['PERMLANE_AFTER_CMPX']),
                ('hazard_longbranch.s', 1, ['PERMLANE_AFTER_CMPX']),
                ('hazard_dealloc.s', 0, ['DEALLOC_NO_NOP']),
                ('hazard_clause.s', 0, ['CLAUSE_BRANCH', 'CLAUSE_MIXED', 'CLAUSE_TOO_LONG'])]:
            self.assertEqual(fixture_rules(name), (code, want), name)

    @needs_llvm_mc
    def test_clean_fixtures(self):
        for name in ('hazard_wmma_clean.s', 'hazard_permlane_clean.s', 'hazard_dealloc_clean.s',
                     'hazard_clause_clean.s', 'hazard_longbranch_clean.s'):
            self.assertEqual(fixture_rules(name), (0, []), name)
        # The CALLS note is the only finding: no CFG_FALLTHROUGH, no REACHABLE_CODE_END.
        self.assertEqual(fixture_rules('hazard_noreturn_clean.s'), (0, ['CALLS']))

    @needs_llvm_mc
    def test_long_branch_finding_is_at_the_target(self):
        _, rep = run_check(CHECK_FIX / 'hazard_longbranch.s')
        self.assertEqual([(f['rule'], f['pc'], f['producer']) for f in all_findings(rep)],
                         [('PERMLANE_AFTER_CMPX', 0x2c, 0x4)])

    @needs_llvm_mc
    def test_strict_fails_on_review(self):
        code, _ = run_check(CHECK_FIX / 'hazard_dealloc.s', '--strict')
        self.assertEqual(code, 1)

    @needs_minimal_kernel
    def test_whole_kernel(self):
        body = '''v_cmpx_gt_u32 v0, v1
            s_cbranch_execz 1
            v_add_f32 v4, v5, v6
            v_nop
            v_permlane16_b32 v2, v3, s0, s1
            s_nop 0
            s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
            s_endpgm'''
        # The body reads registers nothing defines (v1, v3, v5, v6): UNINITIALIZED_READ is not under test.
        code, rep = check_text(kernel_asm(body), '--no-liveness')
        self.assertEqual((code, sorted(f['rule'] for f in all_findings(rep))), (1, ['PERMLANE_AFTER_CMPX']))
        code, rep = check_text(kernel_asm(body.replace('v_nop', 'v_mov_b32 v3, v3')), '--no-liveness')
        self.assertEqual((code, [f['rule'] for f in all_findings(rep)]), (0, []))


class CompiledTests(unittest.TestCase):
    """rules.cfg on unedited compiler output (clang 18 here; LLVM 20+ adds s_wait_alu in long branches)."""
    def check(self, name, *flags):
        """(report, text of the compiler's .s files)."""
        with tempfile.TemporaryDirectory() as d:
            rep = run_check(CHECK_FIX / name, '--cxx', gfx1201_compiler(), '--flag', '-nogpuinc -nogpulib', *flags,
                            '--no-incremental', '--cache-dir', d, '--keep', d, '--no-liveness')[1]
            return rep, ''.join(p.read_text() for p in Path(d).rglob('*.s'))

    @needs_hip(gfx1201_compiler(), CHECK_FIX / 'hazard_noreturn.hip')
    def test_noreturn_call_and_trap(self):
        rep, asm = self.check('hazard_noreturn.hip')
        self.assertIn('s_trap 2', asm)
        self.assertEqual([k['name'] for k in rep['kernels'] if not k.get('callee')][:1], ['kn'])
        got = {f['rule'] for f in all_findings(rep)}
        self.assertFalse(got & {'CFG_FALLTHROUGH', 'REACHABLE_CODE_END'}, got)

    @needs_hip(gfx1201_compiler(), CHECK_FIX / 'hazard_longbranch.hip')
    def test_long_branches_resolved(self):
        rep, asm = self.check('hazard_longbranch.hip', '--flag', '-mllvm -amdgpu-s-branch-bits=5')
        got = {f['rule'] for f in all_findings(rep)}
        # Every long branch is a direct edge: no CALLS note, no CFG_TARGET, no fall-through finding.
        self.assertFalse(got & {'CALLS', 'CFG_TARGET', 'CFG_INDIRECT', 'CFG_FALLTHROUGH', 'REACHABLE_CODE_END'}, got)
        self.assertIn('s_setpc_b64', asm, 'the compiler emitted no long branch: the test proves nothing')


if __name__ == '__main__':
    unittest.main()
