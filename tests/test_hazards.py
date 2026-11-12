"""radisa.hazards: LLVM-fixed gfx1201 hazards over the whole direct CFG, DEALLOC_NO_NOP, s_clause lint."""
import unittest
from radisa import hazards, isadb
from radisa.rules import cfg, local_checks
from support import seq, CHECK_FIX, run_check, all_findings, needs_llvm_mc, needs_minimal_kernel, kernel_asm, check_text

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
        for p in (('v_permlanex16_b32', 'v2, v3, s0, s1'), ('v_permlane16_var_b32', 'v2, v3'),
                  ('v_permlanex16_var_b32', 'v2, v3')):
            self.assertIn('PERMLANE_AFTER_CMPX', rules(CMPX, p, ('s_endpgm',)), p)


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
                ('hazard_dealloc.s', 0, ['DEALLOC_NO_NOP']),
                ('hazard_clause.s', 0, ['CLAUSE_BRANCH', 'CLAUSE_MIXED', 'CLAUSE_TOO_LONG'])]:
            self.assertEqual(fixture_rules(name), (code, want), name)

    @needs_llvm_mc
    def test_clean_fixtures(self):
        for name in ('hazard_wmma_clean.s', 'hazard_permlane_clean.s', 'hazard_dealloc_clean.s',
                     'hazard_clause_clean.s'):
            self.assertEqual(fixture_rules(name), (0, []), name)

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


if __name__ == '__main__':
    unittest.main()
