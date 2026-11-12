"""SYS_STORE_WAIT (radisa/memorder.py): opt-in, review-only, the literal reading of LLVM's
SIGfx12CacheControl::insertWaitsBeforeSystemScopeStore; and waits.solve, which it reuses."""
import unittest

from support import CHECK_FIX, check_args, needs_llvm_mc

from radisa import check, memorder, rules, waits


def report(path, enable=None):
    args = check_args(path)
    if enable is not None:
        args.pass_options = {'enable': enable}
    added = memorder.kernel_pass not in check.KERNEL_PASSES   # registered by check.py once integrated
    if added:
        check.KERNEL_PASSES.append(memorder.kernel_pass)
    try:
        return check.analyze(args)
    finally:
        if added:
            check.KERNEL_PASSES.remove(memorder.kernel_pass)


def rule(rep, name):
    return [f for f in check.all_findings(rep) if f['rule'] == name]


def ins(op, args, pc, size=4, raw='00000000'):
    return {'pc': pc, 'size': size, 'raw': raw, 'op': op, 'args': args, 'text': f'{op} {args}'}


class SysStoreWait(unittest.TestCase):
    @needs_llvm_mc
    def test_off_by_default(self):
        rep = report(CHECK_FIX / 'memorder_sys_store.s')
        self.assertEqual(rule(rep, 'SYS_STORE_WAIT'), [])
        self.assertEqual(rep['status'], 'PASS')

    @needs_llvm_mc
    def test_enabled_reports_pending_counters(self):
        rep = report(CHECK_FIX / 'memorder_sys_store.s', {'SYS_STORE_WAIT'})
        f, = rule(rep, 'SYS_STORE_WAIT')
        self.assertEqual(f['severity'], 'review')
        self.assertTrue(f['unconfirmed'])
        self.assertEqual(set(f['pending']), {'load', 'km', 'store'})
        self.assertEqual(f['required_wait'], ['s_wait_loadcnt 0', 's_wait_kmcnt 0', 's_wait_storecnt 0'])
        self.assertIn('scope:SCOPE_SYS', f['instruction'])
        self.assertIn('insertWaitsBeforeSystemScopeStore', f['source'])
        self.assertIn('unconfirmed', f['message'])
        self.assertEqual(rep['status'], 'PASS')            # review only
        self.assertEqual(check.finalize(rep, True)['status'], 'FAIL')

    @needs_llvm_mc
    def test_clean_with_llvm_waits(self):
        rep = report(CHECK_FIX / 'memorder_sys_store_clean.s', 'SYS_STORE_WAIT,OTHER')
        self.assertEqual(rule(rep, 'SYS_STORE_WAIT'), [])
        self.assertEqual(rep['status'], 'PASS')

    def test_enabled_parsing(self):
        self.assertEqual(memorder.enabled(None), set())
        self.assertEqual(memorder.enabled({'enable': 'sys_store_wait, x'}), {'SYS_STORE_WAIT', 'X'})
        self.assertEqual(memorder.enabled({'enable': ['SYS_STORE_WAIT']}), {'SYS_STORE_WAIT'})

    def test_store_selection(self):
        self.assertTrue(memorder.system_scope_store(ins('global_store_b32', 'v0, v1, s[0:1] scope:SCOPE_SYS', 0)))
        self.assertTrue(memorder.system_scope_store(ins('buffer_store_b32', 'v1, off, s[0:3], null scope:SCOPE_SYS', 0)))
        self.assertFalse(memorder.system_scope_store(ins('global_store_b32', 'v0, v1, s[0:1] scope:SCOPE_DEV', 0)))
        self.assertFalse(memorder.system_scope_store(ins('global_atomic_add_u32', 'v0, v1, s[0:1] scope:SCOPE_SYS', 0)))
        self.assertFalse(memorder.system_scope_store(ins('ds_store_b32', 'v0, v1', 0)))

    def test_callee_stores_after_call_are_pending(self):
        # updateEventWaitcntAfter / setStateOnFunctionEntryOrReturn: after a call only STOREcnt
        # may still be outstanding (the callee's stores).
        seq = [ins('global_load_b32', 'v1, v0, s[0:1]', 0, 8),
               ins('s_swappc_b64', 's[30:31], s[4:5]', 8),
               ins('global_store_b32', 'v0, v2, s[0:1] scope:SCOPE_SYS', 12, 8),
               ins('s_endpgm', '', 20)]
        edges, reachable, _ = rules.cfg(seq)
        ctx = {'kernel': {'name': 'k', 'descriptor': {}}, 'instructions': seq, 'edges': edges,
               'reachable': reachable, 'table': check.isadb.load(None), 'wave': 32, 'origin': None,
               'models': None, 'options': {'enable': {'SYS_STORE_WAIT'}}}
        f, = memorder.kernel_pass(ctx)
        self.assertEqual(f['pending'], {'store': [8]})
        self.assertIn('a store of the function s_swappc_b64 calls', f['message'])


class Solve(unittest.TestCase):
    def test_reuses_the_fixpoint_of_analyze(self):
        table = check.isadb.load(None)
        seq = [ins('global_load_b32', 'v1, v0, s[0:1]', 0, 8), ins('s_endpgm', '', 8)]
        edges, reachable, _ = rules.cfg(seq)
        models = waits.operand_models(seq, table)
        waits.analyze(seq, edges, reachable, table, 32, None, models)
        first = waits.solve(seq, edges, table, 32, models)
        self.assertIs(waits.solve(seq, edges, table, 32, models), first)
        state_in, m, status = first
        self.assertIs(m, models)
        self.assertEqual(status, 'converged')
        self.assertEqual(set(state_in[8]['load']), {0})
        self.assertEqual(state_in[8]['load'][0][0], frozenset({('v', 1)}))
        # A different callee flag is a different problem.
        self.assertIn(waits.ENTRY, waits.solve(seq, edges, table, 32, models, callee=True)[0][0]['load'])


if __name__ == '__main__':
    unittest.main()
