"""Non-kernel functions (radisa/functions.py) and the call/callee conventions of radisa/waits.py,
both as LLVM's SIInsertWaitcnts implements them (see waits.CALLS)."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from support import CHECK_FIX, llvm_mc, needs_minimal_kernel, kernel_asm, check_text, all_findings

from radisa import binary, check, decode, functions, isadb, rules, sgpr, waits

TABLE = isadb.load(None)


def assemble(text: str, out: Path) -> binary.ELF:
    src = out.with_suffix('.s')
    src.write_text(text)
    subprocess.run([llvm_mc(), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-filetype=obj', str(src),
                    '-o', str(out)], check=True, capture_output=True)
    return binary.ELF(out.read_bytes())


def lint(text: str) -> dict:
    """{name: (inventory entry, findings, stats)} for every kernel and function of the assembled text,
    the way check_object_bytes runs them once functions are wired in."""
    with tempfile.TemporaryDirectory() as d:
        obj = Path(d, 'a.o')
        elf = assemble(text, obj)
        ks, issues = binary.kernel_inventory(elf)
        fs = functions.function_inventory(elf, ks, issues)
        dec = decode.decode_object(obj, [k for k in ks + fs if 'code_hex' in k], decode.select_backend(None, None))
    out = {}
    for k in ks + fs:
        ins = dec[k['symbol']]['instructions']
        edges, reachable, _ = rules.cfg(ins)
        found, stats = waits.analyze(ins, edges, reachable, TABLE, 32, callee=bool(k.get('callee')))
        out[k['name']] = (k, found, stats)
    return out


def rule(found, name):
    return [f for f in found if f['rule'] == name]


def ins(op, args, pc, size=4):
    return {'pc': pc, 'size': size, 'raw': '00000000', 'op': op, 'args': args, 'text': f'{op} {args}'}


def run(seq, callee=False):
    edges, reachable, _ = rules.cfg(seq)
    return waits.analyze(seq, edges, reachable, TABLE, 32, callee=callee)


HELPER = '''\t.p2align\t2
\t.type\t{name},@function
{name}:
{body}.Lfunc_end_{name}:
\t.size\t{name}, .Lfunc_end_{name}-{name}
'''


def helper(name, body):
    return HELPER.format(name=name, body=''.join(f'\t{l.strip()}\n' for l in body.strip().splitlines()))


class Inventory(unittest.TestCase):
    @needs_minimal_kernel
    def test_lists_functions_not_kernels(self):
        text = kernel_asm('s_endpgm', extra_functions=helper('f1', 's_setpc_b64 s[30:31]')
                          + '\t.type\tnosize,@function\nnosize:\n\ts_setpc_b64 s[30:31]\n'
                          + '\t.globl\talias\n\t.type\talias,@function\n\t.set\talias, f1\n\t.size\talias, 4\n')
        with tempfile.TemporaryDirectory() as d:
            elf = assemble(text, Path(d, 'a.o'))
            ks, _ = binary.kernel_inventory(elf)
            issues = []
            fs = functions.function_inventory(elf, ks, issues)
        f, = fs                                      # f1 and its alias are one function
        self.assertIn(f['name'], ('f1', 'alias'))
        self.assertTrue(f['callee'])
        self.assertEqual(f['descriptor'], {})
        self.assertEqual(f['metadata']['.wavefront_size'], 32)
        self.assertEqual(f['size'], 4)
        self.assertEqual(bytes.fromhex(f['code_hex']).hex(), f['code_hex'])
        self.assertEqual(f['address'], ks[0]['address'] + ks[0]['size'])
        self.assertTrue(any('nosize' in m for m in issues))

    @needs_minimal_kernel
    def test_kernel_alone_has_no_functions(self):
        with tempfile.TemporaryDirectory() as d:
            elf = assemble(kernel_asm('s_endpgm'), Path(d, 'a.o'))
            ks, _ = binary.kernel_inventory(elf)
            self.assertEqual(functions.function_inventory(elf, ks), [])


class CalleeConvention(unittest.TestCase):
    @needs_minimal_kernel
    def test_missing_entry_and_return_waits(self):
        r = lint((CHECK_FIX / 'callee_missing_waits.s').read_text())
        k, kf, _ = r['k']
        self.assertEqual(kf, [])                     # the caller relies on the convention
        h, hf, stats = r['helper']
        self.assertTrue(h['callee'])
        mw, = rule(hf, 'MISSING_WAIT')
        self.assertEqual(mw['severity'], 'error')
        self.assertIsNone(mw['producer'])
        self.assertEqual(mw['producer_kind'], 'caller')
        self.assertIn('v1', mw['registers'])
        self.assertIn('s_wait_loadcnt 0', mw['required_wait'])
        self.assertIn('SIInsertWaitcnts', mw['source'])
        ret, = rule(hf, 'CALLEE_RETURN_OUTSTANDING')
        self.assertEqual(ret['severity'], 'review')
        self.assertIn('load', ret['pending'])
        self.assertIn('s_wait_loadcnt 0', ret['required_wait'])
        self.assertEqual(stats['callee_returns_outstanding'], 1)
        self.assertEqual(rule(hf, 'CFG_FALLTHROUGH'), [])

    @needs_minimal_kernel
    def test_compiled_callee_is_clean(self):
        # llc 18 output: the prologue waits and the wait before the return are LLVM's own.
        r = lint((CHECK_FIX / 'callee_compiled.s').read_text())
        self.assertEqual(set(r), {'k', 'helper'})
        for name, (_, found, _) in r.items():
            self.assertEqual(found, [], name)

    def test_entry_waits_clear_the_caller(self):
        prologue = [ins('s_wait_loadcnt_dscnt', '0x0', 0), ins('s_wait_expcnt', '0x0', 4),
                    ins('s_wait_samplecnt', '0x0', 8), ins('s_wait_bvhcnt', '0x0', 12),
                    ins('s_wait_kmcnt', '0x0', 16)]
        for p in prologue:
            p['raw'] = '00000000'
        body = [ins('v_add_f32_e32', 'v2, v1, v1', 20), ins('s_setpc_b64', 's[30:31]', 24)]
        self.assertEqual(run(prologue + body, callee=True)[0], [])
        found, _ = run([dict(i, pc=i['pc'] - 20) for i in body], callee=True)
        self.assertEqual([f['rule'] for f in found], ['MISSING_WAIT', 'CALLEE_RETURN_OUTSTANDING'])
        # The same code as a kernel: nothing is outstanding at a kernel's entry.
        self.assertEqual(run([dict(i, pc=i['pc'] - 20) for i in body])[0], [])

    def test_store_and_global_inv_may_stay_outstanding_at_return(self):
        # generateWaitcntInstBefore: getAllZeroWaitcnt(IncludeVSCnt=false) + simplifyVmemInvOnlyWait.
        seq = [ins('s_wait_loadcnt_dscnt', '0x0', 0), ins('s_wait_expcnt', '0x0', 4),
               ins('s_wait_samplecnt', '0x0', 8), ins('s_wait_bvhcnt', '0x0', 12), ins('s_wait_kmcnt', '0x0', 16),
               ins('global_store_b32', 'v0, v1, s[0:1]', 20, 8), ins('global_inv', 'scope:SCOPE_SYS', 28, 8),
               ins('s_setpc_b64', 's[30:31]', 36)]
        self.assertEqual(rule(run(seq, callee=True)[0], 'CALLEE_RETURN_OUTSTANDING'), [])
        seq.insert(7, ins('global_load_b32', 'v3, v0, s[0:1]', 36, 8))
        seq[-1] = ins('s_setpc_b64', 's[30:31]', 44)
        f, = rule(run(seq, callee=True)[0], 'CALLEE_RETURN_OUTSTANDING')
        self.assertEqual(f['pending'], {'load': [28, 36]})

    def test_tail_call_is_not_a_return(self):
        seq = [ins('global_load_b32', 'v3, v0, s[0:1]', 0, 8), ins('s_setpc_b64', 's[4:5]', 8)]
        found, _ = run(seq, callee=True)
        self.assertEqual(rule(found, 'CALLEE_RETURN_OUTSTANDING'), [])

    def test_callee_tracks_every_sgpr_pair(self):
        # AMDGPUWaitSGPRHazards: "Callee must consider all SGPRs as tracked."
        seq = [ins('s_mov_b32', 's4, 1', 0), ins('s_add_co_u32', 's5, s4, 1', 4), ins('s_endpgm', '', 8)]
        edges, reachable, _ = rules.cfg(seq)
        models = waits.operand_models(seq, TABLE)
        self.assertEqual(sgpr.analyze(seq, edges, reachable, models)[0], [])
        f, = sgpr.analyze(seq, edges, reachable, models, callee=True)[0]
        self.assertEqual(f['rule'], 'MISSING_ALU_WAIT')
        # Through waits.analyze the entry MISSING_WAIT comes first: s_mov_b32 overwrites s4 while
        # a scalar load of the caller may still write it.
        self.assertEqual([x['rule'] for x in run(seq, callee=True)[0]], ['MISSING_WAIT', 'MISSING_ALU_WAIT'])


class CallSite(unittest.TestCase):
    def test_call_drains_all_but_storecnt(self):
        seq = [ins('global_load_b32', 'v1, v0, s[0:1]', 0, 8), ins('global_store_b32', 'v0, v1, s[2:3]', 8, 8),
               ins('s_swappc_b64', 's[30:31], s[4:5]', 16), ins('v_add_f32_e32', 'v2, v1, v1', 20),
               ins('s_endpgm', '', 24)]
        edges, reachable, _ = rules.cfg(seq)
        # The store reads v1 the load writes: that one is a real MISSING_WAIT, not the add after the call.
        found, _ = waits.analyze(seq, edges, reachable, TABLE)
        self.assertEqual([(f['rule'], f['pc']) for f in found], [('MISSING_WAIT', 8)])
        state = waits.solve(seq, edges, TABLE)[0][20]
        self.assertEqual({d: set(e) for d, e in state.items() if e}, {'store': {8, 16}})

    def test_call_waits_for_scalar_load_of_its_address(self):
        seq = [ins('s_load_b64', 's[4:5], s[0:1], 0x0', 0, 8), ins('s_swappc_b64', 's[30:31], s[4:5]', 8),
               ins('s_endpgm', '', 12)]
        f, = run(seq)[0]
        self.assertEqual((f['rule'], f['counter'], f['pc']), ('MISSING_WAIT', 'km', 8))
        seq.insert(1, ins('s_wait_kmcnt', '0x0', 8))
        seq[1]['raw'] = '00000000'
        seq[2] = ins('s_swappc_b64', 's[30:31], s[4:5]', 12)
        seq[3] = ins('s_endpgm', '', 16)
        self.assertEqual(run(seq)[0], [])

    def test_call_does_not_wait_for_vector_arguments(self):
        # A call waits for nothing but SMEM writing its operands; the callee waits at its entry.
        seq = [ins('global_load_b32', 'v1, v0, s[0:1]', 0, 8), ins('s_swappc_b64', 's[30:31], s[4:5]', 8),
               ins('s_endpgm', '', 12)]
        self.assertEqual(run(seq)[0], [])


class EndToEnd(unittest.TestCase):
    @needs_minimal_kernel
    def test_check_reports_callee_once_wired(self):
        """Through `radisa check`, after the integrator wires functions.function_inventory into
        check.check_object_bytes and passes callee= to waits.analyze (see integration notes)."""
        code, rep = check_text((CHECK_FIX / 'callee_missing_waits.s').read_text())
        names = [k['name'] for k in rep['kernels']]
        if 'helper' not in names:
            self.skipTest('radisa.functions is not wired into radisa.check yet')
        rules_ = {f['rule'] for f in all_findings(rep)}
        self.assertIn('CALLEE_RETURN_OUTSTANDING', rules_)
        self.assertIn('MISSING_WAIT', rules_)
        self.assertNotIn('CFG_FALLTHROUGH', {f['rule'] for f in all_findings(rep) if f['severity'] == 'error'})
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()
