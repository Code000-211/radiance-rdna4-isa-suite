"""Non-kernel functions (radisa/functions.py) and the call/callee conventions of radisa/waits.py,
both as LLVM's SIInsertWaitcnts implements them (see waits.CALLS)."""
import io
import re
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from support import (CHECK_FIX, llvm_mc, needs_llvm_mc, needs_minimal_kernel, kernel_asm, check_text,
                     all_findings, run_check, lit24_objdump)

from radisa import binary, check, decode, functions, isadb, rules, sgpr, waits

TABLE = isadb.load(None)


def assemble(text: str, out: Path) -> binary.ELF:
    src = out.with_suffix('.s')
    src.write_text(text)
    subprocess.run([llvm_mc(), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-filetype=obj', str(src),
                    '-o', str(out)], check=True, capture_output=True)
    return binary.ELF(out.read_bytes())


def lint(text: str, cfg_issues: bool = False) -> dict:
    """{name: (inventory entry, findings, stats)} for every kernel and function of the assembled text
    (functions.object_inventory, so also for an object without kernels), the way check_object_bytes
    runs them. With `cfg_issues`, the findings include those of rules.cfg."""
    with tempfile.TemporaryDirectory() as d:
        obj = Path(d, 'a.o')
        elf = assemble(text, obj)
        ks, fs, issues = functions.object_inventory(elf)
        dec = decode.decode_object(obj, [k for k in ks + fs if 'code_hex' in k], decode.select_backend(None, None))
    out = {}
    for k in ks + fs:
        ins = dec[k['symbol']]['instructions']
        edges, reachable, cf = rules.cfg(ins)
        found, stats = waits.analyze(ins, edges, reachable, TABLE, k['metadata']['.wavefront_size'],
                                     callee=bool(k.get('callee')))
        out[k['name']] = (k, (cf + found) if cfg_issues else found, stats)
    return out


def encoding(line: str) -> bytes:
    """The bytes llvm-mc encodes one gfx1201 instruction to."""
    p = subprocess.run([llvm_mc(), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-show-encoding'],
                       input=line + '\n', capture_output=True, text=True, check=True)
    return bytes(int(b, 16) for b in re.search(r'encoding: \[([^\]]*)\]', p.stdout)[1].split(','))


def wave64(text: str) -> str:
    """kernel_asm() text with a wave64 kernel (descriptor and metadata)."""
    return (text.replace('.amdhsa_wavefront_size32 1', '.amdhsa_wavefront_size32 0')
            .replace('.wavefront_size: 32', '.wavefront_size: 64'))


PROLOGUE = ('s_wait_loadcnt_dscnt', 's_wait_expcnt', 's_wait_samplecnt', 's_wait_bvhcnt', 's_wait_kmcnt')


def rule(found, name):
    return [f for f in found if f['rule'] == name]


def ins(op, args, pc, size=4, raw='00000000'):
    return {'pc': pc, 'size': size, 'raw': raw, 'op': op, 'args': args, 'text': f'{op} {args}'}


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
        self.assertEqual(bytes.fromhex(f['code_hex']), encoding('s_setpc_b64 s[30:31]'))
        self.assertEqual(f['address'], ks[0]['address'] + ks[0]['size'])
        self.assertTrue(any('nosize' in m for m in issues))

    @needs_minimal_kernel
    def test_unsized_function_is_reported_unchecked(self):
        text = kernel_asm('s_endpgm', extra_functions='\t.type\tnosize,@function\nnosize:\n\ts_setpc_b64 s[30:31]\n')
        with tempfile.TemporaryDirectory() as d:
            elf = assemble(text, Path(d, 'a.o'))
            ks, _ = binary.kernel_inventory(elf)
            issues, unchecked = [], []
            self.assertEqual(functions.function_inventory(elf, ks, issues, unchecked=unchecked), [])
            self.assertEqual(unchecked, ['nosize'])
            self.assertEqual(functions.object_inventory(elf, unchecked=(u := []))[1], [])
            self.assertEqual(u, ['nosize'])

    @needs_minimal_kernel
    def test_wave64_kernel_gives_wave64_functions(self):
        text = wave64(kernel_asm('s_endpgm', extra_functions=helper('f1', 's_setpc_b64 s[30:31]')))
        with tempfile.TemporaryDirectory() as d:
            ks, fs, _ = functions.object_inventory(assemble(text, Path(d, 'a.o')))
        self.assertEqual(ks[0]['metadata']['.wavefront_size'], 64)
        f, = fs
        self.assertEqual(f['metadata']['.wavefront_size'], 64)
        self.assertEqual(f['issues'], ['relocatable ELF: entry relocation and load mapping are unresolved'])

    def test_wave_size_sources(self):
        k = lambda w: {'metadata': {'.wavefront_size': w}}
        self.assertEqual(functions._wave([k(64), k(64)]), (64, None))
        self.assertEqual(functions._wave([k(64)], 32), (64, None))     # a kernel's word wins
        w, why = functions._wave([])
        self.assertEqual(w, 32)
        self.assertIn('no kernel', why)
        self.assertEqual(functions._wave([], 64), (64, None))
        w, why = functions._wave([k(32), k(64)])
        self.assertEqual(w, 32)
        self.assertIn('disagree', why)
        self.assertIn('[32, 64]', why)

    @needs_llvm_mc
    def test_object_without_kernels(self):
        # amdhsa.kernels: [] (llc output of a -fgpu-rdc translation unit): its function is listed.
        with tempfile.TemporaryDirectory() as d:
            elf = assemble((CHECK_FIX / 'callee_rdc.s').read_text(), Path(d, 'a.o'))
            with self.assertRaisesRegex(binary.FormatError, 'amdhsa.kernels'):
                binary.kernel_inventory(elf)
            ks, fs, issues = functions.object_inventory(elf)
            self.assertTrue(functions.has_functions(elf))
            _, fs64, _ = functions.object_inventory(elf, wave=64)
        self.assertEqual(ks, [])
        f, = fs
        self.assertEqual((f['name'], f['address'], f['size']), ('_Z4leafPKfi', 0, 88))
        self.assertTrue(f['callee'])
        self.assertIn('no kernel in the object declares a wave size; wave32 assumed', f['issues'])
        self.assertEqual(fs64[0]['metadata']['.wavefront_size'], 64)
        self.assertNotIn('no kernel in the object declares a wave size; wave32 assumed', fs64[0]['issues'])

    @needs_llvm_mc
    def test_assembly_without_metadata(self):
        text = '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"\n\t.text\n' + helper('f1', 's_setpc_b64 s[30:31]')
        with tempfile.TemporaryDirectory() as d:
            elf = assemble(text, Path(d, 'a.o'))
            ks, fs, _ = functions.object_inventory(elf, wave=32)
            self.assertTrue(functions.has_functions(elf))
        self.assertEqual(ks, [])
        self.assertEqual([f['name'] for f in fs], ['f1'])

    @needs_llvm_mc
    def test_no_kernels_and_no_functions_still_raises(self):
        text = (CHECK_FIX / 'callee_rdc.s').read_text().replace('@function', '@object')
        with tempfile.TemporaryDirectory() as d:
            elf = assemble(text, Path(d, 'a.o'))
            self.assertFalse(functions.has_functions(elf))
            with self.assertRaisesRegex(binary.FormatError, 'amdhsa.kernels'):
                functions.object_inventory(elf)
        with tempfile.TemporaryDirectory() as d:
            # A snippet: no function symbol at all.
            self.assertFalse(functions.has_functions(assemble('\ts_nop 0\n', Path(d, 'a.o'))))

    def test_relative_pc(self):
        self.assertEqual(functions.relative_pc({'address': 0x30}, 0x38), 8)
        self.assertEqual(functions.relative_pc({'address': 0}, 0x38), 0x38)
        self.assertEqual(functions.relative_pc({}, 4), 4)

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
        mw, km = rule(hf, 'MISSING_WAIT')
        # global_load_b32 reads s[0:1], which a scalar load of the caller may still be writing.
        self.assertEqual((km['counter'], km['registers'], km['required_wait']), ('km', ['s0', 's1'], 's_wait_kmcnt 0'))
        self.assertEqual(mw['counter'], 'load')
        self.assertNotIn('kmcnt', mw['required_wait'])
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

    def test_entry_state_per_register_file(self):
        # A scalar load of the caller writes SGPRs (KMcnt), a vector/LDS/sample/BVH load or an
        # export VGPRs: after s_wait_kmcnt 0 an SGPR is safe however much vector memory is
        # outstanding, and the other way round (RDNA4 ISA 5.7.1).
        sgpr_first = [ins('s_wait_kmcnt', '0x0', 0), ins('s_add_co_u32', 's4, s4, 1', 4),
                      ins('s_wait_loadcnt_dscnt', '0x0', 8), ins('s_wait_expcnt', '0x0', 12),
                      ins('s_wait_samplecnt', '0x0', 16), ins('s_wait_bvhcnt', '0x0', 20),
                      ins('v_mov_b32_e32', 'v0, s4', 24), ins('s_setpc_b64', 's[30:31]', 28)]
        self.assertEqual(rule(run(sgpr_first, callee=True)[0], 'MISSING_WAIT'), [])
        # The VGPR read before the vector waits: blamed on the vector counters only.
        vgpr_early = [ins('s_wait_kmcnt', '0x0', 0), ins('v_add_f32_e32', 'v1, v2, v2', 4),
                      ins('s_wait_loadcnt_dscnt', '0x0', 8), ins('s_wait_expcnt', '0x0', 12),
                      ins('s_wait_samplecnt', '0x0', 16), ins('s_wait_bvhcnt', '0x0', 20),
                      ins('s_setpc_b64', 's[30:31]', 24)]
        f, = rule(run(vgpr_early, callee=True)[0], 'MISSING_WAIT')
        self.assertEqual((f['pc'], f['producer_kind'], f['registers']), (4, 'caller', ['v1', 'v2']))
        self.assertNotEqual(f['counter'], 'km')
        self.assertEqual(f['required_wait'], 's_wait_loadcnt 0, s_wait_dscnt 0, s_wait_expcnt 0, '
                                             's_wait_samplecnt 0, s_wait_bvhcnt 0')
        # The SGPR read before s_wait_kmcnt 0, after every vector wait: blamed on KMcnt only.
        sgpr_early = [ins('s_wait_loadcnt_dscnt', '0x0', 0), ins('s_wait_expcnt', '0x0', 4),
                      ins('s_wait_samplecnt', '0x0', 8), ins('s_wait_bvhcnt', '0x0', 12),
                      ins('s_add_co_u32', 's4, s4, 1', 16), ins('s_wait_kmcnt', '0x0', 20),
                      ins('s_setpc_b64', 's[30:31]', 24)]
        f, = rule(run(sgpr_early, callee=True)[0], 'MISSING_WAIT')
        self.assertEqual((f['pc'], f['counter'], f['registers'], f['required_wait']),
                         (16, 'km', ['s4'], 's_wait_kmcnt 0'))
        self.assertIn('a memory operation of the caller', f['message'])

    def test_entry_registers_by_counter(self):
        seq = [ins('v_add_f32_e32', 'v1, v2, s4', 0), ins('s_setpc_b64', 's[30:31]', 4)]
        regs = waits.callee_entry_registers(seq, waits.operand_models(seq, TABLE))
        self.assertEqual(set(regs), set(waits.CALLEE_ENTRY))
        self.assertEqual(regs['km'], {('s', 4)})     # not exec (implicit read), never s[30:31]
        for d in ('load', 'ds', 'exp', 'sample', 'bvh'):
            self.assertEqual(regs[d], {('v', 1), ('v', 2)}, d)
        self.assertEqual(regs['store'], frozenset())

    @needs_llvm_mc
    def test_function_only_object_is_linted(self):
        # A code object with amdhsa.kernels: [] (issue: it was NO_KERNELS and never linted).
        (_, found, _), = lint((CHECK_FIX / 'callee_rdc_bug.s').read_text()).values()
        fmac = [f for f in rule(found, 'MISSING_WAIT') if f['producer'] is not None]
        self.assertEqual({tuple(f['registers']) for f in fmac}, {('v2',), ('v0',)})
        self.assertEqual({f['pc'] for f in fmac}, {0x38})
        ret, = rule(found, 'CALLEE_RETURN_OUTSTANDING')
        self.assertIn('load', ret['pending'])

    @needs_llvm_mc
    def test_compiled_function_only_object_is_clean(self):
        # Unedited llc 18 output. MISSING_ALU_WAIT is the documented LLVM<20 gap (no s_wait_alu).
        (k, found, _), = lint((CHECK_FIX / 'callee_rdc.s').read_text(), cfg_issues=True).values()
        self.assertTrue(k['callee'])
        self.assertEqual({f['rule'] for f in found} - {'MISSING_ALU_WAIT', 'CFG_INDIRECT'}, set())
        # The s_code_end padding after the function is outside its symbol size, so never reached.
        self.assertNotIn('REACHABLE_CODE_END', {f['rule'] for f in found})

    @needs_llvm_mc
    def test_noreturn_call_and_trap_do_not_fall_through(self):
        # clang 18 -O2: kernel kn ends with the call to noreturn die(); die ends with s_trap 2.
        r = lint((CHECK_FIX / 'callee_noreturn.s').read_text(), cfg_issues=True)
        self.assertEqual(set(r), {'kn', '_Z3diePi', '_Z3chkPii'})
        ft = {n: rule(found, 'CFG_FALLTHROUGH') for n, (_, found, _) in r.items()}
        # rules.cfg: a final s_swappc_b64 and s_trap 2 have no fall-through successor.
        self.assertEqual(ft, {'kn': [], '_Z3diePi': [], '_Z3chkPii': []})
        for name, (_, found, _) in r.items():
            self.assertEqual(rule(found, 'MISSING_WAIT') + rule(found, 'CALLEE_RETURN_OUTSTANDING'), [], name)

    def test_setpc_does_not_fall_through(self):
        # s_setpc_b64 returns or jumps; the S_CODE_END padding after a function is not reached.
        seq = [ins('s_setpc_b64', 's[30:31]', 0), ins('s_code_end', '', 4)]
        _, reachable, issues = rules.cfg(seq)
        self.assertNotIn(4, reachable)
        self.assertNotIn('REACHABLE_CODE_END', {f['rule'] for f in issues})

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


def straight_kernel(n: int, every: int = 5) -> list[dict]:
    """n instructions, every `every`-th a global_store_b32 that is never waited for (normal: stores
    are not waited for before s_endpgm), the rest VALU; then s_endpgm."""
    seq, pc = [], 0
    for k in range(n):
        if k % every == 0:
            seq.append(ins('global_store_b32', f'v0, v{1 + k % 200}, s[0:1]', pc, 8))
        else:
            seq.append(ins('v_add_f32_e32', f'v{1 + k % 200}, v{1 + (k + 1) % 200}, v{1 + (k + 2) % 200}', pc))
        pc += seq[-1]['size']
    return seq + [ins('s_endpgm', '', pc)]


class Fixpoint(unittest.TestCase):
    def test_many_unwaited_stores_stay_near_linear(self):
        # 50k instructions, 10k stores: the old per-producer copy took ~77 s here.
        seq = straight_kernel(50000)
        edges, reachable, _ = rules.cfg(seq)
        t = time.perf_counter()
        found, stats = waits.analyze(seq, edges, reachable, TABLE, 32)
        self.assertLess(time.perf_counter() - t, 30)
        self.assertEqual((found, stats['fixpoint']), ([], 'converged'))
        # Every store is still there: the 64 youngest with their age, the older ones at CAP.
        st = waits.solve(seq, edges, TABLE)[0][seq[-1]['pc']]
        self.assertEqual(len(st['store']), 10000)
        self.assertEqual(sorted(a for _, a in st['store'].values())[:3], [0, 1, 2])
        self.assertEqual(st['store'][seq[0]['pc']], (frozenset(), waits.CAP))

    def test_loop_back_to_the_entry_keeps_the_entry_state(self):
        # waits.solve gives an instruction with one predecessor that predecessor's out-state without a
        # join; the entry has the caller's state as one more predecessor, so a loop that branches
        # back to the first instruction (after draining every counter) must not replace it.
        drain = [ins(op, '0x0', 4 + 4 * k) for k, op in enumerate(('s_wait_loadcnt', 's_wait_dscnt', 's_wait_kmcnt',
                                                                      's_wait_samplecnt', 's_wait_bvhcnt',
                                                                      's_wait_expcnt'))]
        back = (-(len(drain) + 2)) & 0xffff
        seq = [ins('v_add_f32', 'v2, v1, v1', 0), *drain,
               ins('s_cbranch_scc1', '', 28, raw=f'{back & 0xff:02x}{back >> 8:02x}0000'), ins('s_setpc_b64', 's[30:31]', 32)]
        self.assertEqual(rules.cfg(seq)[0][28], [0, 32])
        found, _ = run(seq, callee=True)
        self.assertEqual([(f['rule'], f['pc']) for f in found], [('MISSING_WAIT', 0)])

    def test_saturated_producer_is_still_reported(self):
        # A load with CAP+ younger loads is still outstanding until a wait: its use is a MISSING_WAIT.
        seq = [ins('global_load_b32', 'v1, v0, s[0:1]', 0, 8)]
        pc = 8
        for k in range(waits.CAP + 6):
            seq.append(ins('global_load_b32', f'v{2 + k % 50}, v0, s[0:1]', pc, 8))
            pc += 8
        seq += [ins('v_add_f32_e32', 'v60, v1, v1', pc), ins('s_endpgm', '', pc + 4)]
        found, _ = run(seq)
        f, = [f for f in found if f['producer'] == 0]
        self.assertEqual((f['pc'], f['required_wait']), (pc, f's_wait_loadcnt {waits.CAP}'))
        # Any S_WAIT_LOADCNT (at most 63 < CAP) drops it.
        w = ins('s_wait_loadcnt', '0x3f', pc, raw='3f000000')
        seq2 = seq[:-2] + [w, ins('v_add_f32_e32', 'v60, v1, v1', pc + 4), ins('s_endpgm', '', pc + 8)]
        self.assertEqual([f for f in run(seq2)[0] if f['producer'] == 0], [])

    def test_loop_reaches_the_same_fixpoint(self):
        # A loop whose back edge carries more than CAP stores joins saturated and young producers.
        body = [ins('global_store_b32', f'v0, v{k % 9 + 1}, s[0:1]', 4 + 8 * k, 8) for k in range(70)]
        end = 4 + 8 * 70
        br = ins('s_cbranch_scc1', '', end, raw=((-(end + 4 - 4) // 4) & 0xffff).to_bytes(2, 'little').hex() + '0000')
        seq = [ins('s_mov_b32', 's2, 0', 0)] + body + [br, ins('global_load_b32', 'v1, v0, s[0:1]', end + 4, 8),
                                                       ins('v_add_f32_e32', 'v3, v1, v1', end + 12),
                                                       ins('s_endpgm', '', end + 16)]
        edges, reachable, _ = rules.cfg(seq)
        self.assertIn(4, edges[end])
        state_in, _, status = waits.solve(seq, edges, TABLE)
        self.assertEqual(status, 'converged')
        head = state_in[4]['store']
        self.assertEqual(len(head), 70)                          # every store, via the back edge
        self.assertEqual(head[4 + 8 * 69], (frozenset(), 0))     # the youngest at the loop head
        self.assertEqual(head[4], (frozenset(), waits.CAP))                # 69 younger, capped
        self.assertEqual(head[4 + 8 * 6][1], waits.CAP - 1)
        f, = run(seq)[0]
        self.assertEqual((f['rule'], f['pc'], f['producer']), ('MISSING_WAIT', end + 12, end + 4))

    def test_no_state_outside_the_models(self):
        seq = [ins('global_load_b32', 'v1, v0, s[0:1]', 0, 8), ins('s_endpgm', '', 8)]
        other = [ins('ds_load_b32', 'v1, v0', 0, 8), ins('s_endpgm', '', 8)]
        e1, e2 = rules.cfg(seq)[0], rules.cfg(other)[0]
        m1, m2 = waits.operand_models(seq, TABLE), waits.operand_models(other, TABLE)
        a = waits.solve(seq, e1, TABLE, 32, m1)
        b = waits.solve(other, e2, TABLE, 32, m2)
        self.assertIs(waits.solve(seq, e1, TABLE, 32, m1), a)    # kept with its own models
        self.assertIs(waits.solve(other, e2, TABLE, 32, m2), b)
        self.assertEqual(set(b[0][8]), {'ds'})
        self.assertFalse([n for n, v in vars(waits).items() if isinstance(v, tuple) and len(v) == 5])
        self.assertFalse(hasattr(waits, '_SOLVED'))
        # A plain dict of models is solved every time and keeps nothing.
        plain = dict(m1)
        self.assertIsNot(waits.solve(seq, e1, TABLE, 32, plain), waits.solve(seq, e1, TABLE, 32, plain))


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


    @needs_minimal_kernel
    def test_kernel_filter_selects_functions_too(self):
        text = (CHECK_FIX / 'callee_missing_waits.s').read_text()
        backend = decode.select_backend(None, None)
        with tempfile.TemporaryDirectory() as d:
            data = assemble(text, Path(d, 'a.o')).data
            only_h, _ = check.check_object_bytes(data, 'a.o', TABLE, backend, 'helper', Path(d))
            only_k, _ = check.check_object_bytes(data, 'a.o', TABLE, backend, 'k', Path(d))
            both, _ = check.check_object_bytes(data, 'a.o', TABLE, backend, None, Path(d))
        self.assertEqual([r['name'] for r in only_h], ['helper'])
        self.assertTrue(only_h[0]['callee'])
        self.assertEqual([r['name'] for r in only_k], ['k'])
        calls, = [f for f in only_k[0]['findings'] if f['rule'] == 'CALLS']
        self.assertIn('not checked in this run (--kernel)', calls['message'])
        self.assertEqual(calls['callees_checked'], [])
        self.assertEqual(sorted(r['name'] for r in both), ['helper', 'k'])
        calls, = [f for f in both[[r['name'] for r in both].index('k')]['findings'] if f['rule'] == 'CALLS']
        self.assertEqual(calls['callees_checked'], ['helper'])

    # ---- check.py wiring: function-only .s / objects (functions.has_functions, object_inventory),
    # offsets inside a function (functions.relative_pc), the CALLS note for unsized functions.

    @needs_llvm_mc
    def test_function_only_assembly_is_checked_as_callee(self):
        code, rep = check_text((CHECK_FIX / 'callee_rdc_bug.s').read_text())
        k, = rep['kernels']
        self.assertEqual(k['name'], '_Z4leafPKfi')
        self.assertTrue(k['callee'])
        rules_ = {f['rule'] for f in all_findings(rep)}
        self.assertIn('MISSING_WAIT', rules_)
        self.assertIn('CALLEE_RETURN_OUTSTANDING', rules_)
        self.assertNotIn('REACHABLE_CODE_END', rules_)
        self.assertEqual(code, 1)

    @needs_llvm_mc
    def test_compiled_function_only_assembly_has_no_code_end_error(self):
        _, rep = check_text((CHECK_FIX / 'callee_rdc.s').read_text())
        errors = {f['rule'] for f in all_findings(rep) if f['severity'] == 'error'}
        self.assertNotIn('REACHABLE_CODE_END', errors)
        self.assertEqual([k['name'] for k in rep['kernels']], ['_Z4leafPKfi'])
        self.assertEqual(errors - {'MISSING_ALU_WAIT'}, set())     # the LLVM<20 s_wait_alu gap only

    @needs_llvm_mc
    def test_object_without_kernels_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            obj = Path(d, 'rdc_bug.o')
            assemble((CHECK_FIX / 'callee_rdc_bug.s').read_text(), obj)
            code, rep = run_check(obj)
        self.assertEqual([k['name'] for k in rep['kernels']], ['_Z4leafPKfi'])
        self.assertIn('MISSING_WAIT', {f['rule'] for f in all_findings(rep)})
        self.assertEqual((rep['status'], code), ('FAIL', 1))

    @needs_minimal_kernel
    def test_text_report_gives_offsets_inside_the_function(self):
        # An object has no source lines, so the text report locates findings as name+offset.
        from radisa.cli import main
        with tempfile.TemporaryDirectory() as d:
            obj = Path(d, 'cmw.o')
            assemble((CHECK_FIX / 'callee_missing_waits.s').read_text(), obj)
            with redirect_stdout(io.StringIO()) as out:
                main(['check', str(obj), '--json', str(Path(d, 'r.json'))])
            rep = __import__('json').loads(Path(d, 'r.json').read_text())
        text = out.getvalue()
        h, = [k for k in rep['kernels'] if k['name'] == 'helper']
        self.assertGreater(h['address'], 0)                    # helper follows kernel k
        ret, = [f for f in h['findings'] if f['rule'] == 'CALLEE_RETURN_OUTSTANDING']
        self.assertNotIn(f"helper+{ret['pc']:#x}:", text)
        # helper: v_add_f32 (+0x0), global_load_b32 (+0x4, 12 bytes on gfx12), s_setpc_b64 (+0x10).
        self.assertEqual(ret['pc'] - h['address'], 0x10)
        self.assertIn('cmw.o:helper+0x0: error: [MISSING_WAIT]', text)
        self.assertIn('cmw.o:helper+0x4: error: [MISSING_WAIT]', text)
        self.assertIn('cmw.o:helper+0x10: warning: [CALLEE_RETURN_OUTSTANDING]', text)

    @needs_minimal_kernel
    def test_calls_note_names_an_unsized_function(self):
        text = (CHECK_FIX / 'callee_missing_waits.s').read_text().replace(
            '\t.size\thelper, .Lfunc_end_helper-helper\n', '')
        _, rep = check_text(text)
        calls, = [f for f in all_findings(rep) if f['rule'] == 'CALLS']
        self.assertNotIn('holds no function body', calls['message'])
        self.assertIn('helper', calls['message'])
        self.assertIn('NOT checked', calls['message'])
        self.assertIn('OBJECT_EVIDENCE', {f['rule'] for f in all_findings(rep)})


class NoreturnCall(unittest.TestCase):
    """NORETURN_CALL_RETURNS: a call that ends the code is taken as noreturn (rules.cfg); where its
    callee is a function of the same object with a reachable return, that is reported ('review')."""

    def found(self, rep):
        return [(f['rule'], f['severity'], f.get('callee')) for f in all_findings(rep)
                if f['rule'] == 'NORETURN_CALL_RETURNS']

    @needs_minimal_kernel
    def test_final_call_to_returning_callee(self):
        _, rep = check_text((CHECK_FIX / 'callee_noreturn_returns.s').read_text())
        self.assertEqual(self.found(rep), [('NORETURN_CALL_RETURNS', 'review', 'helper')])
        f, = [f for f in all_findings(rep) if f['rule'] == 'NORETURN_CALL_RETURNS']
        self.assertEqual(f['loc']['line'], 15)                       # the s_swappc_b64 line of the .s
        self.assertIn('helper returns (s_setpc_b64 s[30:31] at +0x10)', f['message'])
        calls, = [f for f in all_findings(rep) if f['rule'] == 'CALLS']
        self.assertIn('noreturn', calls['message'])
        self.assertEqual(calls['noreturn_call_pcs'], [f['pc']])

    @needs_minimal_kernel
    def test_object_and_linked_object(self):
        # Relocatable object: the R_AMDGPU_REL32_LO relocation names the callee (a section symbol +
        # addend here, `helper` is local); linked: the s_add literals hold the offset.
        import shutil
        with tempfile.TemporaryDirectory() as d:
            obj = Path(d, 'nr.o')
            assemble((CHECK_FIX / 'callee_noreturn_returns.s').read_text(), obj)
            _, rep = run_check(obj)
            self.assertEqual(self.found(rep), [('NORETURN_CALL_RETURNS', 'review', 'helper')])
            lld = shutil.which('ld.lld')
            if lld:
                so = Path(d, 'nr.so')
                subprocess.run([lld, '-shared', str(obj), '-o', str(so)], check=True, capture_output=True)
                _, rep = run_check(so)
                self.assertEqual(self.found(rep), [('NORETURN_CALL_RETURNS', 'review', 'helper')])

    @needs_llvm_mc
    def test_noreturn_callee_is_clean(self):
        # clang 18 -O2: kn ends with its call to die(), which ends with s_trap 2 (no return).
        _, rep = check_text((CHECK_FIX / 'callee_noreturn.s').read_text())
        self.assertEqual(self.found(rep), [])
        self.assertTrue(any('noreturn' in f['message'] for f in all_findings(rep) if f['rule'] == 'CALLS'))

    def test_call_target_shapes(self):
        def seq_(*xs):
            out, pc = [], 0
            for op, args, size in xs:
                out.append({'op': op, 'args': args, 'pc': pc, 'size': size, 'text': f'{op} {args}'})
                pc += size
            return out
        # Linked: after s_getpc_b64 (pc 4) + 0x100 (with the gfx12 s_sext_i32_i16 on the high half).
        x = seq_(('s_getpc_b64', 's[2:3]', 4), ('s_sext_i32_i16', 's3, s3', 4), ('s_add_co_u32', 's2, s2, 0x100', 8),
                 ('s_add_co_ci_u32', 's3, s3, 0', 8), ('s_swappc_b64', 's[30:31], s[2:3]', 4))
        self.assertEqual(functions.call_target(x, 4, {}, 1), (0x104, None))
        # Relocatable: S + A - (P - address after s_getpc_b64), P = the s_add literal at 8 + 4.
        self.assertEqual(functions.call_target(x, 4, {(1, 12): (functions.R_AMDGPU_REL32_LO, 0x40, 1, 12)}, 1),
                         (0x40 + 12 - 8, 1))
        # Another pair, a missing s_getpc_b64, or a write of the low half in between: unresolved.
        self.assertIsNone(functions.call_target(x[:4] + seq_(('s_swappc_b64', 's[30:31], s[4:5]', 4)), 4, {}, 1))
        self.assertIsNone(functions.call_target(x[1:], 3, {}, 1))
        y = x[:1] + seq_(('s_sext_i32_i16', 's2, s2', 4)) + x[2:]
        self.assertIsNone(functions.call_target(y, 4, {}, 1))
        # ROCm LLVM 24 prints inline-valued literals as lit(0x..): same target, also backwards.
        z = [dict(i) for i in x]
        z[2]['args'], z[3]['args'] = 's2, s2, lit(0x14)', 's3, s3, lit(0x0)'
        self.assertEqual(functions.call_target(z, 4, {}, 1), (0x18, None))
        z[2]['args'], z[3]['args'] = 's2, s2, lit(0xfffffff0)', 's3, s3, lit(0xffffffff)'
        self.assertEqual(functions.call_target(z, 4, {}, 1), (4 - 16, None))
        z[2]['args'] = 's2, s2, lit(callee)'
        self.assertIsNone(functions.call_target(z, 4, {}, 1))

    @needs_minimal_kernel
    def test_linked_object_llvm24_disassembly(self):
        # The linked object as ROCm LLVM 24 disassembles it (`s_add_co_u32 s4, s4, lit(0x14)`): the
        # target run lost NORETURN_CALL_RETURNS here (int('lit(0x14)', 0) fails).
        import shutil
        lld = shutil.which('ld.lld')
        with tempfile.TemporaryDirectory() as d:
            exe = lit24_objdump(d)
            if not lld or exe is None:
                self.skipTest('needs ld.lld and llvm-objdump')
            obj, so = Path(d, 'nr.o'), Path(d, 'nr.so')
            assemble((CHECK_FIX / 'callee_noreturn_returns.s').read_text(), obj)
            subprocess.run([lld, '-shared', str(obj), '-o', str(so)], check=True, capture_output=True)
            self.assertRegex(subprocess.run([str(exe), '-d', '--mcpu=gfx1201', str(so)], capture_output=True,
                                            text=True).stdout, r's_add_co_u32 s\d+, s\d+, lit\(0x')
            _, rep = run_check(so, '--objdump', exe)
        self.assertEqual(self.found(rep), [('NORETURN_CALL_RETURNS', 'review', 'helper')])


if __name__ == '__main__':
    unittest.main()
