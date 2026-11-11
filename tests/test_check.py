"""`radisa check`: AMD XML operand table, counter model, HIP/assembly front ends."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import tempfile
import unittest

from radisa import hipsrc, incremental, isadb, vopd, waits
from radisa.cli import main
from radisa.rules import cfg
from support import ROOT, instruction, seq, vopd_bytes

FIX = ROOT / 'fixtures' / 'check'
TABLE = isadb.load()


def names(model):
    return sorted(f'{a}{b}' for a, b in model)


def model(op, args, wave=32):
    return isadb.operand_model(TABLE, op, args, wave)


class TableTests(unittest.TestCase):
    def test_provenance(self):
        src = TABLE['source']
        self.assertEqual(src['architecture'], 'AMD RDNA 4')
        self.assertEqual(src['xml_license'], 'MIT')
        self.assertRegex(src['xml_sha256'], '^[0-9a-f]{64}$')
        self.assertEqual(len(TABLE['instructions']), 1264)

    def test_vopd_opcodes_agree_with_xml(self):
        xml = {}
        for name, e in TABLE['instructions'].items():
            for f in e['forms']:
                if f['enc'].startswith('VOPDXY'):
                    xml[f['opcode']] = name
        self.assertEqual(xml, {k: 'V_DUAL_' + v.upper() for k, v in vopd.OPS.items()})

    def test_manual_citations(self):
        self.assertIn('PDF p.62', isadb.cite(TABLE, '5.7.1'))
        self.assertEqual(TABLE['instructions']['S_WAIT_LOADCNT_DSCNT']['pdf_page'], 288)

    def test_implicit_exec_and_scc(self):
        m = model('s_and_saveexec_b32', 's0, vcc_lo')
        self.assertTrue(m['known'])
        self.assertEqual(names(m['reads']), ['s106', 's126'])
        self.assertEqual(names(m['writes']), ['s0', 's126', 'scc0'])

    def test_wave64_exec_pair(self):
        m = model('v_mov_b32_e32', 'v1, v2', wave=64)
        self.assertEqual(names(m['reads']), ['s126', 's127', 'v2'])

    def test_vcmpx_exec_destination_not_printed(self):
        m = model('v_cmpx_gt_u32_e32', '0x80, v0')
        self.assertTrue(m['known'])
        self.assertEqual(names(m['writes']), ['s126'])

    def test_carry_in_and_out(self):
        m = model('v_add_co_ci_u32_e32', 'v11, vcc_lo, 0, v11, vcc_lo')
        self.assertEqual(names(m['reads']), ['s106', 's126', 'v11'])
        self.assertEqual(names(m['writes']), ['s106', 'v11'])

    def test_atomic_return_selects_vdst(self):
        rtn = model('global_atomic_add_u32', 'v2, v2, v4, s[8:9] th:TH_ATOMIC_RETURN')
        self.assertEqual(names(rtn['writes']), ['v2'])
        self.assertTrue(rtn['returns'])
        none = model('global_atomic_add_u32', 'v2, v4, s[8:9]')
        self.assertTrue(none['known'])
        self.assertEqual(names(none['writes']), [])
        self.assertEqual(waits.domains(none, 'global_atomic_add_u32'), ['store'])
        self.assertEqual(waits.domains(rtn, 'global_atomic_add_u32'), ['load'])

    def test_memory_domains(self):
        self.assertEqual(waits.domains(model('flat_load_b32', 'v1, v[2:3]'), ''), ['load', 'ds'])
        self.assertEqual(waits.domains(model('ds_load_b32', 'v1, v3 offset:256'), ''), ['ds'])
        self.assertEqual(waits.domains(model('s_load_b32', 's0, s[0:1], 0x20'), ''), ['km'])
        self.assertEqual(waits.domains(model('global_store_b32', 'v1, v2, s[4:5]'), ''), ['store'])

    def test_negated_register_range_last(self):
        m = model('v_add_f64_e64', 'v[0:1], v[2:3], -v[4:5]')
        self.assertTrue(m['known'])
        self.assertEqual(names(m['reads']), ['s126', 'v2', 'v3', 'v4', 'v5'])

    def test_unknown_mnemonic_is_not_harmless(self):
        m = model('v_made_up_b32', 'v1, v2')
        self.assertFalse(m['known'])
        self.assertEqual(names(m['reads']), ['v1', 'v2'])


def run_waits(*spec):
    ins = seq(*spec)
    edges, reachable, _ = cfg(ins)
    return waits.analyze(ins, edges, reachable, TABLE)[0]


class CounterModelTests(unittest.TestCase):
    def test_use_without_wait(self):
        f = run_waits(('global_load_b32', 'v0, v[2:3], off'), ('v_add_f32', 'v1, v0, v2'), ('s_endpgm',))
        self.assertEqual([x['rule'] for x in f], ['MISSING_WAIT'])
        self.assertEqual(f[0]['required_wait'], 's_wait_loadcnt 0')

    def test_wait_zero(self):
        self.assertEqual(run_waits(('global_load_b32', 'v0, v[2:3], off'), ('s_wait_loadcnt', '0x0', 0),
                                   ('v_add_f32', 'v1, v0, v2'), ('s_endpgm',)), [])

    def test_nonzero_wait_in_order(self):
        ok = run_waits(('global_load_b32', 'v0, v[2:3], off'), ('global_load_b32', 'v5, v[2:3], off'),
                       ('s_wait_loadcnt', '0x1', 1), ('v_add_f32', 'v1, v0, v2'), ('s_endpgm',))
        self.assertEqual(ok, [])
        bad = run_waits(('global_load_b32', 'v0, v[2:3], off'), ('global_load_b32', 'v5, v[2:3], off'),
                        ('s_wait_loadcnt', '0x1', 1), ('v_add_f32', 'v1, v5, v2'), ('s_endpgm',))
        self.assertEqual(len(bad), 1)

    def test_kmcnt_out_of_order(self):
        f = run_waits(('s_load_b32', 's4, s[0:1], 0x0'), ('s_load_b32', 's5, s[0:1], 0x4'),
                      ('s_wait_kmcnt', '0x1', 1), ('s_add_co_u32', 's6, s4, 1'), ('s_endpgm',))
        self.assertEqual(len(f), 1)

    def test_combined_wait_fields(self):
        # SIMM16[13:8] = LOADcnt, SIMM16[5:0] = DScnt.
        base = [('global_load_b32', 'v0, v[2:3], off'), ('ds_load_b32', 'v5, v3')]
        self.assertEqual(run_waits(*base, ('s_wait_loadcnt_dscnt', '0x0', 0x0000),
                                   ('v_add_f32', 'v1, v0, v5'), ('s_endpgm',)), [])
        f = run_waits(*base, ('s_wait_loadcnt_dscnt', '0x100', 0x0100), ('v_add_f32', 'v1, v0, v5'),
                      ('s_endpgm',))
        self.assertEqual([x['counter'] for x in f], ['load'])

    def test_flat_counts_in_dscnt_too(self):
        f = run_waits(('flat_load_b32', 'v0, v[2:3]'), ('s_wait_loadcnt', '0x0', 0),
                      ('v_add_f32', 'v1, v0, v2'), ('s_endpgm',))
        self.assertEqual([x['counter'] for x in f], ['ds'])

    def test_overwrite_of_pending_destination(self):
        f = run_waits(('global_load_b32', 'v0, v[2:3], off'), ('v_mov_b32', 'v0, 0'), ('s_endpgm',))
        self.assertEqual(len(f), 1)
        self.assertIn('overwrites', f[0]['message'])

    def test_same_counter_waw_needs_no_wait(self):
        self.assertEqual(run_waits(('global_load_b32', 'v0, v[2:3], off'), ('global_load_b32', 'v0, v[4:5], off'),
                                   ('s_endpgm',)), [])

    def test_loop_back_edge(self):
        # pc0 use, pc4 load, pc8 branch back to pc0 (imm -3 dwords from next pc).
        f = run_waits(('v_add_f32', 'v4, v1, v4'), ('global_load_b32', 'v1, v[2:3], off'),
                      ('s_cbranch_scc1', '', -3), ('s_endpgm',))
        self.assertEqual([x['pc'] for x in f], [0])

    def test_d16_store_ignores_the_other_half(self):
        lo, hi = ('global_load_d16_b16', 'v3, v[4:5], off'), ('global_load_d16_hi_b16', 'v3, v[6:7], off')
        w1 = ('s_wait_loadcnt', '0x1', 1)
        # The low half has landed; the high-half load is outstanding and not read.
        self.assertEqual(run_waits(lo, hi, w1, ('global_store_b16', 'v[8:9], v3, off'), ('s_endpgm',)), [])
        self.assertEqual(run_waits(hi, lo, w1, ('global_store_d16_hi_b16', 'v[8:9], v3, off'), ('s_endpgm',)), [])
        self.assertEqual(run_waits(lo, hi, w1, ('ds_store_b16', 'v8, v3'), ('s_endpgm',)), [])
        # The half that is read is outstanding.
        for bad in (('global_store_d16_hi_b16', 'v[8:9], v3, off'), ('v_add_nc_u32', 'v9, v3, v3'),
                    ('global_store_b32', 'v[8:9], v3, off')):
            self.assertEqual([x['rule'] for x in run_waits(lo, hi, w1, bad, ('s_endpgm',))], ['MISSING_WAIT'], bad)
        # The data register is also the address: it is read whole.
        self.assertEqual(len(run_waits(lo, hi, w1, ('ds_store_b16', 'v3, v3'), ('s_endpgm',))), 1)

    def test_lds_store_outstanding_at_barrier(self):
        st, sig, wt = ('ds_store_b32', 'v1, v2'), ('s_barrier_signal', '-1'), ('s_barrier_wait', '-1')
        f = run_waits(st, sig, wt, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['pc'], x['later_uses']) for x in f], [('LDS_STORE_BEFORE_BARRIER', 4, [8])])
        self.assertEqual(f[0]['required_wait'], 's_wait_dscnt 0')
        self.assertEqual(run_waits(st, ('s_wait_dscnt', '0x0', 0), sig, wt, ('s_endpgm',)), [])
        # A global store or an unused LDS read may stay outstanding across the barrier.
        self.assertEqual(run_waits(('global_store_b32', 'v[4:5], v2, off'), ('ds_load_b32', 'v7, v1'), sig, wt,
                                   ('s_endpgm',)), [])
        # One path of two leaves the write outstanding (pc0 skips the wait at pc8).
        f = run_waits(('s_cbranch_scc1', '', 2), st, ('s_wait_dscnt', '0x0', 0), st, sig, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['producer']) for x in f], [('LDS_STORE_BEFORE_BARRIER', 12)])

    def test_sgpr_read_after_scalar_write(self):
        vread, write, read = ('v_mov_b32', 'v0, s4'), ('s_mov_b32', 's4, 1'), ('s_add_co_u32', 's6, s4, 1')
        f = run_waits(vread, write, read, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['pc'], x['required_wait']) for x in f],
                         [('MISSING_ALU_WAIT', 8, 's_wait_alu depctr_sa_sdst(0)')])
        self.assertEqual(run_waits(vread, write, ('s_wait_alu', 'depctr_sa_sdst(0)', 0xff9e), read, ('s_endpgm',)), [])
        # The wrong field does not help; the other register of the pair is hazardous as well.
        self.assertEqual(len(run_waits(vread, write, ('s_wait_alu', 'depctr_va_vcc(0)', 0xff9d), read, ('s_endpgm',))), 1)
        self.assertEqual(len(run_waits(vread, ('s_mov_b32', 's5, 1'), ('s_add_co_u32', 's6, s5, 1'), ('s_endpgm',))), 1)
        # No vector instruction has read the pair: nothing to wait for.
        self.assertEqual(run_waits(write, read, ('s_endpgm',)), [])
        self.assertEqual(run_waits(('v_mov_b32', 'v0, s8'), write, read, ('s_endpgm',)), [])
        # Four ds_nop make the hardware forget which pairs vector instructions read (wave32).
        nop = ('ds_nop', '')
        self.assertEqual(run_waits(vread, nop, nop, nop, nop, write, read, ('s_endpgm',)), [])
        self.assertEqual(len(run_waits(vread, nop, nop, nop, write, read, ('s_endpgm',))), 1)
        # A scalar memory read commits everything.
        self.assertEqual(run_waits(vread, write, ('s_load_b32', 's9, s[0:1], 0x0'), read, ('s_endpgm',)), [])

    def test_sgpr_read_after_vector_write(self):
        vread, vwrite = ('v_cndmask_b32', 'v0, v1, v2, s8'), ('v_cmp_eq_u32', 's8, v0, v1')
        f = run_waits(vread, vwrite, vread, ('s_endpgm',))
        self.assertEqual([(x['pc'], x['required_wait']) for x in f], [(8, 's_wait_alu depctr_va_sdst(0)')])
        self.assertEqual(run_waits(vread, vwrite, ('s_wait_alu', 'depctr_va_sdst(0)', 0xf19f), vread, ('s_endpgm',)), [])
        # A scalar instruction may read it at once.
        self.assertEqual(run_waits(vread, vwrite, ('s_and_b32', 's9, s8, s8'), ('s_endpgm',)), [])
        vcc_read, vcc_write = ('v_cndmask_b32', 'v0, v1, v2, vcc_lo'), ('v_cmp_eq_u32', 'vcc_lo, v0, v1')
        f = run_waits(vcc_read, vcc_write, vcc_read, ('s_endpgm',))
        self.assertEqual([(x['pc'], x['required_wait']) for x in f], [(8, 's_wait_alu depctr_va_vcc(0)')])
        self.assertEqual(run_waits(vcc_read, vcc_write, ('s_wait_alu', 'depctr_va_vcc(0)', 0xff9d), vcc_read,
                                   ('s_endpgm',)), [])

    def test_vopd_cndmask_reads_vcc(self):
        dual = instruction('v_dual_cndmask_b32', 'v5, v2, v5 :: v_dual_cndmask_b32 v4, v4, v2', pc=8, size=8)
        dual.update(vopd_independent_decode=True, raw=vopd_bytes(x=9, y=9, dx=5, dy=4).hex())
        head = seq(('v_cndmask_b32', 'v0, v1, v2, vcc_lo'), ('v_cmp_gt_i32', 'vcc_lo, v6, v21'))
        tail = [instruction('s_endpgm', pc=16)]
        def run(ins):
            edges, reachable, _ = cfg(ins)
            return waits.analyze(ins, edges, reachable, TABLE)[0]
        self.assertEqual([(x['rule'], x['pc'], x['required_wait']) for x in run(head + [dual] + tail)],
                         [('MISSING_ALU_WAIT', 8, 's_wait_alu depctr_va_vcc(0)')])
        mov = dict(dual, op='v_dual_mov_b32', args='v5, v2 :: v_dual_mov_b32 v4, v3', raw=vopd_bytes(x=8, y=8, dx=5, dy=4).hex())
        self.assertEqual(run(head + [mov] + tail), [])

    def test_ttmp_read_between_vector_write_and_read(self):
        # LLVM numbers ttmp registers as SGPR 0: the scalar read of ttmp9 ends, for the compiler, the
        # pending vector write of s0, and it emits no wait before the read of s0.
        track, write = ('v_mov_b32', 'v0, s0'), ('v_add_co_u32', 'v1, s0, s12, v1')
        read = ('v_add_co_ci_u32', 'v2, null, s13, v2, s0')
        f = run_waits(track, write, ('s_ashr_i32', 's35, ttmp9, 31'), read, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['severity'], x['pc']) for x in f], [('ALU_WAIT_TTMP', 'review', 12)])
        # Without the ttmp read the compiler waits, so a missing wait is an error.
        f = run_waits(track, write, ('s_ashr_i32', 's35, s20, 31'), read, ('s_endpgm',))
        self.assertEqual([(x['rule'], x['severity'], x['pc']) for x in f], [('MISSING_ALU_WAIT', 'error', 12)])

    def test_sgpr_pair_tracked_later_in_a_loop(self):
        # The pair of s4 is first read by a vector instruction at pc16, inside the loop that starts at
        # pc8. LLVM's pass joins that into the loop head, so the scalar read of s4 at pc8 ends the pending
        # vector write of s8 and the compiler emits no wait at pc12: not an error here either.
        loop = (('s_and_b32', 's9, s4, s4'), ('v_cndmask_b32', 'v0, v1, v2, s8'),
                ('v_mov_b32', 'v3, s4'), ('s_cbranch_scc1', '', -4), ('s_endpgm',))
        self.assertEqual(run_waits(('v_mov_b32', 'v0, s8'), ('v_cmp_eq_u32', 's8, v0, v1'), *loop), [])
        # Without the scalar read the wait is required on every path.
        f = run_waits(('v_mov_b32', 'v0, s8'), ('v_cmp_eq_u32', 's8, v0, v1'), ('s_nop', '0'), *loop[1:])
        self.assertEqual([(x['rule'], x['pc']) for x in f], [('MISSING_ALU_WAIT', 12)])

    def test_sgpr_hazard_on_one_path(self):
        # pc0 branches over the wait at pc12 to the read at pc16.
        f = run_waits(('v_mov_b32', 'v0, s4'), ('s_mov_b32', 's4, 1'), ('s_cbranch_scc1', '', 1),
                      ('s_wait_alu', 'depctr_sa_sdst(0)', 0xff9e), ('s_add_co_u32', 's6, s4, 1'), ('s_endpgm',))
        self.assertEqual([(x['rule'], x['pc']) for x in f], [('MISSING_ALU_WAIT', 16)])

    def test_one_finding_per_producer(self):
        f = run_waits(('global_load_b32', 'v0, v[2:3], off'), ('v_add_f32', 'v1, v0, v2'),
                      ('v_add_f32', 'v3, v0, v2'), ('s_endpgm',))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0]['later_uses'], [8])


class SourceTests(unittest.TestCase):
    ASM = '\n'.join([
        '\t.text', '\t.file\t0 "/src" "k.hip"', 'k:', '\t.loc\t0 7 3 prologue_end',
        '\tv_mov_b32_e32 v0, 0', ';;#ASMSTART', '\tds_load_b32 v1, v0', ';;#ASMEND',
        '\ts_endpgm', '.Lfunc_end0:', '\t.section\t.AMDGPU.csdata', '\t.amdgpu_metadata',
        'amdhsa.kernels:', '  - .name: k', '\t.end_amdgpu_metadata'])

    def test_parse_asm(self):
        fns = hipsrc.parse_asm(self.ASM)
        self.assertEqual(list(fns), ['k'])
        self.assertEqual([i['mnemonic'] for i in fns['k']], ['v_mov_b32_e32', 'ds_load_b32', 's_endpgm'])
        self.assertEqual([i['inline_asm'] for i in fns['k']], [False, True, False])
        self.assertEqual(fns['k'][0]['loc'], {'file': '/src/k.hip', 'line': 7, 'column': 3})

    def test_source_map_tolerates_alignment_nops(self):
        fns = hipsrc.parse_asm(self.ASM)
        dec = [{'pc': 0, 'op': 'v_mov_b32_e32'}, {'pc': 4, 'op': 's_nop'}, {'pc': 8, 'op': 'ds_load_b32'},
               {'pc': 12, 'op': 's_endpgm'}]
        m, why = hipsrc.source_map(dec, fns['k'])
        self.assertIsNone(why)
        self.assertTrue(m[8]['inline_asm'])

    def test_compile_commands_flags(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'k.hip')
            src.write_text('')
            db = Path(d, 'compile_commands.json')
            db.write_text(json.dumps([{'directory': d, 'file': 'k.hip', 'command':
                                       f'/opt/rocm/bin/clang++ -DX=1 -Iinc -O2 --offload-arch=gfx1201 '
                                       f'-MD -MT k.o -MF k.d -x hip -c k.hip -o k.o -mcumode'}]))
            cc, flags, cwd = hipsrc.flags_from_compile_commands(db, src)
        self.assertEqual(cc, '/opt/rocm/bin/clang++')
        self.assertEqual(flags, ['-DX=1', '-Iinc', '-O2', '-mcumode'])

    def test_lexical_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'k.hip')
            p.write_text('__global__ void k(float* o) {\n'
                         '  // asm volatile("ds_load_b32 %0, %1") in a comment is ignored\n'
                         '  float v; asm volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(0));\n'
                         '  v = __shfl_xor(v, 1, 64);\n}\n')
            got = {(f['rule'], f['loc']['line']) for f in hipsrc.source_checks(p, lambda s: s)}
        self.assertEqual(got, {('HIP_INLINE_ASM_MEMORY', 3), ('HIP_WAVE64_WIDTH', 4),
                               ('HIP_NO_LAUNCH_BOUNDS', 1)})


def tool(name, near=None):
    try:
        return hipsrc.find_tool(name, near=near)
    except Exception:
        return None


def gfx1201_compiler():
    for c in (os.environ.get('RADISA_TEST_CXX'), '/usr/lib/llvm-20/bin/clang++', '/opt/rocm/llvm/bin/clang++',
              shutil.which('clang++')):
        if c and Path(c).exists():
            return c
    return None


def run_check(*argv):
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as d, redirect_stdout(out):
        j = Path(d, 'r.json')
        code = main(['check', *map(str, argv), '--json', str(j)])
        report = json.loads(j.read_text())
    return code, report, out.getvalue()


def findings(report):
    return report['findings'] + [f for k in report['kernels'] for f in k['findings']]


@unittest.skipUnless(tool('llvm-mc') or os.environ.get('RAD_ISA_REQUIRE_LLVM'), 'llvm-mc not available')
class AssemblyCheckTests(unittest.TestCase):
    def test_missing_waits(self):
        code, r, text = run_check(FIX / 'missing_waits.s')
        self.assertEqual(code, 1)
        got = [(f['rule'], f['loc']['line']) for f in findings(r)]
        self.assertEqual(got, [('MISSING_WAIT', 4), ('MISSING_WAIT', 14), ('MISSING_WAIT', 19)])
        self.assertIn('missing_waits.s:19:1: error: [MISSING_WAIT]', text)

    def test_clean(self):
        code, r, _ = run_check(FIX / 'clean_waits.s')
        self.assertEqual((code, r['status'], findings(r)), (0, 'PASS', []))

    def test_wmma(self):
        code, r, _ = run_check(FIX / 'wmma_hazard.s')
        self.assertEqual(code, 1)
        self.assertEqual([f['rule'] for f in findings(r)], ['WMMA_D_TO_AB'])

    def test_d16_halves(self):
        code, r, _ = run_check(FIX / 'd16_halves.s')
        self.assertEqual((code, findings(r)), (0, []))
        self.assertEqual(r['kernels'][0]['operand_model']['d16_other_half_stores'], 1)
        code, r, _ = run_check(FIX / 'd16_missing_wait.s')
        self.assertEqual(code, 1)
        self.assertEqual([(f['rule'], f['loc']['line']) for f in findings(r)],
                         [('MISSING_WAIT', 5), ('MISSING_WAIT', 10)])

    def test_alu_waits(self):
        code, r, _ = run_check(FIX / 'clean_alu_waits.s')
        self.assertEqual((code, findings(r)), (0, []))
        self.assertEqual(r['kernels'][0]['operand_model']['alu_waits'], 3)
        code, r, _ = run_check(FIX / 'missing_alu_waits.s')
        self.assertEqual(code, 1)
        self.assertEqual([(f['rule'], f['loc']['line'], f['required_wait']) for f in findings(r)],
                         [('MISSING_ALU_WAIT', 4, 's_wait_alu depctr_va_sdst(0)'),
                          ('MISSING_ALU_WAIT', 6, 's_wait_alu depctr_sa_sdst(0)'),
                          ('MISSING_ALU_WAIT', 10, 's_wait_alu depctr_va_vcc(0)')])

    def test_snippet_may_end_without_endpgm(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'x.s')
            p.write_text('v_add_f32 v4, v1, v2\n')
            code, r, _ = run_check(p)
        self.assertEqual(code, 0)
        self.assertEqual([(f['rule'], f['severity']) for f in findings(r)], [('CFG_FALLTHROUGH', 'info')])

    def test_whole_kernel_is_clean(self):
        code, r, _ = run_check(FIX / 'kernel_reduce.s')
        self.assertEqual((code, r['status'], findings(r)), (0, 'PASS', []))
        self.assertEqual(r['kernels'][0]['instructions'], 22)

    def mutants(self):
        """kernel_reduce.s with one fault each: (what, text, rule that has to report it)."""
        lines = (FIX / 'kernel_reduce.s').read_text().splitlines(keepends=True)
        expected = {'s_wait_kmcnt 0x0': 'MISSING_WAIT', 's_wait_loadcnt 0x0': 'MISSING_WAIT',
                    's_wait_dscnt 0x0': None, 's_endpgm': 'CFG_FALLTHROUGH'}
        dscnt = iter(['LDS_STORE_BEFORE_BARRIER', 'MISSING_WAIT'])
        for n, ln in enumerate(lines):
            if ln.strip() in expected:
                rule = expected[ln.strip()] or next(dscnt)
                yield f'line {n + 1} `{ln.strip()}` removed', ''.join(lines[:n] + lines[n + 1:]), rule
        n = next(i for i, ln in enumerate(lines) if ln.strip() == 's_endpgm')
        yield ('conditional branch as last instruction',
               ''.join(lines[:n] + ['\ts_cbranch_scc1 .LBB0_2\n'] + lines[n + 1:]), 'CFG_FALLTHROUGH')
        yield ('a VGPR above the allocation', ''.join(lines).replace(
            'v_add_f32_e32 v0, v2, v3', 'v_add_f32_e32 v0, v2, v9'), 'VGPR_OUT_OF_RANGE')
        yield ('wait for the barrier moved behind it', ''.join(lines).replace(
            '\ts_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1\n',
            '\ts_barrier_signal -1\n\ts_barrier_wait -1\n\ts_wait_dscnt 0x0\n'), 'LDS_STORE_BEFORE_BARRIER')

    def test_every_single_fault_in_a_kernel_fails(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            for what, text, rule in self.mutants():
                p = Path(d, 'm.s')
                p.write_text(text)
                code, r, _ = run_check(p)
                errors = {f['rule'] for f in findings(r) if f['severity'] == 'error'}
                self.assertEqual((code, r['status'], errors), (1, 'FAIL', {rule}), what)
                seen.append(rule)
        self.assertEqual(len(seen), 8)
        self.assertEqual(set(seen), {'MISSING_WAIT', 'LDS_STORE_BEFORE_BARRIER', 'CFG_FALLTHROUGH',
                                     'VGPR_OUT_OF_RANGE'})

    def test_vgpr_out_of_range(self):
        # The descriptor allocates 8 VGPRs (granule of 8 in wave32): v7 is the last one the kernel owns.
        text = (FIX / 'kernel_reduce.s').read_text()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'm.s')
            p.write_text(text.replace('v_add_f32_e32 v0, v2, v3', 'v_add_f32_e32 v0, v2, v7'))
            code, r, _ = run_check(p)
            self.assertEqual((code, findings(r)), (0, []))
            p.write_text(text.replace('v_add_f32_e32 v0, v2, v3', 'v_add_f32_e32 v0, v2, v8'))
            code, r, _ = run_check(p)
            f = [x for x in findings(r) if x['rule'] == 'VGPR_OUT_OF_RANGE']
            self.assertEqual(code, 1)
            self.assertEqual([(x['instruction'], x['registers'], x['allocated_vgprs']) for x in f],
                             [('v_add_f32_e32 v0, v2, v8', ['v8'], 8)])
            self.assertIn('3.3.2', f[0]['source'])

    def test_assembler_error_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'x.s')
            p.write_text('v_add_f32 v4, v1\n')
            code, r, text = run_check(p)
        self.assertEqual(code, 1)
        self.assertIn('too few operands', text)


class FingerprintTests(unittest.TestCase):
    MODULE = """target triple = "amdgcn-amd-amdhsa"
@lut = internal addrspace(4) constant [2 x i32] [i32 1, i32 2]
define amdgpu_kernel void @a(ptr %p) #0 {
  %v = load i32, ptr addrspace(4) @lut, align 4, !tbaa !5
  store i32 %v, ptr %p, align 4, !dbg !9
  call void asm sideeffect "s_nop 0", ""(), !srcloc !12
  ret void
}
define amdgpu_kernel void @b(ptr %p) #1 {
  store i32 7, ptr %p, align 4, !dbg !10
  ret void
}
attributes #0 = { "target-features"="+wavefrontsize32" }
attributes #1 = { "target-features"="+wavefrontsize32" }
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char"}
!9 = !DILocation(line: 4, column: 3, scope: !11)
!10 = !DILocation(line: 9, column: 3, scope: !11)
!12 = !{i64 120}
"""

    def prints(self, text):
        mod = incremental.parse_module(text)
        fp = incremental.Fingerprints(mod, 'salt')
        return {k: fp.of(k) for k in mod['kernels']}

    def test_lines_and_metadata_numbering_do_not_count(self):
        base = self.prints(self.MODULE)
        moved = self.MODULE.replace('line: 4', 'line: 7').replace('i64 120', 'i64 160')
        renumbered = self.MODULE.replace('!tbaa !5', '!tbaa !15').replace('!5 = ', '!15 = ')
        self.assertEqual(self.prints(moved), base)
        self.assertEqual(self.prints(renumbered), base)

    def test_code_globals_and_attributes_count(self):
        base = self.prints(self.MODULE)
        self.assertNotEqual(self.prints(self.MODULE.replace('i32 7', 'i32 8'))['b'], base['b'])
        self.assertEqual(self.prints(self.MODULE.replace('i32 7', 'i32 8'))['a'], base['a'])
        self.assertNotEqual(self.prints(self.MODULE.replace('i32 2]', 'i32 3]'))['a'], base['a'])
        changed = self.MODULE.replace('attributes #1 = { "target-features"="+wavefrontsize32" }',
                                      'attributes #1 = { "target-features"="+wavefrontsize32,+cumode" }')
        self.assertNotEqual(self.prints(changed)['b'], base['b'])

    def test_line_map(self):
        m = incremental._line_map(['a', 'b', 'c'], ['x', 'a', 'b', 'y', 'c'])
        self.assertEqual(m, {1: 2, 2: 3, 3: 5})


@unittest.skipUnless(gfx1201_compiler(), 'no HIP-capable clang')
class HipCheckTests(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        os.environ['RADISA_CACHE'] = self.cache.name

    def tearDown(self):
        os.environ.pop('RADISA_CACHE', None)
        self.cache.cleanup()

    def check(self, name, *extra):
        return run_check(FIX / name, '--cxx', gfx1201_compiler(), '--flag', '-nogpuinc -nogpulib', *extra)

    def test_per_kernel_codegen_matches_whole_unit(self):
        full = self.check('clean_compiled.hip', '--no-incremental')[1]
        inc = self.check('clean_compiled.hip')[1]
        shape = lambda r: {k['name']: (k['instructions'], k['metadata']) for k in r['kernels']}
        self.assertEqual(shape(inc), shape(full))
        self.assertEqual(inc['incremental']['mode'], 'per-kernel')
        # llc where the toolchain has it, clang's code generator where it does not (ROCm LLVM 24).
        self.assertIn(inc['incremental']['codegen'], ('llc', 'clang'))

    def test_cache_follows_edits(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'k.hip')
            src.write_text((FIX / 'inline_asm_lds.hip').read_text())
            run = lambda: run_check(src, '--cxx', gfx1201_compiler(), '--flag', '-nogpuinc -nogpulib')[1]
            first = run()
            self.assertEqual(first['incremental']['compiled'], 2)
            src.write_text('// moved\n// down\n' + src.read_text())
            moved = run()
            self.assertEqual((moved['incremental']['compiled'], moved['incremental']['reused']), (0, 2))
            [f] = [f for k in moved['kernels'] for f in k['findings'] if f['rule'] == 'MISSING_WAIT']
            self.assertEqual((f['loc']['line'], f['producer_origin']['line']), (17, 16))
            self.assertIn('k.hip:16', f['message'])
            text = src.read_text().replace('ds_load_b32 %0, %1" :', 'ds_load_b32 %0, %1\\n\\ts_wait_dscnt 0x0" :', 1)
            src.write_text(text)
            fixed = run()
            self.assertEqual(fixed['incremental']['compiled'], 1)
            self.assertEqual([f for k in fixed['kernels'] for f in k['findings'] if f['rule'] == 'MISSING_WAIT'], [])

    def test_inline_asm_race_points_at_source(self):
        code, r, _ = self.check('inline_asm_lds.hip')
        self.assertEqual(code, 1)
        by_kernel = {k['name']: [f for f in k['findings'] if f['rule'] == 'MISSING_WAIT'] for k in r['kernels']}
        self.assertEqual(by_kernel['fixed'], [])
        [f] = by_kernel['racy']
        self.assertEqual(f['loc']['line'], 15)
        self.assertTrue(f['producer_origin']['inline_asm'])
        self.assertEqual(f['producer_origin']['line'], 14)
        lexical = [f for f in r['findings'] if f['rule'] == 'HIP_INLINE_ASM_MEMORY']
        self.assertEqual([f['loc']['line'] for f in lexical], [14])

    def test_compiler_scheduled_code_is_clean(self):
        code, r, _ = self.check('clean_compiled.hip')
        self.assertEqual(code, 0, findings(r))
        self.assertEqual(r['operand_model']['unknown'], 0)

    def test_barrier_without_release_fence(self):
        code, r, text = self.check('barrier_no_fence.hip')
        self.assertEqual(code, 1)
        f = [x for x in findings(r) if x['severity'] == 'error']
        self.assertEqual([(x['rule'], x['loc']['line']) for x in f], [('LDS_STORE_BEFORE_BARRIER', 11)])
        self.assertIn('barrier_no_fence.hip:10', f[0]['message'])

    def test_compile_error_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'bad.hip')
            p.write_text('__attribute__((global)) void k() { undeclared(); }\n')
            code, r, text = run_check(p, '--cxx', gfx1201_compiler(), '--flag', '-nogpuinc -nogpulib')
        self.assertEqual(code, 1)
        self.assertEqual([f['rule'] for f in r['findings']], ['COMPILE'])


if __name__ == '__main__':
    unittest.main()
