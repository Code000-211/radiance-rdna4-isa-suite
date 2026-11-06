"""`radisa check`: AMD XML operand table, counter model, HIP/assembly front ends."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import tempfile
import unittest

from radisa import hipsrc, isadb, vopd, waits
from radisa.cli import main
from radisa.rules import cfg
from support import ROOT, seq

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

    def test_assembler_error_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'x.s')
            p.write_text('v_add_f32 v4, v1\n')
            code, r, text = run_check(p)
        self.assertEqual(code, 1)
        self.assertIn('too few operands', text)


@unittest.skipUnless(gfx1201_compiler(), 'no HIP-capable clang')
class HipCheckTests(unittest.TestCase):
    def check(self, name):
        return run_check(FIX / name, '--cxx', gfx1201_compiler(), '--flag', '-nogpuinc -nogpulib')

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

    def test_compile_error_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'bad.hip')
            p.write_text('__attribute__((global)) void k() { undeclared(); }\n')
            code, r, text = run_check(p, '--cxx', gfx1201_compiler(), '--flag', '-nogpuinc -nogpulib')
        self.assertEqual(code, 1)
        self.assertEqual([f['rule'] for f in r['findings']], ['COMPILE'])


if __name__ == '__main__':
    unittest.main()
