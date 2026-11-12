"""UNINITIALIZED_READ (radisa/liveness.py): use before definition against the gfx1201 entry ABI."""
import os
from pathlib import Path
import tempfile
import unittest

from radisa import check, hipsrc, isadb, liveness, rules, waits
from support import (CHECK_FIX, all_findings, check_args, hip_compiles, needs_llvm_mc, needs_minimal_kernel,
                     require_llvm, run_check, seq, _need)

TABLE = isadb.load()
# s[0:1] kernarg pointer (user SGPR count 2, kernarg segment ptr), s2 work-group id X: kernel_asm().
DESC = {'wave': 32, 'encoded_vgprs': 16, 'rsrc1': 1, 'rsrc2': (2 << 1) | (1 << 7), 'rsrc3': 0,
        'properties': (1 << 3) | (1 << 10), 'kernarg': 16, 'kernarg_preload': 0}


class Registered:
    """liveness.kernel_pass in check.KERNEL_PASSES for the duration (unless already registered)."""
    def __enter__(self):
        self.added = liveness.kernel_pass not in check.KERNEL_PASSES
        if self.added:
            check.KERNEL_PASSES.append(liveness.kernel_pass)

    def __exit__(self, *exc):
        if self.added:
            check.KERNEL_PASSES.remove(liveness.kernel_pass)


def run_pass(*spec, descriptor=DESC, options=None, callee=False):
    ins = seq(*spec)
    edges, reachable, _ = rules.cfg(ins)
    kernel = {'name': 'x', 'descriptor': descriptor, 'metadata': {'.wavefront_size': 32}}
    if callee:
        kernel['callee'] = True
    ctx = {'kernel': kernel, 'instructions': ins, 'edges': edges, 'reachable': reachable, 'table': TABLE,
           'wave': 32, 'origin': None, 'models': waits.operand_models(ins, TABLE), 'options': options or {}}
    return liveness.kernel_pass(ctx)


def regs(findings):
    return sorted(r for f in findings for r in f['registers'])


class EntryStateTests(unittest.TestCase):
    def test_kernel_asm_descriptor(self):
        m, why = liveness.entry_state(DESC)
        self.assertEqual(why['defined'], 's0-s2, v0, EXEC, TTMP0-15')
        for r in ('s0', 's1', 's2', 'v0', 'exec_lo', 'exec_hi', 'ttmp7', 'ttmp9'):
            self.assertTrue(m & liveness.mask(isadb.registers(r)), r)
        for r in ('s3', 'v1', 'v2', 'vcc_lo', 'm0', 'scc'):
            self.assertFalse(m & liveness.mask(isadb.registers(r)), r)

    def test_user_and_system_sgprs(self):
        # Private segment buffer (4) + dispatch ptr (2) + kernarg ptr (2) + 3 preloaded kernargs = 11
        # user SGPRs although USER_SGPR_COUNT says 8; then work-group id X, Y, Z and work-group info.
        d = {**DESC, 'properties': 0b1011 | (1 << 10), 'kernarg_preload': 3 | (5 << 7),
             'rsrc2': (8 << 1) | (0b1111 << 7) | (2 << 11)}
        m, why = liveness.entry_state(d)
        self.assertEqual(why['user_sgprs'], 11)
        self.assertEqual(why['system_sgprs'], ['work-group id X', 'work-group id Y', 'work-group id Z',
                                               'work-group info'])
        self.assertTrue(m >> 14 & 1)
        self.assertFalse(m >> 15 & 1)
        # Packed work-item IDs: ENABLE_VGPR_WORKITEM_ID = 2 still sets up v0 only.
        self.assertEqual(why['vgprs'], ['v0'])
        _, unpacked = liveness.entry_state(d, packed_tid=False)
        self.assertEqual(unpacked['vgprs'], ['v0', 'v1', 'v2'])


class AnalysisTests(unittest.TestCase):
    def test_reads_before_write(self):
        f = run_pass(('v_add_f32', 'v2, v200, v1'), ('s_add_co_u32', 's6, s7, s2'),
                     ('v_add_f32', 'v3, v200, v2'), ('s_endpgm',))
        self.assertEqual(regs(f), ['s7', 'v1', 'v200'])
        self.assertEqual({x['severity'] for x in f}, {'review'})
        self.assertEqual({x['rule'] for x in f}, {'UNINITIALIZED_READ'})
        first = [x for x in f if 'v200' in x['registers']][0]
        self.assertEqual((first['pc'], first['later_uses']), (0, [8]))
        self.assertIn('packs the work-item IDs into v0', first['message'])
        self.assertTrue(all('AMDGPUUsage' in x['source'] for x in f))

    def test_written_before_read(self):
        self.assertEqual(run_pass(('s_load_b64', 's[4:5], s[0:1], 0x0'), ('v_mov_b32', 'v1, s2'),
                                  ('s_wait_kmcnt', '0x0'), ('v_add_f32', 'v2, v1, v0'),
                                  ('global_store_b32', 'v1, v2, s[4:5]'), ('s_endpgm',)), [])

    def test_one_path_without_write(self):
        # s_cbranch_scc1 +1 skips the write of v4: a uniform branch, a real path.
        f = run_pass(('s_cmp_eq_u32', 's2, 0'), ('s_cbranch_scc1', '', 1), ('v_mov_b32', 'v4, 1.0'),
                     ('v_add_f32', 'v5, v4, v0'), ('s_endpgm',))
        self.assertEqual(regs(f), ['v4'])

    def test_implicit_operands(self):
        # V_CNDMASK reads VCC; S_CSELECT reads SCC; neither is set up at launch.
        f = run_pass(('v_cndmask_b32', 'v1, 0, v0, vcc_lo'), ('s_cselect_b32', 's3, 1, 0'), ('s_endpgm',))
        self.assertEqual(regs(f), ['scc', 'vcc_lo'])
        self.assertEqual(run_pass(('v_cmp_gt_u32', 'vcc_lo, 64, v0'), ('v_cndmask_b32', 'v1, 0, v0, vcc_lo'),
                                  ('s_cmp_eq_u32', 's2, 0'), ('s_cselect_b32', 's3, 1, 0'), ('s_endpgm',)), [])

    def test_idioms(self):
        self.assertEqual(run_pass(('v_xor_b32', 'v1, v1, v1'), ('v_sub_nc_u32', 'v2, v2, v2'),
                                  ('s_xor_b32', 's8, s8, s8'), ('s_sub_co_u32', 's9, s9, s9'),
                                  ('v_add_f32', 'v3, v1, v2'), ('s_add_co_u32', 's3, s8, s9'), ('s_endpgm',)), [])
        # A self-move reads nothing and defines nothing: the later read is still reported.
        self.assertEqual(regs(run_pass(('v_mov_b32', 'v6, v6'), ('v_add_f32', 'v1, v6, v0'), ('s_endpgm',))),
                         ['v6'])
        # v_xor of two different registers is no idiom.
        self.assertEqual(regs(run_pass(('v_xor_b32', 'v1, v1, v0'), ('s_endpgm',))), ['v1'])

    def test_sendmsg_m0(self):
        self.assertEqual(run_pass(('s_sendmsg', 'sendmsg(MSG_DEALLOC_VGPRS)'), ('s_endpgm',)), [])

    def test_divergent_if_else(self):
        # if (v0 < 64) v6 = 2 else v6 = 1; the two execz edges are taken only when no lane runs the
        # code they skip, so no lane reads v6 unwritten.
        spec = [('v_cmp_gt_u32', 'vcc_lo, 64, v0'), ('s_and_saveexec_b32', 's9, vcc_lo'),
                ('s_xor_b32', 's9, exec_lo, s9'), ('s_cbranch_execz', '', 1), ('v_mov_b32', 'v6, 2.0'),
                ('s_and_not1_saveexec_b32', 's9, s9'), ('s_cbranch_execz', '', 1), ('v_mov_b32', 'v6, 1.0'),
                ('s_or_b32', 'exec_lo, exec_lo, s9'), ('v_add_f32', 'v7, v6, v0'), ('s_endpgm',)]
        self.assertEqual(run_pass(*spec), [])
        # The same CFG with a uniform (SCC) branch is a real path past both writes.
        uniform = [(op if op != 's_cbranch_execz' else 's_cbranch_scc1', *rest) for op, *rest in spec]
        self.assertEqual(regs(run_pass(*uniform)), ['v6'])
        # s_cbranch_execnz: the fall-through is the no-lane edge.
        f = run_pass(('v_cmpx_gt_u32', '64, v0'), ('s_cbranch_execnz', '', 1), ('s_branch', '', 1),
                     ('v_mov_b32', 'v6, 2.0'), ('v_add_f32', 'v7, v6, v0'), ('s_endpgm',))
        self.assertEqual(f, [])

    def test_tuples(self):
        # v5 is the undef upper element of a 64-bit value; v[8:9] is not written at all.
        f = run_pass(('v_mov_b32', 'v4, v0'), ('global_store_b64', 'v0, v[4:5], s[0:1]'),
                     ('global_store_b64', 'v0, v[8:9], s[0:1]'), ('s_endpgm',))
        self.assertEqual(regs(f), ['v8', 'v9'])

    def test_d16_load_defines(self):
        self.assertEqual(run_pass(('global_load_d16_b16', 'v4, v0, s[0:1]'), ('s_wait_loadcnt', '0x0'),
                                  ('v_add_f16', 'v4, v4, v4'), ('s_endpgm',)), [])

    def test_call_defines_everything(self):
        self.assertEqual(run_pass(('s_mov_b64', 's[10:11], 0'), ('s_swappc_b64', 's[30:31], s[10:11]'),
                                  ('v_add_f32', 'v7, v7, v40'), ('s_add_co_u32', 's3, s50, s51'),
                                  ('s_endpgm',)), [])

    def test_unknown_instruction(self):
        # Outside the operand table: it defines what it names; its own reads are a guess, not reported.
        self.assertEqual(run_pass(('v_made_up_op', 'v9, v10'), ('v_add_f32', 'v1, v9, v10'), ('s_endpgm',)), [])

    def test_unreachable_not_reported(self):
        self.assertEqual(run_pass(('s_endpgm',), ('v_add_f32', 'v1, v9, v10'), ('s_endpgm',)), [])

    def test_loop_carried(self):
        # v3 is written at the end of the loop body only: the first iteration reads it unwritten.
        f = run_pass(('s_mov_b32', 's3, 4'), ('v_add_f32', 'v4, v3, v0'), ('v_mov_b32', 'v3, v4'),
                     ('s_add_co_i32', 's3, s3, -1'), ('s_cmp_lg_u32', 's3, 0'), ('s_cbranch_scc1', '', -5),
                     ('s_endpgm',))
        self.assertEqual(regs(f), ['v3'])
        self.assertEqual(f[0]['pc'], 4)

    def test_scope(self):
        spec = [('v_add_f32', 'v2, v200, v201'), ('s_endpgm',)]
        self.assertEqual(run_pass(*spec, callee=True), [])
        self.assertEqual(run_pass(*spec, descriptor={}), [])
        self.assertEqual(run_pass(*spec, options={'liveness': False}), [])
        f = run_pass(*spec, descriptor={}, options={'liveness_entry': ['v200']})
        self.assertEqual(regs(f), ['v201'])
        self.assertIn('v200, EXEC, TTMP0-15', f[0]['message'])


class CheckIntegrationTests(unittest.TestCase):
    @needs_minimal_kernel
    def test_uninit_kernel(self):
        with Registered():
            code, r = run_check(CHECK_FIX / 'liveness_uninit.s')
            strict, _ = run_check(CHECK_FIX / 'liveness_uninit.s', '--strict')
        f = [x for x in all_findings(r) if x['rule'] == 'UNINITIALIZED_READ']
        self.assertEqual(sorted((x['instruction'], tuple(x['registers'])) for x in f),
                         [('s_add_co_u32 s6, s7, s2', ('s7',)), ('v_add_f32_e32 v5, v4, v3', ('v4',)),
                          ('v_add_nc_u32_e32 v3, v1, v2', ('v1',))])
        self.assertEqual((code, r['status'], strict), (0, 'PASS', 1))

    @needs_minimal_kernel
    def test_clean_kernel(self):
        with Registered():
            code, r = run_check(CHECK_FIX / 'liveness_clean.s')
        self.assertEqual([x for x in all_findings(r) if x['rule'] == 'UNINITIALIZED_READ'], [])
        self.assertEqual((code, r['status']), (0, 'PASS'))

    @needs_llvm_mc
    def test_snippet_needs_entry_option(self):
        path = CHECK_FIX / 'liveness_snippet.s'
        with Registered():
            plain = check.analyze(check_args(path))
            args = check_args(path)
            args.pass_options = {'liveness_entry': ['s[40:41]', 'v0']}
            given = check.analyze(args)
        self.assertEqual([x for x in all_findings(plain) if x['rule'] == 'UNINITIALIZED_READ'], [])
        self.assertEqual(regs(x for x in all_findings(given) if x['rule'] == 'UNINITIALIZED_READ'),
                         ['v200', 'v201'])


# Compiler output must not trigger the rule, although LLVM may read undef registers (vector elements
# never set, phi inputs that are undef on a path, `implicit-def`). Divergent if/else and switch,
# partial vectors, loops, calls, scratch, cross-lane operations.
HIP = r'''
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
typedef float f4 __attribute__((ext_vector_type(4)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
#define T __builtin_amdgcn_workitem_id_x()
extern "C" __global__ void ifelse(float* o, const float* a, const float* b) {
  int t = T + __builtin_amdgcn_workgroup_id_x() * 256; float x;
  if (a[t] > 0.f) x = __builtin_sqrtf(a[t]) * b[t]; else x = __builtin_expf(b[t]) + a[t + 1];
  o[t] = x;
}
extern "C" __global__ void sw(int* o, const int* a) {
  int t = T; int v;
  switch (a[t] & 3) { case 0: v = a[t + 1]; break; case 1: v = t * 3; break; case 2: v = a[t + 2] * a[t]; break; default: v = 7; }
  o[t] = v;
}
extern "C" __global__ void partial(f4* o, const float* a) { int t = T; f4 v; v.x = a[t]; v.z = a[t + 1]; o[t] = v; }
extern "C" __global__ void halves(h2* o, const _Float16* a, const h2* b) { int t = T; h2 v; v.x = a[t]; v.y = a[t + 64]; o[t] = v * b[t]; }
extern "C" __global__ void uni(float* o, const float* a, int flag) {
  int t = T; float x; if (flag) x = a[t] * 3.f; float y = a[t + 1]; if (flag) y += x; o[t] = y;
}
extern "C" __global__ void loops(float* o, const float* a, int n, int m) {
  int t = T; float acc = 0, acc2 = 1;
  for (int i = 0; i < n; ++i) { acc += a[i * 256 + t]; if (acc > 3) acc2 *= acc; }
  for (int j = t; j < m; j += 7) acc2 += a[j];
  o[t] = acc + acc2;
}
extern "C" __global__ void xyz(int* o, int w, int h) {
  int x = T + __builtin_amdgcn_workgroup_id_x() * 16, y = __builtin_amdgcn_workitem_id_y() + __builtin_amdgcn_workgroup_id_y() * 16;
  int z = __builtin_amdgcn_workitem_id_z() + __builtin_amdgcn_workgroup_id_z() * 4;
  o[(z * h + y) * w + x] = x ^ y ^ z;
}
extern "C" __global__ void lanes(int* o, const int* a) {
  int t = T; int v = a[t];
  o[t] = __builtin_amdgcn_readfirstlane(v) + __builtin_amdgcn_ds_bpermute((t ^ 1) << 2, v)
       + __builtin_amdgcn_permlanex16(0, v, 0x01234567, 0x89abcdef, true, false)
       + __builtin_amdgcn_update_dpp(0, v, 0x111, 0xf, 0xf, false);
}
__device__ __attribute__((noinline)) float helper(float x, const float* p) { return x * p[3] + __builtin_sinf(x); }
extern "C" __global__ void caller(float* o, const float* a) { int t = T; o[t] = helper(a[t], a) + helper(a[t + 1], o); }
extern "C" __global__ void priv(float* o, const float* a, int k) {
  float arr[64]; int t = T;
  for (int i = 0; i < 64; ++i) arr[i] = a[t + i];
  o[t] = arr[k & 63] + arr[(k + t) & 63];
}
'''


def _cxx():
    try:
        return hipsrc.find_tool('clang++', None)
    except Exception:
        return None


class CompilerOutputTests(unittest.TestCase):
    """clang -O0, -O1, -O3 output of HIP: no UNINITIALIZED_READ (with clang 18 checked: 0 of 10 kernels at each level)."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.src = Path(cls.dir.name) / 'liveness_compiled.hip'
        cls.src.write_text(HIP)
        cls.why = hip_compiles(_cxx(), str(cls.src))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_no_finding_on_compiler_output(self):
        if self.why and not require_llvm():
            self.skipTest(f'needs a clang that compiles HIP for gfx1201: {self.why}')
        for opt in ('-O0', '-O1', '-O3'):
            with self.subTest(opt=opt), Registered():
                code, r = run_check(self.src, '--no-incremental', '--flag', f'-nogpuinc -nogpulib {opt}')
                kernels = [k for k in r['kernels'] if not k.get('callee')]   # noinline helpers are checked too
                self.assertEqual(len(kernels), 10, [f['message'] for f in r['findings']])
                self.assertEqual([(f.get('instruction'), f['registers']) for f in all_findings(r)
                                  if f['rule'] == 'UNINITIALIZED_READ'], [])


if __name__ == '__main__':
    unittest.main()
