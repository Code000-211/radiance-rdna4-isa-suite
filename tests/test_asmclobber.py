"""ASM_UNDECLARED_CLOBBER (radisa/asmclobber.py), its incremental wiring, and the lexical asm
spellings/macros of hipsrc.source_checks.

The IR tests use hand-written IR in the shape clang emits (checked against clang 18.1.3 output);
the HIP tests compile fixtures/check/asm_clobber.hip with the clang found here."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import CHECK_FIX, ROOT, check_text, hip_compiles, run_check  # noqa: E402

sys.path.insert(0, str(ROOT))
from radisa import asmclobber, hipsrc, incremental, isadb  # noqa: E402

TABLE = isadb.load()


def module(*calls, attrs='"target-features"="+wavefrontsize32"', extra=''):
    body = '\n'.join(f'  {c}' for c in calls)
    return f'''target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k(ptr %o, i32 %a) #0 !dbg !5 {{
{body}
  ret void
}}
attributes #0 = {{ {attrs} }}
!1 = !DIFile(filename: "k.hip", directory: "/src")
!5 = distinct !DISubprogram(name: "k", scope: !1, file: !1, line: 1, unit: !0)
!6 = distinct !DILexicalBlock(scope: !5, line: 2, column: 3)
!7 = !DILocation(line: 4, column: 3, scope: !6)
{extra}'''


def call(tmpl, cons, ret='i32', args='i32 %a', dbg=True):
    return f'%r{abs(hash((tmpl, cons))) % 10**6} = tail call {ret} asm sideeffect "{tmpl}", "{cons}"({args}) #1' + \
        (', !dbg !7' if dbg else '')


def findings(ir, path=None):
    return asmclobber.check_ir_text(ir, TABLE, path)


def regs(ir):
    return sorted(r for f in findings(ir) for r in f['registers'])


class ParseTests(unittest.TestCase):
    def test_unescape_and_constraints(self):
        self.assertEqual(asmclobber.ir_unescape(r'a\0A\09b\22\5C'), 'a\n\tb"\\')
        ops, clob = asmclobber.parse_constraints('=v,=&s,v,0,{v1},={s[2:3]},i,~{vcc},~{v[4:7]},~{memory}')
        self.assertEqual([o['kind'] for o in ops], ['v', 's', 'v', 'tied', 'reg', 'reg', 'imm'])
        self.assertEqual([o['output'] for o in ops], [True, True, False, False, False, True, False])
        self.assertEqual(clob, ['vcc', 'v[4:7]', 'memory'])

    def test_types(self):
        self.assertEqual([asmclobber.type_bits(t) for t in
                          ('i32', 'i64', 'half', 'double', '<4 x float>', 'ptr', 'ptr addrspace(3)',
                           '{ i32, i64 }', '[2 x i32]')], [32, 64, 16, 64, 128, 64, 32, 96, 64])
        self.assertEqual(asmclobber.leading_type('<2 x i32> noundef %x'), '<2 x i32>')
        self.assertEqual(asmclobber.leading_type('ptr addrspace(3) %p'), 'ptr addrspace(3)')

    def test_template_lines(self):
        t = 'v_mov_b32 v1, 0 ; comment v2\n.p2align 2\nlbl:\n  // only a comment\nl2: s_nop 0 /* v3 */\n'
        self.assertEqual(asmclobber.template_lines(t), [(0, 'v_mov_b32 v1, 0'), (4, 's_nop 0')])
        self.assertEqual(asmclobber.substitute('v_mov_b32 ${0:x}, $1 $$ ${:uid}', ['v192', 's40']),
                         'v_mov_b32 v192, s40 $ 0')


class ClobberTests(unittest.TestCase):
    def test_literal_write_undeclared_and_declared(self):
        bad = findings(module(call(r'v_mov_b32 v7, $1\0A\09v_add_nc_u32 $0, v7, $1', '=v,v')))
        self.assertEqual([(f['rule'], f['severity'], f['registers']) for f in bad],
                         [('ASM_UNDECLARED_CLOBBER', 'review', ['v7'])])
        self.assertIn('"v7"', bad[0]['message'])
        self.assertEqual(bad[0]['loc'], {'file': '/src/k.hip', 'line': 4, 'column': 3})
        self.assertEqual((bad[0]['function'], bad[0]['kernel']), ('k', True))
        self.assertEqual(regs(module(call(r'v_mov_b32 v7, $1\0A\09v_add_nc_u32 $0, v7, $1', '=v,v,~{v7}'))), [])

    def test_implicit_scc_vcc_exec_m0(self):
        self.assertEqual(regs(module(call('s_add_u32 $0, $1, 1', '=s,s'))), ['scc'])
        self.assertEqual(regs(module(call('s_add_u32 $0, $1, 1', '=s,s,~{scc}'))), [])
        self.assertEqual(regs(module(call('v_cmp_eq_u32 0, $1', '=v,v'))), ['vcc_lo'])
        for c in ('~{vcc}', '~{vcc_lo}'):
            self.assertEqual(regs(module(call('v_cmp_eq_u32 0, $1', '=v,v,' + c))), [])
        self.assertEqual(regs(module(call('v_cmpx_eq_u32 0, $1', '=v,v'))), ['exec_lo'])
        self.assertEqual(regs(module(call('v_cmpx_eq_u32 0, $1', '=v,v,~{exec}'))), [])
        self.assertEqual(regs(module(call('s_mov_b32 m0, $1', '=v,s'))), ['m0'])
        self.assertEqual(regs(module(call('s_mov_b32 m0, $1', '=v,s,~{m0}'))), [])

    def test_wave64_vcc_is_a_pair(self):
        ir = module(call('v_cmp_eq_u32 0, $1', '=v,v,~{vcc_lo}'), attrs='"target-features"="+wavefrontsize64"')
        self.assertEqual(regs(ir), ['vcc_hi'])

    def test_input_operands(self):
        bad = findings(module(call('v_mov_b32 $0, 0', 'v', ret='void')))
        self.assertEqual([(f['registers'], f['input_operands']) for f in bad], [(['$0'], [0])])
        # Tied ("+v" in the source) and outputs may be written.
        self.assertEqual(regs(module(call('v_add_nc_u32 $0, $0, 1', '=v,0'))), [])
        self.assertEqual(regs(module(call(r'v_add_nc_u32 $0, $1, 1\0Av_mov_b32 $2, 0', '=v,v,=v',
                                          ret='{ i32, i32 }'))), [])

    def test_explicit_registers(self):
        self.assertEqual(regs(module(call('v_mov_b32 $0, 1', '={v1}', args=''))), [])
        self.assertEqual(regs(module(call('v_mov_b32 v1, 1', '={v1}', args=''))), [])
        self.assertEqual(regs(module(call('v_mov_b32 v1, 1', '{v1}', ret='void'))), ['v1'])

    def test_wide_operands_and_ranges(self):
        self.assertEqual(regs(module(call('global_load_b64 $0, $1, off', '=v,v', ret='i64', args='ptr %o'))), [])
        self.assertEqual(regs(module(call('s_getpc_b64 s[4:5]', '=v', args=''))), ['s4', 's5'])
        self.assertEqual(regs(module(call('s_getpc_b64 s[4:5]', '=v,~{s4}', args=''))), ['s5'])
        self.assertEqual(regs(module(call('s_getpc_b64 s[4:5]', '=v,~{s4},~{s5}', args=''))), [])
        # clang rejects range clobbers, LLVM IR from other front ends may carry them.
        self.assertEqual(regs(module(call('v_mov_b32 v5, 0', '=v,~{v[4:7]}', args=''))), [])

    def test_reads_and_waits_are_not_writes(self):
        ir = module(call(r's_wait_loadcnt 0x0\0As_nop 0\0Av_add_f32 $0, s4, v3', '=v', ret='float', args=''))
        self.assertEqual(regs(ir), [])

    def test_unknown_instruction_is_info(self):
        [f] = findings(module(call('v_frobnicate v9, $1', '=v,v')))
        self.assertEqual((f['severity'], f['operand_model'], f['registers']), ('info', 'assumed', ['v9']))
        self.assertEqual(findings(module(call('v_frobnicate $0, $1', '=v,v'))), [])

    def test_no_debug_location(self):
        [f] = findings(module(call('v_mov_b32 v7, 0', '=v', args='', dbg=False)), 'k.hip')
        self.assertNotIn('loc', f)
        self.assertIn('no debug location', f['message'])

    def test_srcloc_points_at_the_template_line(self):
        src = 'void k() {\n  asm volatile("s_nop 0\\n"\n               "v_mov_b32 v7, 0" ::);\n}\n'
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'k.hip').write_text(src)
            quote = src.index('"')
            second = src.index('v_mov_b32')
            base = 9000
            extra = (f'!1 = !DIFile(filename: "k.hip", directory: "{d}")\n'
                     f'!8 = !{{i64 {base}, i64 {base + second - quote}}}\n'
                     '!9 = !DILocation(line: 2, column: 3, scope: !5)\n')
            ir = module('call void asm sideeffect "s_nop 0\\0Av_mov_b32 v7, 0", ""() #1, !dbg !9, !srcloc !8',
                        extra=extra).replace('!1 = !DIFile(filename: "k.hip", directory: "/src")\n', '')
            [f] = findings(ir)
            self.assertEqual(f['loc'], {'file': str(Path(d, 'k.hip')), 'line': 3, 'column': 17})
            # A macro location (bit 31) falls back to the statement line.
            [g] = findings(ir.replace(f'i64 {base + second - quote}', f'i64 {(1 << 31) + 5}'))
            self.assertEqual(g['loc']['line'], 2)

    def test_attach(self):
        r = [{'name': 'k', 'findings': []}]
        fs = findings(module(call('v_mov_b32 v7, 0', '=v', args='')))
        self.assertEqual(asmclobber.attach(r, fs + [{'function': 'helper'}]), [{'function': 'helper'}])
        self.assertEqual(len(r[0]['findings']), 1)


class ReviewRegressionTests(unittest.TestCase):
    """Issues of the adversarial review (scratchpad review-asmclobber/issues.md)."""

    def test_copies_of_one_statement_give_one_finding(self):
        c = call(r'v_mov_b32 v7, $1\0Av_mov_b32 $0, v7', '=v,v')
        fs = findings(module(*[c.replace('%r', f'%u{i}r') for i in range(8)]))
        self.assertEqual([(f['registers'], f['copies']) for f in fs], [(['v7'], 8)])
        # Different source locations stay apart.
        other = c.replace('%r', '%x').replace('!dbg !7', '!dbg !8')
        fs = findings(module(c, other, extra='!8 = !DILocation(line: 9, column: 3, scope: !6)'))
        self.assertEqual(sorted(f['loc']['line'] for f in fs), [4, 9])

    def test_exec_saved_and_restored_is_not_reported(self):
        save = r's_mov_b32 $1, exec_lo\0As_mov_b32 exec_lo, -1\0Av_mov_b32 $0, $2\0As_mov_b32 exec_lo, $1'
        self.assertEqual(regs(module(call(save, '=&v,=&s,v', ret='{ i32, i32 }'))), [])
        saveexec = r's_or_saveexec_b32 $1, -1\0Av_mov_b32 $0, $2\0As_mov_b32 exec_lo, $1'
        self.assertEqual(regs(module(call(saveexec, '=&v,=&s,v,~{scc}', ret='{ i32, i32 }'))), [])
        self.assertEqual(regs(module(call(r's_mov_b32 $0, m0\0As_mov_b32 m0, 0\0As_mov_b32 m0, $0', '=&s', args=''))), [])
        # Not restored: overwritten save slot, or restored from somewhere else.
        bad = r's_mov_b32 $1, exec_lo\0As_mov_b32 exec_lo, -1\0As_mov_b32 $1, 0\0As_mov_b32 exec_lo, $1'
        self.assertEqual(regs(module(call(bad, '=&v,=&s,v', ret='{ i32, i32 }'))), ['exec_lo'])
        self.assertEqual(regs(module(call(r's_mov_b32 exec_lo, -1\0As_mov_b32 exec_lo, s5', '=v', args=''))),
                         ['exec_lo'])

    def test_numeric_and_uid_labels(self):
        for t in (r's_cbranch_scc0 1f\0A1: v_mov_b32 v7, 0\0Av_mov_b32 $0, 0',
                  r'L${:uid}: v_mov_b32 v7, 0\0Av_mov_b32 $0, 0'):
            [f] = findings(module(call(t, '=v', args='')))
            self.assertEqual((f['severity'], f['registers']), ('review', ['v7']))
        self.assertEqual(asmclobber.template_lines('1: s_nop 0\nL${:uid}:\nv_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0'),
                         [(0, 's_nop 0'), (2, 'v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0')])

    def test_single_register_in_brackets(self):
        self.assertEqual(regs(module(call(r'v_mov_b32 v[7], 0\0Av_mov_b32 $0, 0', '=v', args=''))), ['v7'])
        self.assertEqual(regs(module(call(r's_mov_b32 s[7], 0\0Av_mov_b32 $0, 0', '=v', args=''))), ['s7'])
        self.assertEqual(regs(module(call(r'v_mov_b32 v[7], 0\0Av_mov_b32 $0, 0', '=v,~{v7}', args=''))), [])

    def test_vopd_halves_and_no_return_atomics(self):
        self.assertEqual(findings(module(call('v_dual_mov_b32 $0, v7 :: v_dual_mov_b32 $1, 0', '=v,=v',
                                              ret='{ i32, i32 }', args=''))), [])
        [f] = findings(module(call('v_dual_mov_b32 $0, v8 :: v_dual_mov_b32 v7, 0', '=v', args='')))
        self.assertEqual((f['severity'], f['registers']), ('review', ['v7']))
        for t, a in (('image_atomic_add_uint $0, $1, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D', 'i32 %a, i32 %a'),
                     ('buffer_atomic_add_u32 v7, v9, s[0:3], null offen', ''),
                     ('buffer_atomic_cmpswap_b32 v[7:8], v9, s[0:3], null offen', '')):
            self.assertEqual(findings(module(call(t, 'v,v' if a else '', ret='void', args=a))), [], t)
        # The returning form writes its VDATA register.
        self.assertEqual(regs(module(call('buffer_atomic_add_u32 v7, v9, s[0:3], null offen th:TH_ATOMIC_RETURN',
                                          '', ret='void', args=''))), ['v7'])
        [f] = findings(module(call('v_frobnicate v9, $1', '=v,v')))
        self.assertIn('v_frobnicate is not in the AMD ISA table', f['message'])

    def test_clobber_named_like_a_stand_in(self):
        self.assertEqual(regs(module(call('v_mov_b32 $0, 0', 'v,~{v192}', ret='void'))), ['$0'])
        self.assertEqual(regs(module(call('s_mov_b32 $0, 0', 's,~{s40}', ret='void'))), ['$0'])
        self.assertEqual(regs(module(call('v_mov_b32 $0, 0', 'v,{v192}', ret='void', args='i32 %a, i32 %a'))), ['$0'])

    def test_callbr_location_on_continuation_line(self):
        line = ('callbr void asm sideeffect "s_cmp_eq_u32 $0, 0\\0A s_mov_b32 s9, 0\\0A s_cbranch_scc1 ${1:l}", '
                '"s,!i,~{scc}"(i32 %a) #1\n          to label %4 [label %3], !dbg !7')
        [f] = findings(module(line), 'k.hip')
        self.assertEqual((f['registers'], f['loc']['line']), (['s9'], 4))
        self.assertNotIn('no debug location', f['message'])


class IncrementalWiringTests(unittest.TestCase):
    def test_kernel_findings_from_parsed_module(self):
        mod = incremental.parse_module(module(call('v_mov_b32 v7, 0', '=v', args='')))
        [f] = incremental.kernel_asm_findings('k', mod, TABLE)
        self.assertEqual((f['rule'], f['function'], f['loc']['line']), ('ASM_UNDECLARED_CLOBBER', 'k', 4))

    def test_schema_bumped(self):
        # Results cached before the rule existed lack its findings.
        self.assertEqual(incremental.SCHEMA, 'radisa-incremental-v4')


class LexicalAsmTests(unittest.TestCase):
    def lex(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'k.hip')
            p.write_text(text)
            return [f for f in hipsrc.source_checks(p, lambda s: s) if f['rule'] == 'HIP_INLINE_ASM_MEMORY']

    def test_every_spelling(self):
        text = ('void f(float &v, int a) {\n'
                '  __asm__ __volatile__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                '  asm volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                '  __asm volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                '  __asm__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                '  asm __volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                '  my_asm("ds_load_b32");  basm("ds_load_b32");\n'
                '  asm volatile("ds_load_b32 %0, %1\\n s_wait_dscnt 0" : "=v"(v) : "v"(a));\n}\n')
        self.assertEqual([f['loc']['line'] for f in self.lex(text)], [2, 3, 4, 5, 6])

    def test_macro_reported_at_use_with_definition_line(self):
        text = ('#define LDS_LOAD(v, a) \\\n'
                '  __asm__ __volatile__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a))\n'
                '#define WRAP(v, a) LDS_LOAD(v, a)\n'
                '#define UNUSED(v) asm volatile("ds_load_b32 %0, %0" : "+v"(v))\n'
                '#define OBJ asm volatile("global_load_b32 v1, v[2:3], off")\n'
                'void f(float &v, int a) {\n'
                '  LDS_LOAD(v, a);\n'
                '  WRAP(v, a);\n'
                '  OBJ;\n'
                '  int LDS_LOAD = 0;\n}\n')
        got = self.lex(text)
        self.assertEqual([f['loc']['line'] for f in got], [7, 8, 9])
        self.assertIn('macro LDS_LOAD defined at line 1', got[0]['message'])
        self.assertEqual([m['name'] for m in got[1]['macros']], ['WRAP', 'LDS_LOAD'])
        self.assertEqual(got[2]['macros'], [{'name': 'OBJ', 'line': 5}])

    def test_probe_file(self):
        probe = ('#define LDS_LOAD(v, a) __asm__ __volatile__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a))\n'
                 'void f(float &v, int a) {\n'
                 '  __asm__ __volatile__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                 '  asm volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                 '  LDS_LOAD(v, a);\n'
                 '  asm volatile("v_mov_b32 v7, %1" : "=v"(v) : "v"(a));\n}\n')
        self.assertEqual([f['loc']['line'] for f in self.lex(probe)], [3, 4, 5])


class LexicalReviewTests(unittest.TestCase):
    lex = LexicalAsmTests.lex

    def test_asm_labels_of_declarations(self):
        text = ('extern __device__ int buffer_count __asm__("buffer_count");\n'
                'extern __device__ int ds_count asm("ds_load_b32");\n'
                '__device__ int g(int) __asm__("global_load_b32");\n'
                'int h(int a) __attribute__((used)) asm("buffer_load_b32");\n'
                'void f(float &v, int a, int c) {\n'
                '  if (c) asm volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                '  else asm volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(a));\n'
                '  while (c--) __asm__("ds_load_b32 v1, v2");\n}\n')
        self.assertEqual([f['loc']['line'] for f in self.lex(text)], [6, 7, 8])

    def test_uses_in_directives_and_strings(self):
        text = ('#define PREFETCH_ASM asm volatile("global_load_b32 v1, v[2:3], off")\n'
                '#ifdef PREFETCH_ASM\n#undef PREFETCH_ASM\n#endif\n'
                '#define LDS_LOAD(v, a) __asm__ __volatile__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a))\n'
                'void h() { const char *m = "do not call LDS_LOAD(x, y) or asm(\\"ds_load_b32\\") here"; }\n'
                'void g(float &v, int a) { LDS_LOAD(v, a); }\n')
        self.assertEqual([f['loc']['line'] for f in self.lex(text)], [7])

    def test_conditional_definitions(self):
        text = ('#ifdef OLD_TOOLCHAIN\n'
                '#define LDS_LOAD(v, a) __asm__ __volatile__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a))\n'
                '#else\n#define LDS_LOAD(v, a) do { v = lds[a]; } while (0)\n#endif\n'
                'void g(float &v, int a) { LDS_LOAD(v, a); }\n')
        [f] = self.lex(text)
        self.assertEqual((f['severity'], f['loc']['line']), ('info', 6))
        self.assertIn('only 1 of the 2 definitions', f['message'])
        # `#if 0` code is never compiled: its definitions and statements do not count.
        text = ('#if 0\n#define LDS_LOAD(v, a) __asm__("ds_load_b32 %0, %1" : "=v"(v) : "v"(a))\n'
                'void old(float &v) { asm("ds_load_b32 v1, v2"); }\n'
                '#else\n#define LDS_LOAD(v, a) __asm__("ds_load_b32 %0, %1\\n s_wait_dscnt 0" : "=v"(v) : "v"(a))\n'
                '#endif\nvoid g(float &v, int a) { LDS_LOAD(v, a); }\n')
        self.assertEqual(self.lex(text), [])
        text = text.replace('\\n s_wait_dscnt 0', '')
        self.assertEqual([(f['severity'], f['loc']['line']) for f in self.lex(text)], [('review', 7)])

    def test_comment_marker_inside_asm_string(self):
        text = ('void f(float &v, unsigned long long p) {\n'
                '  asm volatile("global_load_b32 %0, %1, off // fetch\\n s_wait_loadcnt 0" : "=v"(v) : "v"(p));\n'
                '  asm volatile("global_load_b32 %0, %1, off /* x */" : "=v"(v) : "v"(p)); // s_wait_loadcnt 0\n}\n')
        self.assertEqual([f['loc']['line'] for f in self.lex(text)], [3])

    def test_many_macros_linear(self):
        import time
        lines = [f'#define ASM{i}(v) asm volatile("v_mov_b32 %0, {i}" : "=v"(v))' for i in range(600)]
        lines += [f'#define CONST{i} {i}' for i in range(3000)]
        lines += ['#define LOAD(v) asm volatile("ds_load_b32 %0, v1" : "=v"(v))', 'void f(int v) {']
        lines += [f'  ASM{i % 600}(v); v += CONST{i};' for i in range(3000)] + ['  LOAD(v);', '}']
        t = time.monotonic()
        got = self.lex('\n'.join(lines) + '\n')
        self.assertLess(time.monotonic() - t, 10)
        self.assertEqual([f['loc']['line'] for f in got], [len(lines) - 1])


class ParseAsmTests(unittest.TestCase):
    def test_block_comments_and_own_lines(self):
        text = ('\t.text\nk:\n/* a comment line */\n\tv_mov_b32 v0, 0 /* inline */\n'
                '/* a block\n   over two lines */\n\ts_endpgm\n')
        fns = hipsrc.parse_asm(text)
        self.assertEqual([i['mnemonic'] for i in fns['k']], ['v_mov_b32', 's_endpgm'])
        self.assertEqual([i['loc'] for i in fns['k']], [None, None])
        fns = hipsrc.parse_asm(text, asm_file='k.s')
        self.assertEqual([i['loc'] for i in fns['k']], [{'file': 'k.s', 'line': 4, 'column': 1},
                                                         {'file': 'k.s', 'line': 7, 'column': 1}])
        fns = hipsrc.parse_asm('radisa_snippet:\n' + text.split('k:\n', 1)[1], asm_file='s.s', line_offset=-1)
        self.assertEqual([i['loc']['line'] for i in fns['radisa_snippet']], [2, 5])
        # A .loc still wins over the .s line.
        fns = hipsrc.parse_asm('\t.file\t0 "/src" "k.hip"\nk:\n\t.loc\t0 7 3\n\ts_nop 0\n', asm_file='k.s')
        self.assertEqual(fns['k'][0]['loc'], {'file': '/src/k.hip', 'line': 7, 'column': 3})


def hip_compiler():
    for c in (os.environ.get('RADISA_TEST_CXX'), '/opt/rocm/llvm/bin/clang++', shutil.which('clang++')):
        if c and Path(c).exists() and not hip_compiles(c, str(CHECK_FIX / 'asm_clobber.hip')):
            return c
    return None


CXX = hip_compiler()
EXPECTED = {('undeclared', 18, ('v7',)), ('undeclared', 20, ('scc',)), ('undeclared', 21, ('vcc_lo',)),
            ('undeclared', 22, ('$0',)), ('undeclared', 23, ('v9',)), ('undeclared', 10, ('v8',))}


def asm_findings(report):
    return {(k['name'], f['loc']['line'], tuple(f['registers']))
            for k in report['kernels'] for f in k['findings'] if f['rule'] == 'ASM_UNDECLARED_CLOBBER'}


@unittest.skipUnless(CXX or os.environ.get('RAD_ISA_REQUIRE_LLVM') == '1', 'no clang that compiles HIP for gfx1201')
class HipTests(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        os.environ['RADISA_CACHE'] = self.cache.name

    def tearDown(self):
        os.environ.pop('RADISA_CACHE', None)
        self.cache.cleanup()

    def run_on(self, src):
        return run_check(src, '--cxx', CXX, '--flag', '-nogpuinc -nogpulib')[1]

    def test_incremental_reports_and_cache_keeps_them(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'asm_clobber.hip')
            src.write_text((CHECK_FIX / 'asm_clobber.hip').read_text())
            first = self.run_on(src)
            self.assertEqual(asm_findings(first), EXPECTED)
            for f in (f for k in first['kernels'] for f in k['findings'] if f['rule'] == 'ASM_UNDECLARED_CLOBBER'):
                self.assertEqual(f['severity'], 'review')
            again = self.run_on(src)
            self.assertEqual(again['incremental']['reused'], 2)
            self.assertEqual(asm_findings(again), EXPECTED)
            src.write_text('// two\n// lines\n' + src.read_text())
            moved = self.run_on(src)
            self.assertEqual((moved['incremental']['compiled'], moved['incremental']['reused']), (0, 2))
            self.assertEqual(asm_findings(moved), {(k, line + 2, r) for k, line, r in EXPECTED})

    def test_old_schema_cache_is_recompiled_with_asm_findings(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'asm_clobber.hip')
            src.write_text((CHECK_FIX / 'asm_clobber.hip').read_text())
            self.run_on(src)
            [cache] = list(Path(self.cache.name).glob('*/cache.json'))
            data = json.loads(cache.read_text())
            # What a v2 run stored (no ASM_UNDECLARED_CLOBBER findings) must be discarded, not reused.
            data['schema'] = 'radisa-incremental-v2'
            for e in data['kernels'].values():
                e['result']['findings'] = [f for f in e['result']['findings'] if f['rule'] != 'ASM_UNDECLARED_CLOBBER']
            cache.write_text(json.dumps(data))
            again = self.run_on(src)
            self.assertEqual((again['incremental']['compiled'], again['incremental']['reused']), (2, 0))
            self.assertEqual(asm_findings(again), EXPECTED)

    def test_unrolled_helper_reported_once(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'dup.hip')
            src.write_text('#define __global__ __attribute__((global))\n'
                           '#define __device__ __attribute__((device))\n'
                           '__device__ __attribute__((always_inline)) inline int bump(int a) {\n'
                           '  int r;\n'
                           '  asm volatile("v_mov_b32 v7, %1\\n v_mov_b32 %0, v7" : "=v"(r) : "v"(a));\n'
                           '  return r;\n}\n'
                           'extern "C" __global__ void d1(int* o, int a) {\n'
                           '  int s = 0;\n#pragma unroll\n'
                           '  for (int i = 0; i < 8; ++i) s += bump(a + i);\n  o[0] = s;\n}\n')
            r = self.run_on(src)
        fs = [f for k in r['kernels'] for f in k['findings'] if f['rule'] == 'ASM_UNDECLARED_CLOBBER']
        self.assertEqual([(f['loc']['line'], f['registers']) for f in fs], [(5, ['v7'])])
        self.assertEqual(fs[0].get('copies'), 8)

    def test_whole_module_mode_keeps_helper_findings(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, 'noinline.hip')
            src.write_text((CHECK_FIX / 'asm_clobber.hip').read_text().replace(
                '__attribute__((always_inline)) inline', '__attribute__((noinline))'))
            r = self.run_on(src)
        self.assertEqual(r['incremental']['mode'], 'whole-module')
        # The noinline helper is linted as a function of its own (check.check_object_bytes), so its
        # finding is attached to its result instead of becoming a report-level note.
        self.assertEqual(asm_findings(r), EXPECTED - {('undeclared', 10, ('v8',))} | {('_Z6helperi', 10, ('v8',))})
        [helper] = [k for k in r['kernels'] if k['name'] == '_Z6helperi']
        self.assertTrue(helper.get('callee'))
        [h] = [f for f in helper['findings'] if f['rule'] == 'ASM_UNDECLARED_CLOBBER']
        self.assertEqual((h['loc']['line'], h['registers'], h['kernel']), (10, ['v8'], False))
        self.assertEqual([f for f in r['findings'] if f['rule'] == 'ASM_UNDECLARED_CLOBBER'], [])

    def test_non_incremental_ir_matches(self):
        with tempfile.TemporaryDirectory() as d:
            c = hipsrc.compile_ir(CHECK_FIX / 'asm_clobber.hip', Path(d), CXX, ['-nogpuinc', '-nogpulib'], None)
        self.assertEqual(c['returncode'], 0, c['stderr'])
        got = {(f['function'], f['loc']['line'], tuple(f['registers']))
               for f in asmclobber.check_ir_text(c['ir'], TABLE, str(CHECK_FIX / 'asm_clobber.hip'))}
        self.assertEqual(got, EXPECTED)

    def test_lexical_findings_of_the_fixture(self):
        r = self.run_on(CHECK_FIX / 'asm_clobber.hip')
        # The macro ZERO_V9 holds no memory instruction: no lexical hint, but the IR check sees it.
        self.assertEqual([f for f in r['findings'] if f['rule'] == 'HIP_INLINE_ASM_MEMORY'], [])


class SnippetUnaffectedTests(unittest.TestCase):
    def test_no_asm_rule_outside_hip(self):
        from support import llvm_mc
        if not llvm_mc():
            self.skipTest('no llvm-mc')
        _, r = check_text('v_mov_b32 v7, 0\ns_endpgm\n')
        self.assertFalse([f for k in r['kernels'] for f in k['findings'] if f['rule'].startswith('ASM_')])


if __name__ == '__main__':
    unittest.main()
