"""radisa/suppress.py: rule registry, configuration, inline suppression and baseline of `radisa check`."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from support import CHECK_FIX, needs_llvm_mc

from radisa import check, memorder, suppress

PKG = Path(suppress.__file__).resolve().parent


def parse(path, *argv):
    """`radisa check path *argv` with this module's options, as the integrated CLI will parse it."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command')
    check.add_parser(sub)
    c = sub.choices['check']
    if not any('--baseline' in a.option_strings for a in c._actions):   # until check.py wires it
        suppress.add_arguments(c)
    args = p.parse_args(['check', str(path), *map(str, argv)])
    args.pass_options = {**check.pass_options(args), **suppress.options_from(args)}
    return args


def analyze(path, *argv, passes=()):
    args = parse(path, *argv)
    added_f = suppress.report_filter not in check.REPORT_FILTERS
    if added_f:
        check.REPORT_FILTERS.append(suppress.report_filter)
    added_p = [fn for fn in passes if fn not in check.KERNEL_PASSES]
    check.KERNEL_PASSES.extend(added_p)
    try:
        return check.analyze(args)
    finally:
        if added_f:
            check.REPORT_FILTERS.remove(suppress.report_filter)
        for fn in added_p:
            check.KERNEL_PASSES.remove(fn)


def rules_of(rep, name=None):
    fs = check.all_findings(rep)
    return [f for f in fs if f['rule'] == name] if name else [f['rule'] for f in fs]


class Workdir(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='radisa-suppress-'))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def copy(self, name, text=None):
        p = self.dir / name
        p.write_text(text if text is not None else (CHECK_FIX / name).read_text())
        return p


class Inline(Workdir):
    @needs_llvm_mc
    def test_line_suppression_reason_and_unused(self):
        rep = analyze(CHECK_FIX / 'suppress_inline.s', '--no-config')
        waits = rules_of(rep, 'MISSING_WAIT')
        self.assertEqual(sorted(f['loc']['line'] for f in waits), [9, 12])   # 6 suppressed, 9 has no reason
        self.assertEqual(len(rep['suppressed']), 1)
        s = rep['suppressed'][0]
        self.assertEqual((s['rule'], s['loc']['line'], s['suppression']['line'], s['suppression']['kind']),
                         ('MISSING_WAIT', 6, 5, 'inline'))
        self.assertIn('recomputed', s['suppression']['reason'])
        self.assertEqual(s['kernel'], '<snippet>')
        [nr] = rules_of(rep, 'SUPPRESSION_NO_REASON')
        self.assertEqual((nr['severity'], nr['loc']['line']), ('review', 9))
        [un] = rules_of(rep, 'UNUSED_SUPPRESSION')
        self.assertEqual((un['severity'], un['loc']['line'], un['suppression_rule']), ('review', 14, 'MISSING_ALU_WAIT'))
        self.assertEqual(un['loc']['file'], str(CHECK_FIX / 'suppress_inline.s'))   # the name the input was given
        self.assertEqual(rep['status'], 'FAIL')
        self.assertEqual(rep['suppression']['suppressed_errors'], {'MISSING_WAIT': 1})
        self.assertIn('1 suppressed inline (hidden errors: MISSING_WAIT x1)', suppress.summary_suffix(rep))

    @needs_llvm_mc
    def test_file_suppression_passes_and_lists_errors(self):
        rep = analyze(CHECK_FIX / 'suppress_file.s', '--no-config')
        self.assertEqual(rules_of(rep), [])
        self.assertEqual(rep['status'], 'PASS')
        self.assertEqual({f['suppression']['kind'] for f in rep['suppressed']}, {'file'})
        self.assertEqual(rep['suppression']['suppressed_errors'], {'MISSING_WAIT': 2})
        self.assertIn('MISSING_WAIT x2', suppress.summary_suffix(rep))

    @needs_llvm_mc
    def test_used_block_comment_suppression_is_clean(self):
        rep = analyze(CHECK_FIX / 'suppress_clean.s', '--no-config')
        self.assertEqual(rules_of(rep), [])
        self.assertEqual(rep['status'], 'PASS')
        self.assertEqual(len(rep['suppressed']), 1)

    @needs_llvm_mc
    def test_no_directives_no_change(self):
        rep = analyze(CHECK_FIX / 'missing_waits.s', '--no-config')
        self.assertEqual(rules_of(rep), ['MISSING_WAIT'] * 3)
        self.assertEqual(rep['suppressed'], [])
        self.assertEqual(suppress.summary_suffix(rep), '')
        clean = analyze(CHECK_FIX / 'clean_waits.s', '--no-config')
        self.assertEqual((rules_of(clean), clean['status']), ([], 'PASS'))

    def test_hip_comments_on_a_synthetic_report(self):
        src = self.copy('k.hip', '\n'.join([
            '#include <hip/hip_runtime.h>',
            '/* radisa: ignore[HIP_NO_LAUNCH_BOUNDS] the host clamps the block size to 64 */',
            '__global__ void a(float *p) { p[0] = 1; }',
            '__global__ void b(float *p) { p[0] = 2; } // radisa: ignore[HIP_NO_LAUNCH_BOUNDS] same',
            '',
            '__global__ void c(float *p) { p[0] = 3; }',
            '/*',
            ' * radisa: ignore[HIP_WAVE64_WIDTH] block-comment continuation line',
            ' */',
            'int x = 1; ; radisa: ignore[HIP_WAVE64_WIDTH] a semicolon is not a comment in C++', '']))
        rep = {'input': str(src), 'findings': [
            {'rule': 'HIP_NO_LAUNCH_BOUNDS', 'severity': 'info', 'pc': None, 'message': 'm',
             'loc': {'file': str(src), 'line': n, 'column': 1}} for n in (3, 4, 6)], 'kernels': []}
        suppress.report_filter(rep, parse(src, '--no-config'))
        self.assertEqual([f['loc']['line'] for f in rep['findings'] if f['rule'] == 'HIP_NO_LAUNCH_BOUNDS'], [6])
        self.assertEqual(len(rep['suppressed']), 2)
        unused = [f for f in rep['findings'] if f['rule'] == 'UNUSED_SUPPRESSION']
        self.assertEqual([f['loc']['line'] for f in unused], [8])   # line 10 is no directive at all

    def test_relative_loc_and_no_loc_findings(self):
        src = self.copy('k.s', '; radisa: ignore-file[REGISTER_SPILL] spills are accepted in this kernel\n'
                               '\ts_endpgm\n')
        rep = {'input': str(src), 'findings': [], 'kernels': [{'name': 'k', 'findings': [
            {'rule': 'REGISTER_SPILL', 'severity': 'review', 'pc': None, 'message': 'spill'}]}]}
        suppress.report_filter(rep, parse(src, '--no-config'))
        self.assertEqual(rep['kernels'][0]['findings'], [])
        self.assertEqual(rep['suppressed'][0]['kernel'], 'k')

    def test_protected_findings_are_never_suppressed(self):
        src = self.copy('k.s', '; radisa: ignore-file[DECODE,CHECK_FAILED] would hide that nothing was analysed\n')
        rep = {'input': str(src), 'findings': [
            {'rule': 'CHECK_FAILED', 'severity': 'incomplete', 'pc': None, 'message': 'x'}],
            'kernels': [{'name': 'k', 'findings': [{'rule': 'DECODE', 'severity': 'incomplete', 'pc': 4,
                                                    'message': 'y', 'loc': {'file': str(src), 'line': 1}}]}]}
        suppress.report_filter(rep, parse(src, '--no-config', '--disable', 'DECODE', '--severity', 'CHECK_FAILED=off'))
        self.assertEqual([f['rule'] for f in rep['kernels'][0]['findings']], ['DECODE'])
        self.assertIn('CHECK_FAILED', [f['rule'] for f in rep['findings']])
        self.assertEqual(rep['suppressed'], [])


class Config(Workdir):
    @needs_llvm_mc
    def test_config_file_upward_and_cli_override(self):
        (self.dir / 'sub').mkdir()
        src = self.dir / 'sub' / 'w.s'
        src.write_text((CHECK_FIX / 'missing_waits.s').read_text())
        (self.dir / '.radisa.json').write_text(json.dumps({'severity': {'MISSING_WAIT': 'warning'}}))
        rep = analyze(src)
        self.assertEqual({f['severity'] for f in rules_of(rep, 'MISSING_WAIT')}, {'review'})
        self.assertEqual(rules_of(rep, 'MISSING_WAIT')[0]['default_severity'], 'error')
        self.assertEqual(rep['status'], 'PASS')
        self.assertEqual(rep['suppression']['config'], str(self.dir / '.radisa.json'))
        rep = analyze(src, '--severity', 'MISSING_WAIT=error')
        self.assertEqual(rep['status'], 'FAIL')
        rep = analyze(src, '--severity', 'missing_wait=off')
        self.assertEqual((rules_of(rep), rep['status']), ([], 'PASS'))
        self.assertEqual(rep['suppression']['disabled'], {'MISSING_WAIT': 3})
        self.assertIn('3 disabled by configuration', suppress.summary_suffix(rep))
        rep = analyze(src, '--no-config')
        self.assertEqual(rep['status'], 'FAIL')

    @needs_llvm_mc
    def test_enable_reaches_kernel_passes(self):
        cfg = self.copy('cfg.json', json.dumps({'enable': ['SYS_STORE_WAIT']}))
        path = CHECK_FIX / 'memorder_sys_store.s'
        self.assertEqual(rules_of(analyze(path, '--no-config', passes=[memorder.kernel_pass]), 'SYS_STORE_WAIT'), [])
        on = analyze(path, '--config', cfg, passes=[memorder.kernel_pass])
        self.assertTrue(rules_of(on, 'SYS_STORE_WAIT'))
        off = analyze(path, '--config', cfg, '--disable', 'SYS_STORE_WAIT', passes=[memorder.kernel_pass])
        self.assertEqual(rules_of(off, 'SYS_STORE_WAIT'), [])
        clean = analyze(CHECK_FIX / 'memorder_sys_store_clean.s', '--enable', 'SYS_STORE_WAIT', '--no-config',
                        passes=[memorder.kernel_pass])
        self.assertEqual(rules_of(clean, 'SYS_STORE_WAIT'), [])

    def test_options_from_and_identity_are_deterministic(self):
        cfg = self.copy('c.json', json.dumps({'enable': ['b_rule', 'A_RULE'], 'disable': ['X'],
                                              'severity': {'Y': 'off', 'Z': 'note'}}))
        o = suppress.options_from(parse(self.dir / 'k.s', '--config', cfg, '--enable', 'x', '--disable', 'A_RULE'))
        self.assertEqual(o['enable'], {'B_RULE', 'X'})
        self.assertEqual(o['disable'], {'A_RULE', 'Y'})
        self.assertIsInstance(o['enable'], frozenset)
        self.assertEqual(str(o['enable']), "RuleSet(['B_RULE', 'X'])")
        self.assertEqual(check.analysis_identity(o), check.analysis_identity(dict(o)))
        self.assertTrue(memorder.enabled(o) >= {'B_RULE', 'X'})

    def test_bad_configuration(self):
        for bad in ('[1]', '{"severity": {"MISSING_WAIT": "loud"}}', '{"colour": 1}', 'not json'):
            cfg = self.copy('bad.json', bad)
            with self.assertRaises(ValueError):
                suppress.settings(parse(self.dir / 'k.s', '--config', cfg))
        with self.assertRaises(ValueError):
            suppress.settings(parse(self.dir / 'k.s', '--no-config', '--severity', 'MISSING_WAIT'))
        with self.assertRaises(ValueError):
            suppress.validate(parse(self.dir / 'k.s', '--no-config', '--baseline', self.dir / 'missing.json'))

    def test_unknown_rule_in_configuration(self):
        src = self.copy('k.s', '\ts_endpgm\n')
        rep = {'input': str(src), 'findings': [], 'kernels': []}
        suppress.report_filter(rep, parse(src, '--no-config', '--disable', 'MISING_WAIT'))
        [f] = rep['findings']
        self.assertEqual((f['rule'], f['rules']), ('CONFIG_UNKNOWN_RULE', ['MISING_WAIT']))
        rep = {'input': str(src), 'findings': [], 'kernels': []}
        suppress.report_filter(rep, parse(src, '--no-config', '--disable', 'MISSING_WAIT'))
        self.assertEqual(rep['findings'], [])


class Baseline(Workdir):
    @needs_llvm_mc
    def test_baseline_survives_unrelated_edits(self):
        src = self.copy('missing_waits.s')
        base = self.dir / 'baseline.json'
        rep = analyze(src, '--no-config', '--write-baseline', base)
        self.assertEqual((rep['status'], len(rep['baselined'])), ('PASS', 3))
        data = json.loads(base.read_text())
        self.assertEqual(data['schema'], suppress.SCHEMA)
        self.assertEqual(len({e['fingerprint'] for e in data['entries']}), 3)
        # Unrelated edits: lines and pcs move, the findings stay hidden.
        src.write_text('; header\n\ts_nop 0\n\ts_mov_b32 s9, 0\n' + src.read_text())
        rep = analyze(src, '--no-config', '--baseline', base)
        self.assertEqual((rules_of(rep), rep['status'], len(rep['baselined'])), ([], 'PASS', 3))
        self.assertIn('3 in baseline (hidden errors: MISSING_WAIT x3)', suppress.summary_suffix(rep))
        # A new hazard is reported; the old ones stay hidden.
        src.write_text(src.read_text().replace('  s_endpgm', '  global_load_b32 v9, v[2:3], off\n'
                                               '  v_mov_b32 v10, v9\n  s_endpgm'))
        rep = analyze(src, '--no-config', '--baseline', base)
        [new] = rules_of(rep, 'MISSING_WAIT')
        self.assertIn('v9', new['message'])
        self.assertEqual(rep['status'], 'FAIL')

    @needs_llvm_mc
    def test_stale_entries_and_config_baseline(self):
        src = self.copy('missing_waits.s')
        base = self.dir / 'b.json'
        analyze(src, '--no-config', '--write-baseline', base)
        (self.dir / '.radisa.json').write_text(json.dumps({'baseline': 'b.json'}))
        src.write_text(src.read_text().replace('  s_add_co_u32 s6, s4, 1\n', ''))
        rep = analyze(src)
        [stale] = rules_of(rep, 'BASELINE_STALE')
        self.assertEqual((stale['severity'], len(stale['stale'])), ('info', 1))
        self.assertEqual(rep['status'], 'PASS')
        rep = analyze(src, '--kernel', 'snippet')   # a partial run says nothing about the rest
        self.assertEqual(rules_of(rep, 'BASELINE_STALE'), [])

    def test_fingerprint_components(self):
        def rep(*fs, kernel='k'):
            return {'input': None, 'findings': [], 'kernels': [{'name': kernel, 'findings': list(fs)}]}

        def f(pc, text='v_add_f32_e32 v1, v2, v3', rule='MISSING_WAIT', prod=None):
            return {'rule': rule, 'severity': 'error', 'pc': pc, 'message': f'at {pc}', 'instruction': text,
                    **({'producer_instruction': prod} if prod else {})}
        fp = lambda r: [x[0] for x in suppress.fingerprints(r)]
        a = fp(rep(f(4), f(8)))
        self.assertEqual(len(set(a)), 2)                                        # occurrence index
        self.assertEqual(a, fp(rep(f(40), f(80))))                              # pcs do not matter
        self.assertEqual(a[:1], fp(rep(f(4, 'V_ADD_F32_e32  v1,v2 ,  v3'))))    # normalised text
        self.assertNotEqual(a[:1], fp(rep(f(4), kernel='other')))
        self.assertNotEqual(a[:1], fp(rep(f(4, rule='MISSING_ALU_WAIT'))))
        self.assertNotEqual(a[:1], fp(rep(f(4, prod='global_load_b32 v1, v[2:3], off'))))
        self.assertEqual(fp(rep({'rule': 'DECODE', 'severity': 'incomplete', 'pc': 0, 'message': 'x'})), [])


class Registry(unittest.TestCase):
    PATTERNS = (r"""['"]rule['"]\s*:\s*['"]([A-Z][A-Z0-9_]*[A-Z0-9])['"]""",
                r"""\b_?finding\(\s*['"]([A-Z][A-Z0-9_]*[A-Z0-9])['"]""",
                r"""\badd\(\s*['"]([A-Z][A-Z0-9_]*[A-Z0-9])['"]""",
                r"""\bRULE\s*=\s*['"]([A-Z][A-Z0-9_]*[A-Z0-9])['"]""",
                r"""\brule\s*=\s*['"]([A-Z][A-Z0-9_]*[A-Z0-9])['"]""",
                r"""\b_hygiene\(\s*['"]([A-Z][A-Z0-9_]*[A-Z0-9])['"]""",
                # batch.py: synthetic_report(path, 'RULE', ...)
                r"""\bsynthetic_report\([^,()]*(?:\([^()]*\))?[^,()]*,\s*['"]([A-Z][A-Z0-9_]*[A-Z0-9])['"]""")

    @classmethod
    def emitted(cls) -> dict[str, set]:
        """Every rule id literal in the radisa package: {id: {files}}. vopd.validate() problem names
        are reported as 'VOPD_' + name (rules.audit_kernel)."""
        out = {}
        for path in sorted(PKG.glob('*.py')):
            text = path.read_text()
            if path.name == 'vopd.py':
                ids = {'VOPD_' + n for n in re.findall(r"problems\.append\(\(\s*'([A-Z][A-Z0-9_]*)'", text)}
                ids |= {'VOPD_' + n for m in re.findall(r"\brule\s*=\s*([^\n]+)", text)
                        for n in re.findall(r"'([A-Z][A-Z0-9_]*)'", m)}
            else:
                ids = {m for p in cls.PATTERNS for m in re.findall(p, text)}
            if path.name == 'suppress.py':
                ids -= {'RULE'}   # docstrings
            for i in ids:
                out.setdefault(i, set()).add(path.name)
        return out

    def test_scanner_finds_known_rules(self):
        found = self.emitted()
        for r in ('MISSING_WAIT', 'MISSING_ALU_WAIT', 'ALU_WAIT_TTMP', 'VOPD_BANK', 'VOPD_RAW_DPMACC', 'VOPD_ANTIDEP',
                  'HIP_WAVE64_WIDTH', 'CFG_FALLTHROUGH', 'NO_KERNELS', 'UNUSED_SUPPRESSION', 'SUPPRESSION_NO_REASON',
                  'BASELINE_STALE', 'CONFIG_UNKNOWN_RULE'):
            self.assertIn(r, found)

    def test_every_emitted_rule_is_registered(self):
        missing = {r: sorted(f) for r, f in self.emitted().items() if r not in suppress.RULES}
        self.assertEqual(missing, {}, 'add these rule ids to radisa/suppress.py RULES')

    def test_every_registered_rule_is_emitted(self):
        stale = sorted(set(suppress.RULES) - set(self.emitted()))
        self.assertEqual(stale, [], 'RULES lists ids no code emits')

    def test_entries_are_complete(self):
        for r, e in suppress.RULES.items():
            self.assertRegex(r, r'^[A-Z][A-Z0-9_]*$')
            self.assertIn(e['severity'], ('error', 'review', 'info', 'incomplete'), r)
            for k in ('summary', 'basis', 'since'):
                self.assertTrue(e[k], (r, k))

    def test_rules_command(self):
        def run(**kw):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = suppress.rules_command(argparse.Namespace(explain=None, json=False, **kw) if 'explain' not in kw
                                              else argparse.Namespace(json=False, **kw))
            return code, buf.getvalue()
        code, out = run()
        self.assertEqual(code, 0)
        self.assertIn('MISSING_WAIT', out)
        self.assertIn(f'{len(suppress.RULES)} rules', out)
        code, out = run(explain='missing_wait')
        self.assertEqual(code, 0)
        self.assertIn('§5.7.1', out)
        self.assertIn('radisa: ignore[MISSING_WAIT]', out)
        self.assertEqual(run(explain='NOPE')[0], 2)
        self.assertIn('cannot be re-graded', run(explain='DECODE')[1])
        p = argparse.ArgumentParser()
        suppress.add_rules_parser(p.add_subparsers(dest='command'))
        a = p.parse_args(['rules', '--json'])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(a.fn(a), 0)
        self.assertEqual(set(json.loads(buf.getvalue())), set(suppress.RULES))


if __name__ == '__main__':
    unittest.main()
