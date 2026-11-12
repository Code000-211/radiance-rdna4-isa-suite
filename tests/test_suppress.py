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


def mw(line=None, pc=4, text='v_add_f32_e32 v4, v1, v4', file=None, severity='error'):
    f = {'rule': 'MISSING_WAIT', 'severity': severity, 'pc': pc, 'message': f'at {pc}', 'instruction': text}
    if line is not None:
        f['loc'] = {'file': str(file), 'line': line, 'column': 1}
    return f


def snippet_report(src, *fs):
    return {'input': str(src), 'findings': [], 'kernels': [{'name': '<snippet>', 'findings': list(fs)}]}


class Review(Workdir):
    """Regressions for the adversarial review of this feature (scratchpad review-suppress)."""

    # 1. a baseline is tied to its input file
    def test_baseline_of_one_file_does_not_hide_another(self):
        a, b = self.copy('one.s', '\ts_endpgm\n'), self.copy('two.s', '\ts_endpgm\n')
        base = self.dir / 'b.json'
        rep = snippet_report(a, mw(pc=4), mw(pc=8, text='v_mov_b32 v1, v2'))
        suppress.report_filter(rep, parse(a, '--no-config', '--write-baseline', base))
        self.assertEqual(len(rep['baselined']), 2)
        # The same fingerprint in another file is a new finding there; nothing is stale there.
        rep = snippet_report(b, mw(pc=4))
        suppress.report_filter(rep, parse(b, '--no-config', '--baseline', base))
        self.assertEqual((len(rep['kernels'][0]['findings']), rep['baselined'], rules_of(rep, 'BASELINE_STALE')),
                         (1, [], []))
        # Writing b's baseline into the same file keeps a's entries; each file passes with it.
        suppress.report_filter(snippet_report(b, mw(pc=4)), parse(b, '--no-config', '--write-baseline', base))
        data = json.loads(base.read_text())
        self.assertEqual(sorted(e['input'] for e in data['entries']), ['one.s', 'one.s', 'two.s'])
        self.assertEqual((data['inputs'], data['input']), (['one.s', 'two.s'], None))
        for src, fs in ((a, [mw(pc=4), mw(pc=8, text='v_mov_b32 v1, v2')]), (b, [mw(pc=4)])):
            (self.dir / '.radisa.json').write_text(json.dumps({'baseline': 'b.json'}))   # repo-wide baseline
            rep = snippet_report(src, *fs)
            suppress.report_filter(rep, parse(src))
            self.assertEqual((rep['kernels'][0]['findings'], len(rep['baselined']), rules_of(rep, 'BASELINE_STALE')),
                             ([], len(fs), []), src)

    @needs_llvm_mc
    def test_baseline_per_file_real_check(self):
        one, two = self.copy('missing_waits.s'), self.copy('two.s', (CHECK_FIX / 'suppress_trailing.s').read_text())
        base = self.dir / 'one.json'
        analyze(one, '--no-config', '--write-baseline', base)
        rep = analyze(two, '--no-config', '--baseline', base)
        self.assertEqual((rep['status'], rep['baselined'], rules_of(rep, 'BASELINE_STALE')), ('FAIL', [], []))

    def test_legacy_single_file_baseline(self):
        a, b = self.copy('one.s', '\ts_endpgm\n'), self.copy('two.s', '\ts_endpgm\n')
        [(fp, _, _)] = suppress.fingerprints(snippet_report(a, mw()))
        old = self.dir / 'old.json'
        old.write_text(json.dumps({'schema': suppress.SCHEMA, 'input': str(a), 'entries': [
            {'fingerprint': fp, 'rule': 'MISSING_WAIT', 'severity': 'error', 'kernel': '<snippet>'}]}))
        self.assertEqual(list(suppress.read_baseline(old, a)), [fp])
        self.assertEqual(suppress.read_baseline(old, b), {})
        self.assertEqual(list(suppress.read_baseline(old)), [fp])
        # Rewriting another file's entries converts the old ones to per-entry 'input'.
        suppress.write_baseline(old, [], {'input': str(b)})
        self.assertEqual([e['input'] for e in json.loads(old.read_text())['entries']], ['one.s'])

    # 2. (suppress side) concurrent writers into one file all land; partial runs keep other kernels
    def test_concurrent_writers_merge(self):
        import threading
        base = self.dir / 'b.json'
        srcs = [self.copy(f'f{i}.s', '\ts_endpgm\n') for i in range(8)]

        def write(src):
            rep = snippet_report(src, mw())
            for k, lst in suppress._all_lists(rep):
                for f in lst:
                    f['_kernel'] = k
            suppress.write_baseline(base, suppress.fingerprints(rep), rep)
        ts = [threading.Thread(target=write, args=(s,)) for s in srcs]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(sorted(e['input'] for e in json.loads(base.read_text())['entries']),
                         sorted(s.name for s in srcs))
        self.assertEqual(list(self.dir.glob('.b.json.*')), [])   # no temporary file left

    def test_kernel_run_replaces_only_its_kernels(self):
        src = self.copy('k.s', '\ts_endpgm\n')
        base = self.dir / 'b.json'
        rep = {'input': str(src), 'findings': [], 'kernels': [{'name': 'ka', 'findings': [mw()]},
                                                                {'name': 'kb', 'findings': [mw()]}]}
        suppress.report_filter(rep, parse(src, '--no-config', '--write-baseline', base))
        rep = {'input': str(src), 'findings': [], 'kernels': [{'name': 'ka', 'findings': []}]}
        suppress.report_filter(rep, parse(src, '--no-config', '--kernel', 'ka', '--write-baseline', base))
        self.assertEqual([e['kernel'] for e in json.loads(base.read_text())['entries']], ['kb'])

    def test_write_baseline_refuses_to_overwrite_other_files(self):
        src = self.copy('k.s', '\ts_endpgm\n')
        other = self.copy('notes.json', '{"mine": 1}')
        with self.assertRaises(ValueError):
            suppress.write_baseline(other, [], {'input': str(src)})
        self.assertEqual(other.read_text(), '{"mine": 1}')

    # 3. kernel-form .s: findings without a line
    def test_unused_message_for_findings_without_line(self):
        src = self.copy('k.s', '\tglobal_load_b32 v1, v[2:3], off\n\t; radisa: ignore[MISSING_WAIT] fine\n'
                               '\tv_add_f32 v4, v1, v4\n\ts_endpgm\n')
        rep = snippet_report(src, mw())   # no loc
        suppress.report_filter(rep, parse(src, '--no-config'))
        [un] = rules_of(rep, 'UNUSED_SUPPRESSION')
        self.assertIn('carry no source line', un['message'])
        self.assertNotIn('no MISSING_WAIT finding', un['message'])
        rep = snippet_report(src, mw(line=3, file=src))   # with a line it is suppressed
        suppress.report_filter(rep, parse(src, '--no-config'))
        self.assertEqual((rules_of(rep), len(rep['suppressed'])), ([], 1))

    @needs_llvm_mc
    def test_kernel_form_assembly(self):
        rep = analyze(CHECK_FIX / 'suppress_kernel.s', '--no-config')
        if rep['suppressed']:   # check.py maps kernel-form .s lines (NEEDS CHANGE ELSEWHERE in the review)
            self.assertEqual((rules_of(rep), rep['status']), ([], 'PASS'))
        else:
            [w] = rules_of(rep, 'MISSING_WAIT')
            self.assertNotIn('loc', w)
            [un] = rules_of(rep, 'UNUSED_SUPPRESSION')
            self.assertIn('carry no source line', un['message'])

    # 4. /* */ directives: standalone and with the reason on the next line
    def test_block_comment_directives(self):
        lines = (CHECK_FIX / 'suppress_block.s').read_text().splitlines()
        ds = suppress.parse_directives(lines, 'x.s')
        self.assertEqual([(d['line'], d['target'], d['reason']) for d in ds],
                         [(4, 5, 'v4 is discarded later'), (8, 10, 'v6 is dead')])
        src = self.copy('suppress_block.s')
        rep = snippet_report(src, mw(line=5, file=src), mw(line=10, pc=32, text='v_add_f32_e32 v6, v5, v5', file=src))
        suppress.report_filter(rep, parse(src, '--no-config'))
        self.assertEqual((rules_of(rep), len(rep['suppressed'])), ([], 2))
        # Inside a block comment, a directive line need not start with '*'.
        ds = suppress.parse_directives(['/*', 'radisa: ignore[X] why', '*/', 'code();'], 'a.hip')
        self.assertEqual([(d['line'], d['target']) for d in ds], [(2, 4)])
        # Not in a comment: a string literal.
        self.assertEqual(suppress.parse_directives(['s = "// radisa: ignore[X] no";'], 'a.hip'), [])

    @needs_llvm_mc
    def test_block_comment_fixture(self):
        rep = analyze(CHECK_FIX / 'suppress_block.s', '--no-config')
        self.assertEqual(rules_of(rep, 'SUPPRESSION_NO_REASON'), [])
        if len(rep['suppressed']) == 2:   # line mapping fixed in check.py / hipsrc.py
            self.assertEqual((rules_of(rep), rep['status']), ([], 'PASS'))
        else:
            self.assertTrue(all('carry no source line' in f['message'] for f in rules_of(rep, 'UNUSED_SUPPRESSION')))

    # 6. errors hidden by configuration are named
    def test_configuration_hidden_errors_in_summary(self):
        src = self.copy('k.s', '\ts_endpgm\n')
        rep = snippet_report(src, mw(), mw(pc=8))
        suppress.report_filter(rep, parse(src, '--no-config', '--disable', 'MISSING_WAIT'))
        self.assertIn('2 disabled by configuration (hidden errors: MISSING_WAIT x2)', suppress.summary_suffix(rep))
        rep = snippet_report(src, mw(), mw(pc=8))
        suppress.report_filter(rep, parse(src, '--no-config', '--severity', 'MISSING_WAIT=note'))
        self.assertIn('2 re-graded by configuration (hidden errors: MISSING_WAIT x2)', suppress.summary_suffix(rep))
        rep = snippet_report(src, mw(severity='review'))   # raising a warning to error hides nothing
        suppress.report_filter(rep, parse(src, '--no-config', '--severity', 'MISSING_WAIT=error'))
        self.assertNotIn('hidden errors', suppress.summary_suffix(rep))
        # A re-graded error that is then suppressed inline is counted once.
        src2 = self.copy('k2.s', '\t; radisa: ignore[MISSING_WAIT] ok\n\ts_nop 0\n')
        rep = snippet_report(src2, mw(line=2, file=src2))
        suppress.report_filter(rep, parse(src2, '--no-config', '--severity', 'MISSING_WAIT=warning'))
        self.assertIn('(hidden errors: MISSING_WAIT x1)', suppress.summary_suffix(rep))

    # 7. (suppress side) hidden findings for SARIF
    def test_sarif_suppressions(self):
        src = self.copy('k.s', '\t; radisa: ignore[MISSING_WAIT] reviewed\n\ts_nop 0\n')
        rep = snippet_report(src, mw(line=2, file=src), mw(pc=8, text='v_mov_b32 v1, v2'))
        suppress.report_filter(rep, parse(src, '--no-config', '--write-baseline', self.dir / 'b.json'))
        kinds = sorted(suppress.sarif_suppression(f)['kind'] for f in suppress.hidden_findings(rep))
        self.assertEqual(kinds, ['external', 'inSource'])
        [ins] = [suppress.sarif_suppression(f) for f in rep['suppressed']]
        self.assertEqual((ins['justification'], ins['status']), ('reviewed', 'accepted'))

    # 8. a trailing directive covers its own line only
    def test_trailing_directive_covers_own_line(self):
        ds = suppress.parse_directives(['\ts_nop 0 ; radisa: ignore[MISSING_WAIT] x', '\tv_add_f32 v4, v1, v4'], 'a.s')
        self.assertEqual(ds[0]['covers'], {1})
        ds = suppress.parse_directives(['; radisa: ignore[A] x', '; radisa: ignore[B] y', '; other', 'code', 'more'],
                                       'a.s')
        self.assertEqual([d['covers'] for d in ds], [{1, 4}, {2, 4}])
        ds = suppress.parse_directives(['; radisa: ignore[A] x', '', 'code'], 'a.s')   # not over a blank line
        self.assertEqual(ds[0]['covers'], {1})

    @needs_llvm_mc
    def test_trailing_fixture(self):
        rep = analyze(CHECK_FIX / 'suppress_trailing.s', '--no-config')
        [w] = rules_of(rep, 'MISSING_WAIT')
        self.assertEqual((w['loc']['line'], rep['status'], rep['suppressed']), (5, 'FAIL', []))
        [un] = rules_of(rep, 'UNUSED_SUPPRESSION')
        self.assertEqual(un['loc']['line'], 4)

    # 9. header directives are not reported unused by a TU that does not need them
    def test_header_directive_not_reported_unused(self):
        (self.dir / 'hdr').mkdir()
        hdr = self.dir / 'hdr' / 'util.h'
        hdr.write_text('int f();\n// radisa: ignore[MISSING_WAIT] used by b.hip only\nint g();\n'
                       '// radisa: ignore[MISSING_WAIT]\nint h();\n')
        src = self.copy('a.hip', '#include "hdr/util.h"\n')
        rep = {'input': str(src), 'kernels': [], 'findings': [
            {'rule': 'MISSING_ALU_WAIT', 'severity': 'error', 'pc': None, 'message': 'm',
             'loc': {'file': str(hdr), 'line': 1, 'column': 1}}]}
        suppress.report_filter(rep, parse(src, '--no-config'))
        self.assertEqual(rules_of(rep, 'UNUSED_SUPPRESSION'), [])
        [nr] = rules_of(rep, 'SUPPRESSION_NO_REASON')   # a missing reason is wrong in every TU
        self.assertEqual(nr['loc']['line'], 4)


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
