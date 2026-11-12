"""hwprobe/: hardware probes for the hazard rules (generated HIP pairs, host runner, run script).

What runs here (no GPU, LLVM/clang 18): generation is deterministic and the checked-in cases are
current; the host-predicted results are what each probe computes; where clang compiles a case for
gfx1201 with -nogpuinc, `radisa check` reports exactly the expected rule on the `_hazard` kernel and
no error on the `_fixed` kernel; the host runner builds against hwprobe/mock (g++) and counts, reports
and exits as documented. Nothing here runs a probe on hardware.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from support import ROOT, hip_compiles, require_llvm, run_check

sys.path.insert(0, str(ROOT / 'hwprobe'))
import generate  # noqa: E402

HW = ROOT / 'hwprobe'
CASES = {c['name']: c for c in generate.CASES}


def gfx1201_compiler():
    for c in (os.environ.get('RADISA_TEST_CXX'), '/usr/lib/llvm-20/bin/clang++', '/opt/rocm/llvm/bin/clang++',
              shutil.which('clang++')):
        if c and Path(c).exists():
            return c
    return None


def _skip_unless(why: str, what: str):
    """unittest skip with the reason; never skipped with RAD_ISA_REQUIRE_LLVM=1 (the test then fails)."""
    if why and not require_llvm():
        raise unittest.SkipTest(f'{what}: {why}')


class GenerationTests(unittest.TestCase):
    def test_deterministic_and_checked_in(self):
        first, second = generate.sources(), generate.sources()
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), sorted([f'{n}.hip' for n in CASES] + ['manifest.json']))
        for name, text in first.items():
            self.assertEqual((HW / 'cases' / name).read_text(), text, f'{name} is stale: run python3 hwprobe/generate.py')
        self.assertEqual(generate.main(['--check']), 0)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(generate.main(['--out', d]), 0)
            self.assertEqual({p.name: p.read_text() for p in Path(d).iterdir()}, first)

    def test_check_reports_stale_files(self):
        with tempfile.TemporaryDirectory() as d:
            generate.main(['--out', d])
            Path(d, 'wmma_d_to_a.hip').write_text('// edited\n')
            with redirect_stderr(StringIO()) as err:
                self.assertEqual(generate.main(['--out', d, '--check']), 1)
            self.assertIn('wmma_d_to_a.hip', err.getvalue())

    def test_manifest(self):
        m = json.loads((HW / 'cases' / 'manifest.json').read_text())
        self.assertFalse(m['hardware_validated'])
        self.assertEqual((m['target'], m['wave'], m['block']), ('gfx1201', 32, 256))
        rules = {c['name']: c['variants']['hazard']['expect'] for c in m['cases']}
        self.assertEqual(rules, {
            'ds_load_wait': 'MISSING_WAIT', 'global_load_wait': 'MISSING_WAIT',
            'lds_store_barrier': 'LDS_STORE_BEFORE_BARRIER', 'wmma_d_to_a': 'WMMA_D_TO_AB',
            'cmpx_permlane': 'PERMLANE_AFTER_CMPX', 'alu_wait_carry': 'MISSING_ALU_WAIT',
            'alu_wait_ttmp': 'ALU_WAIT_TTMP', 'd16_other_half': 'MISSING_WAIT'})
        for c in m['cases']:
            self.assertEqual(c['variants']['fixed']['expect'], 'clean')
            self.assertEqual(c['variants']['hazard']['severity'], 'review' if c['name'] == 'alu_wait_ttmp' else 'error')
            self.assertTrue((HW / c['file']).is_file())

    def test_data_is_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate.write_data(Path(a), 'H', grid=2)
            generate.write_data(Path(b), 'H', grid=2)
            for p in sorted(Path(a).iterdir()):
                text = p.read_bytes().replace(a.encode(), b'DIR')
                self.assertEqual(text, Path(b, p.name).read_bytes().replace(b.encode(), b'DIR'), p.name)
            rows = [l.split('\t') for l in Path(a, 'cases.tsv').read_text().splitlines() if not l.startswith('#')]
            self.assertEqual(len(rows), 2 * len(CASES))
            self.assertTrue(all(len(r) == 11 for r in rows))

    def test_expected_results(self):
        words = lambda name: generate.expected(CASES[name], generate.inputs(CASES[name], grid=2))
        inp = lambda name: generate.inputs(CASES[name], grid=2)
        x = inp('ds_load_wait')
        self.assertEqual(words('ds_load_wait')[0], (x[32] + 1) & 0xFFFFFFFF)
        self.assertEqual(words('ds_load_wait')[255], (x[31] + 1) & 0xFFFFFFFF)       # wraps within the workgroup
        self.assertEqual(words('lds_store_barrier')[256 + 230], inp('lds_store_barrier')[256 + 6])
        self.assertEqual(set(words('wmma_d_to_a')), {struct.unpack('<I', struct.pack('<f', 128.0))[0]})
        x, w = inp('cmpx_permlane'), words('cmpx_permlane')
        self.assertEqual((w[32], w[47], w[48], w[63]), (x[47], x[32], 0xDEAD, 0xDEAD))
        x, w = inp('alu_wait_carry'), words('alu_wait_carry')
        for i in range(8):
            s = x[i] + 0x80000000
            self.assertEqual((w[2 * i], w[2 * i + 1]), (s & 0xFFFFFFFF, i + (s >> 32)))
        x, w = inp('d16_other_half'), words('d16_other_half')
        self.assertEqual((w[0], w[1]), (0xA5A50000 | (x[0] & 0xFFFF), x[0]))
        # Both carries occur, so a stale carry is visible.
        self.assertEqual({(v + 0x80000000) >> 32 for v in inp('alu_wait_carry')}, {0, 1})


class CaseLintTests(unittest.TestCase):
    """`radisa check` on every generated case: the rule on `_hazard`, no error on `_fixed`."""
    cache = None

    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def lint(self, name):
        cxx = gfx1201_compiler()
        src = HW / 'cases' / f'{name}.hip'
        _skip_unless('' if cxx else 'no HIP-capable clang', 'needs clang++')
        _skip_unless(hip_compiles(cxx, str(src)), f'needs a clang that compiles {src.name} for gfx1201')
        code, report = run_check(src, '--cxx', cxx, '--flag', '-nogpuinc -nogpulib', '--cache-dir',
                                 os.path.join(self.cache.name, name))
        self.assertNotEqual(report['status'], 'INCOMPLETE', report['findings'])
        return code, {k['name']: k['findings'] for k in report['kernels']}

    def _case(self, name):
        case = CASES[name]
        code, kernels = self.lint(name)
        hazard, fixed = kernels[f'{name}_hazard'], kernels[f'{name}_fixed']
        reported = {(f['rule'], f['severity']) for f in hazard if f['severity'] in ('error', 'review')}
        severity = 'review' if case['rule'] in generate.REVIEW_RULES else 'error'
        self.assertEqual(reported, {(case['rule'], severity)}, hazard)
        self.assertTrue(all(f.get('loc') for f in hazard if f['rule'] == case['rule']))
        self.assertEqual([f for f in fixed if f['severity'] == 'error' or f['rule'] == case['rule']], [])
        self.assertEqual(code, 1 if severity == 'error' else 0)


for _name in CASES:
    setattr(CaseLintTests, f'test_{_name}', lambda self, n=_name: self._case(n))


class RunnerTests(unittest.TestCase):
    """host_runner.cpp against hwprobe/mock (no GPU): parsing, comparison, JSON, exit codes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name, 'host_runner')
        gxx = shutil.which('g++') or shutil.which('c++')
        if not gxx:
            raise unittest.SkipTest('no host C++ compiler to build host_runner.cpp against hwprobe/mock')
        p = subprocess.run([gxx, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1', '-I', str(HW / 'mock'),
                            str(HW / 'host_runner.cpp'), '-o', str(cls.exe)], capture_output=True, text=True)
        if p.returncode:
            raise AssertionError(p.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_runner(self, d, *extra, env=None):
        res = Path(d, 'results.json')
        p = subprocess.run([str(self.exe), str(Path(d, 'data', 'cases.tsv')), '3', str(res), *extra],
                           capture_output=True, text=True, env={**os.environ, **(env or {})})
        return p.returncode, (json.loads(res.read_text()) if res.exists() else None), p.stderr

    def setup_case(self, d, name, hazard_out, fixed_out):
        generate.write_data(Path(d, 'data'), d, grid=1)
        Path(d, f'{name}.hsaco').write_bytes(b'not a code object (mock)')
        for variant, words in (('hazard', hazard_out), ('fixed', fixed_out)):
            if words is not None:
                Path(d, f'{name}.hsaco.{name}_{variant}.mock').write_bytes(struct.pack(f'<{len(words)}I', *words))
        return generate.expected(CASES[name], generate.inputs(CASES[name], grid=1))

    def test_counts_and_verdicts(self):
        with tempfile.TemporaryDirectory() as d:
            want = generate.expected(CASES['wmma_d_to_a'], generate.inputs(CASES['wmma_d_to_a'], grid=1))
            stale = list(want)
            stale[5] = 0
            self.setup_case(d, 'wmma_d_to_a', stale, want)
            code, r, err = self.run_runner(d, 'wmma')
            self.assertEqual(code, 0, err)
            by = {x['variant']: x for x in r['results']}
            self.assertEqual((by['hazard']['mismatched_words'], by['hazard']['mismatching_launches']), (3, 3))
            self.assertEqual(by['hazard']['first_mismatch'], {'launch': 0, 'index': 5, 'got': 0, 'want': want[5]})
            self.assertEqual(by['hazard']['verdict'], 'hazard_observed')
            self.assertEqual((by['fixed']['mismatched_words'], by['fixed']['verdict']), (0, 'ok'))
            self.assertTrue(r['fixed_variants_ok'])

    def test_fixed_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            want = self.setup_case(d, 'd16_other_half', None, None)
            # Nothing written: the output keeps the fill pattern; every word of both variants differs.
            code, r, err = self.run_runner(d, 'd16_other_half')
            self.assertEqual(code, 1, err)
            by = {x['variant']: x for x in r['results']}
            self.assertEqual(by['fixed']['verdict'], 'FIXED_VARIANT_MISMATCH')
            self.assertEqual(by['fixed']['mismatched_words'], 3 * len(want))
            self.assertEqual(by['hazard']['verdict'], 'hazard_observed')
            self.assertFalse(r['fixed_variants_ok'])

    def test_wrong_device_is_a_setup_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.setup_case(d, 'wmma_d_to_a', None, None)
            code, r, err = self.run_runner(d, 'wmma', env={'HWPROBE_MOCK_ARCH': 'gfx1100'})
            self.assertEqual((code, r), (2, None))
            self.assertIn('not gfx1201', err)

    def test_run_script_parses(self):
        self.assertEqual(subprocess.run(['bash', '-n', str(HW / 'run_probes.sh')]).returncode, 0)


if __name__ == '__main__':
    unittest.main()
