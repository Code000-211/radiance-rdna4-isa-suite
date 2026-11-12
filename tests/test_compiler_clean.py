"""Differential test: code the compiler scheduled itself must not fail `radisa check`.

Every HIP fixture documented as clean (fixtures/check/clean_*.hip, or a .hip whose header says
"Must pass") is compiled for gfx1201 with the available clang and checked; any 'error' or
'incomplete' finding is a false positive of the linter (or a compiler bug worth reporting).

A fixture the compiler cannot build (clang 18 lacks e.g. the 3-argument __builtin_amdgcn_fence)
is skipped with the compiler's reason, unless RAD_ISA_REQUIRE_LLVM=1.

A fixture that is wave32-only by construction (e.g. it calls a `_w32` WMMA builtin, which clang
rejects with -mwavefrontsize64) says so with a `radisa-test: wave32-only (reason)` comment and is
left out of the wave64 run. Nothing else is skipped there: a compile error of any other fixture
is a failure.

Older upstream LLVM does not yet fix every hazard radisa checks, and then its output is genuinely
hazardous. FIRST_FIXED_IN lists, per rule, the first upstream LLVM release whose AMDGPU backend
inserts the fix; for an upstream clang older than that, findings of the rule are left out of the
assertion (and counted in the test output). AMD clang (ROCm) is always checked in full.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from support import CHECK_FIX, hip_compiles, require_llvm, run_check

# llvmorg-19.1.0 GCNHazardRecognizer.cpp has neither hasVALUReadSGPRHazard nor
# fixVALUReadSGPRHazard; llvmorg-20.1.0 has both (moved to AMDGPUWaitSGPRHazards.cpp in 21.1.0).
FIRST_FIXED_IN = {'MISSING_ALU_WAIT': 20, 'ALU_WAIT_TTMP': 20}
CONFIGS = (('-O3', ()), ('-O3', ('--no-incremental',)), ('-O1', ('--no-incremental',)), ('-O0', ('--no-incremental',)))


def compiler() -> str | None:
    for c in (os.environ.get('RADISA_TEST_CXX'), '/opt/rocm/llvm/bin/clang++', '/usr/lib/llvm-20/bin/clang++',
              shutil.which('clang++')):
        if c and Path(c).exists():
            return c
    return None


def compiler_identity(cxx: str) -> tuple[bool, int | None]:
    """(is AMD clang, major version) from `cxx --version`."""
    out = subprocess.run([cxx, '--version'], capture_output=True, text=True, check=False).stdout
    first = out.splitlines()[0] if out else ''
    m = re.search(r'clang version (\d+)\.', first)
    return 'AMD' in first, int(m.group(1)) if m else None


def clean_fixtures() -> list[Path]:
    out = []
    for p in sorted(CHECK_FIX.glob('*.hip')):
        head = '\n'.join(p.read_text().splitlines()[:3])
        if p.name.startswith('clean_') or re.search(r'\bMust pass\b', head):
            out.append(p)
    return out


WAVE32_ONLY = re.compile(r'^\s*//\s*radisa-test:\s*wave32-only\s*\((?P<why>[^)]+)\)', re.M)


def wave32_only(src: Path) -> str:
    """The reason the fixture gives for being wave32-only, or ''."""
    m = WAVE32_ONLY.search(src.read_text())
    return m['why'].strip() if m else ''


def tolerated(cxx: str) -> set[str]:
    amd, major = compiler_identity(cxx)
    if amd or major is None:
        return set()
    return {rule for rule, first in FIRST_FIXED_IN.items() if major < first}


def findings(report: dict) -> list[dict]:
    return list(report.get('findings', [])) + [f for k in report.get('kernels', []) for f in k.get('findings', [])]


class CompilerClean(unittest.TestCase):
    def test_fixture_list(self):
        names = [p.name for p in clean_fixtures()]
        self.assertIn('clean_compiled.hip', names)
        self.assertIn('clean_compiler_basic.hip', names)
        self.assertNotIn('barrier_no_fence.hip', names)   # a fixture that must fail

    def test_wave32_only_marker(self):
        # clang 24 rejects the _w32 WMMA builtin of clean_compiled.hip with -mwavefrontsize64.
        marked = {p.name for p in clean_fixtures() if wave32_only(p)}
        self.assertEqual(marked, {'clean_compiled.hip'})
        for p in clean_fixtures():   # every fixture using a _w32 builtin carries the marker
            if re.search(r'__builtin_amdgcn_\w*_w32(?:_|\b)', p.read_text()):
                self.assertIn(p.name, marked)
        self.assertEqual(wave32_only(CHECK_FIX / 'clean_compiler_basic.hip'), '')

    def test_clean_fixtures_have_no_errors(self):
        cxx = compiler()
        if not cxx:
            if require_llvm():
                self.fail('no clang++ for gfx1201')
            self.skipTest('no clang++ (set RADISA_TEST_CXX)')
        skip = tolerated(cxx)
        checked = 0
        for src in clean_fixtures():
            with self.subTest(fixture=src.name):
                why = hip_compiles(cxx, src)
                if why:
                    if require_llvm():
                        self.fail(f'{cxx} cannot build {src.name}: {why}')
                    self.skipTest(f'{cxx} cannot build {src.name}: {why}')
                for opt, mode in CONFIGS:
                    with self.subTest(fixture=src.name, opt=opt, mode=mode), tempfile.TemporaryDirectory() as cache:
                        code, report = run_check(src, '--cxx', cxx, f'--flag=-nogpuinc -nogpulib {opt}',
                                                 '--cache-dir', cache, *mode)
                        self.assertTrue(report['kernels'], 'no kernel was checked')
                        bad = [f for f in findings(report)
                               if f['severity'] in ('error', 'incomplete') and f['rule'] not in skip]
                        self.assertFalse(bad, '\n'.join(f"{f['rule']} pc={f.get('pc')} {f.get('loc')}: "
                                                            f"{f['message']}" for f in bad))
                        if not skip:
                            self.assertEqual(code, 0, report.get('status'))
                        checked += 1
        if skip:
            print(f'\n[compiler_clean] {cxx}: findings of {", ".join(sorted(skip))} not asserted '
                  f'(upstream LLVM older than the release that fixes them)')
        if not checked:
            self.skipTest(f'{cxx} builds none of the clean fixtures')

    def test_clean_fixtures_wave64(self):
        """-mwavefrontsize64: the .s names only the processor, so the internal llvm-mc step must be
        given +wavefrontsize64 (hipsrc.assembler_wave_args) or every VCC-pair / SGPR-pair carry
        operand is rejected and the unit ends INCOMPLETE (exit 0 in hook mode, i.e. no coverage)."""
        cxx = compiler()
        if not cxx:
            if require_llvm():
                self.fail('no clang++ for gfx1201')
            self.skipTest('no clang++ (set RADISA_TEST_CXX)')
        skip = tolerated(cxx)
        checked = 0
        for src in clean_fixtures():
            if wave32_only(src):
                continue   # by construction; see the module docstring
            if hip_compiles(cxx, src):
                continue   # reported by test_clean_fixtures_have_no_errors
            for opt, mode in (('-O3', ()), ('-O2', ('--no-incremental',))):
                with self.subTest(fixture=src.name, opt=opt, mode=mode), tempfile.TemporaryDirectory() as cache:
                    code, report = run_check(src, '--cxx', cxx, f'--flag=-nogpuinc -nogpulib {opt} -mwavefrontsize64',
                                             '--cache-dir', cache, *mode)
                    self.assertTrue(report['kernels'], 'no kernel was checked')
                    self.assertEqual({k['metadata']['.wavefront_size'] for k in report['kernels']}, {64})
                    bad = [f for f in findings(report)
                           if f['severity'] in ('error', 'incomplete') and f['rule'] not in skip]
                    self.assertFalse(bad, '\n'.join(f"{f['rule']} pc={f.get('pc')} {f.get('loc')}: "
                                                        f"{f['message']}" for f in bad))
                    if not skip:
                        self.assertEqual(code, 0, report.get('status'))
                    checked += 1
        if not checked:
            self.skipTest(f'{cxx} builds none of the clean fixtures')


class AssemblerWaveArgs(unittest.TestCase):
    def test_from_kernel_descriptor_and_metadata(self):
        from radisa.hipsrc import assembler_wave_args
        w64 = ['-mattr=+wavefrontsize64']
        self.assertEqual(assembler_wave_args('\t.amdhsa_kernel k\n\t\t.amdhsa_wavefront_size32 0\n'), w64)
        self.assertEqual(assembler_wave_args('    .wavefront_size: 64\n'), w64)
        self.assertEqual(assembler_wave_args('\t.amdhsa_kernel k\n\t\t.amdhsa_wavefront_size32 1\n'
                                             '    .wavefront_size: 32\n', ['-mwavefrontsize64']), [])

    def test_without_kernel_uses_last_compile_flag(self):
        from radisa.hipsrc import assembler_wave_args
        self.assertEqual(assembler_wave_args('f:\n\ts_setpc_b64 s[30:31]\n', ['-mwavefrontsize64']),
                         ['-mattr=+wavefrontsize64'])
        self.assertEqual(assembler_wave_args('f:\n', ['-mwavefrontsize64', '-mno-wavefrontsize64']), [])
        self.assertEqual(assembler_wave_args('f:\n', []), [])


if __name__ == '__main__':
    unittest.main()
