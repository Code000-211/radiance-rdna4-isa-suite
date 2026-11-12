"""Differential test: code the compiler scheduled itself must not fail `radisa check`.

Every HIP fixture documented as clean (fixtures/check/clean_*.hip, or a .hip whose header says
"Must pass") is compiled for gfx1201 with the available clang and checked; any 'error' or
'incomplete' finding is a false positive of the linter (or a compiler bug worth reporting).

A fixture the compiler cannot build (clang 18 lacks e.g. the 3-argument __builtin_amdgcn_fence)
is skipped with the compiler's reason, unless RAD_ISA_REQUIRE_LLVM=1.

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


if __name__ == '__main__':
    unittest.main()
