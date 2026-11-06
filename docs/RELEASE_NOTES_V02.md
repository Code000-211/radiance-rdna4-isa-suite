# v0.2.0 release notes

This stage replaces the v0.1 text-only prototype with a binary inventory/decode
pipeline, an independent VOPD checker, scoped static checks and optional AQL
resolution capture. See REVIEW.md for the earlier defects and corrections.

## Executed release checks

- 162 unittest test cases: passed; zero failures, errors or skips.
- Required external tools were enabled through check.sh.
- A real LLVM-generated and linked gfx1201 fixture passed the implemented checks.
- A deliberate raw VOPD bank-conflict mutation failed with exit code 1.
- Empty input collection and audit returned INCOMPLETE with exit code 2.
- Host ELF object/shared-library fatbin extraction was tested with a C++ compiler.
- The optional capture header was compiled with C++17, -Wall -Wextra -Werror,
  and tested for disabled mode, exact snapshot bytes, capture-time hashing,
  thread serialization, directory safety and fork rejection.
- git apply --check succeeded for the optional patch against the uploaded source.

Exact logs, compiler/decoder versions and structured results are in ../evidence/.
No GPU workload, production Radiance code object, or full HIP/HSA engine build was
validated in this environment. That remains a target-host merge requirement.
