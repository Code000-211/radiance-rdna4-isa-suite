# Sources and provenance

## Normative source supplied by the user

AMD, **“RDNA4” Instruction Set Architecture Reference Guide**, 7 April 2025,
707 physical PDF pages / 697 printed pages. Exact uploaded bytes are identified
in `../INPUTS.json`. The PDF is not redistributed with this suite.

Relevant sections reviewed for this implementation:

- §2.4: shader instruction-prefetch padding.
- §§3.2.1,3.3.2,3.3.3: PC, static VGPRs, dynamic VGPR mode and deallocation.
- §4.1.1: cache SCOPE and temporal hints; separate load/store/atomic meanings.
- §§5.4–5.8: message types, branching, barriers, memory/ALU dependency rules.
- §7.2.8: PERMLANE scheduling restriction.
- §§7.8,15.3.7,16.11: VOPD constraints, raw fields and opcode definitions.
- §§7.11–7.12.2: register indexing and WMMA hazards/layouts.
- §11.6: matrix load-transpose operations (not a numerical-layout proof).

Official online copy consulted and visually checked at the VOPD field/constraint
and WMMA hazard tables:

https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content

## External primary sources, separate from ISA-derived rules

LLVM documentation supplies code-object/container/tool details that the shader
ISA guide alone does not define. Retrieved during this review, 2026-10-27.

- AMDGPU ELF flags, AMDHSA metadata and kernel descriptors:
  https://llvm.org/docs/AMDGPUUsage.html
- Clang offload bundle binary format and IDs:
  https://clang.llvm.org/docs/ClangOffloadBundler.html
- llvm-objdump invocation and output:
  https://llvm.org/docs/CommandGuide/llvm-objdump.html
- LLVM C disassembler API:
  https://llvm.org/doxygen/group__LLVMCDisassembler.html
- Physical VGPR allocation/capacity functions, explicitly separate from encoding:
  https://llvm.org/doxygen/AMDGPUTargetParser_8cpp_source.html
  (`getVGPRAllocGranule`, `getTotalNumVGPRs`, `getMaxWavesPerEU`)
- LLVM's AMDGPU operand validation as an implementation cross-check, not a
  replacement for the supplied AMD guide:
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.8/llvm/lib/Target/AMDGPU/AsmParser/AMDGPUAsmParser.cpp

## Repository-derived facts

The uploaded source ZIP, not a guessed upstream current branch, determines the
AQL integration. Reviewed locations include:

```text
core/device/aql.cpp
core/device/aql_code.cpp
cmake/RadAddPlugin.cmake
libr4d/r4d_gemm_w4a8_prefill.hip
libr4d/r4d_attn_decode_h256_gqa6.hip
libr4d/r4d_attn_prefill_h256_gqa6.hip
libr4d/r4d_attn_paged_h256_gqa6.hip
libr4d/isa/res.sh
libr4d/isa/waterfall.sh
```

The first two establish the actual raw-bundle/AQL loader and symbol-resolution
integration. Source comments are not treated as current binary measurements.

## Implementation policy, not AMD mandates

The following are engineering choices made by this suite: strict output-directory
freshness; error/incomplete exit codes; SHA-256 identities; baseline workflow;
resource limits; subprocess isolation/timeouts; coverage reporting; conservative
blocking of unresolved review findings; and the optional capture journal format.
They must not be cited as architectural requirements from the AMD guide.
