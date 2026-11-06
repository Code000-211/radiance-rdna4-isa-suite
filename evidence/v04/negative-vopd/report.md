# RDNA4 binary ISA report

**Status: FAIL**

**This is not full ISA, memory-safety, numerical, or GPU-wedge certification.**

Expected kernels: 1; analyzed: 1; fully decoded: 1
Expected bytes: 36; decoded: 36

## Global evidence findings

- INFO `RUNTIME_NOT_CAPTURED`: runtime-resolved kernels not reconciled; scope is supplied build artifacts only

## `smoke.kd`
Object: `743af10de7545ff2666f5b84aabd0ee3c1a07281dd6d198f586c4f0d3723c1fa`

Decoded 36/36 bytes; 5 instructions.
- ERROR `VOPD_BANK` at 0x1908: SRC0 bank conflict: v4 and v0
