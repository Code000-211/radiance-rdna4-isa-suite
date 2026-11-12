"""Non-kernel functions (callees) of a code object, shaped like the kernels of `binary.kernel_inventory`.

The code-object metadata lists kernels only (`amdhsa.kernels`); a function a kernel calls is an
STT_FUNC symbol in an executable section and nothing else. Each one becomes a kernel-shaped dict
with `'callee': True`, so `check.check_kernel` can run the same rules over it, with what a called
function differs in (LLVM SIInsertWaitcnts / AMDGPUWaitSGPRHazards, see `waits.CALLS`):

  * at entry the caller's memory operations may be outstanding on every counter
    (`waits.analyze(..., callee=True)`); every SGPR pair counts as read by vector instructions
    (`sgpr.analyze(..., callee=True)`, AMDGPUWaitSGPRHazards: "Callee must consider all SGPRs
    as tracked");
  * it ends with `s_setpc_b64 s[30:31]` (SIRegisterInfo::getReturnAddressReg), not s_endpgm, so
    CFG_FALLTHROUGH does not apply; a return must leave every counter but STOREcnt drained
    (CALLEE_RETURN_OUTSTANDING);
  * it has no kernel descriptor (`descriptor` {}), so VGPR_OUT_OF_RANGE does not apply: how
    many VGPRs the wave owns is the calling kernel's descriptor.

The wave size is not recorded for a function; it is the one every kernel of the object declares.
"""
from __future__ import annotations

import struct

from .binary import ELF, FormatError, sha256

SHF_EXECINSTR = 4
STT_FUNC = 2


def _wave(kernels: list[dict]) -> tuple[int, str | None]:
    waves = {k['metadata'].get('.wavefront_size') for k in kernels if isinstance(k.get('metadata'), dict)}
    waves.discard(None)
    if len(waves) == 1:
        return waves.pop(), None
    if not waves:
        return 32, 'no kernel in the object declares a wave size; wave32 assumed'
    return 32, f'the kernels of the object disagree on the wave size ({sorted(waves)}); wave32 assumed'


def function_inventory(elf: ELF, kernels: list[dict], issues: list[str] | None = None) -> list[dict]:
    """Every STT_FUNC symbol in an executable section that is not a kernel entry, as a kernel-shaped
    dict: name, symbol (= name), address, size, section, code_sha256, code_hex, metadata (only
    `.wavefront_size` from the object's kernels, the other counts 0), descriptor {}, issues,
    callee True.

    Aliases (same section, address and size) are listed once; a symbol inside a kernel's extent is
    not a function of its own. A symbol without a size has no extent to check: it is left out and
    a message goes to `issues` (if given). Bytes that cannot be read become an entry without
    `code_hex` and with the reason in its `issues`, as for kernels.
    """
    wave, why = _wave(kernels)
    starts = {(k.get('section'), k.get('address')) for k in kernels if 'address' in k}
    names = {k['name'] for k in kernels}
    extents = [(k.get('section'), k['address'], k['address'] + k.get('size', 0)) for k in kernels if 'address' in k]
    out, seen = [], set()
    for sym in sorted(elf.symbols, key=lambda s: (s.section, s.value, s.name)):
        if sym.type != STT_FUNC or not 0 < sym.section < len(elf.sections):
            continue
        if not elf.sections[sym.section].flags & SHF_EXECINSTR or sym.name in names:
            continue
        address = sym.value
        if (sym.section, address) in starts:
            continue
        if any(sec == sym.section and lo <= address < hi for sec, lo, hi in extents):
            continue
        if not sym.size:
            if issues is not None:
                issues.append(f'function {sym.name} has no symbol size; its code is not checked')
            continue
        key = (sym.section, address, sym.size)
        if key in seen:
            continue
        seen.add(key)
        md = {'.wavefront_size': wave, '.vgpr_count': 0, '.sgpr_count': 0,
              '.group_segment_fixed_size': 0, '.private_segment_fixed_size': 0}
        f = {'name': sym.name, 'symbol': sym.name, 'metadata': md, 'issues': [], 'descriptor': {},
             'callee': True, 'address': address, 'size': sym.size, 'section': sym.section}
        if why:
            f['issues'].append(why)
        try:
            if sym.size % 4:
                raise FormatError('function code size is not a DWORD multiple')
            if elf.type == 1:
                f['issues'].append('relocatable ELF: entry relocation and load mapping are unresolved')
            elif not elf.mapped_executable(address, sym.size):
                raise FormatError('function bytes are not covered by an executable PT_LOAD')
            code = elf.symbol_bytes(sym)
            f.update(code_sha256=sha256(code), code_hex=code.hex())
        except (FormatError, struct.error) as ex:
            f['issues'].append(str(ex))
        out.append(f)
    return out
