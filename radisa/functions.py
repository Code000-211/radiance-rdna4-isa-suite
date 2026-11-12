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

A code object may hold functions and no kernel (`amdhsa.kernels: []`: a -fgpu-rdc translation unit
after llc, a device library), and an assembled `.s` may hold functions and no metadata at all:
`object_inventory` lists their functions all the same.

For the text report (check.py), `relative_pc` gives a finding's offset inside its kernel or function.
"""
from __future__ import annotations

import re
import struct

from .binary import ELF, FormatError, kernel_inventory, sha256

SHF_EXECINSTR = 4
STT_FUNC = 2


def _wave(kernels: list[dict], wave: int | None = None) -> tuple[int, str | None]:
    """(wave size of the object's functions, why it is a guess or None). `wave`: the size the caller
    knows from elsewhere (`radisa check --wave` for an assembled file), used when no kernel says."""
    waves = {k['metadata'].get('.wavefront_size') for k in kernels if isinstance(k.get('metadata'), dict)}
    waves.discard(None)
    if len(waves) == 1:
        return waves.pop(), None
    if not waves and wave in (32, 64):
        return wave, None
    if not waves:
        return 32, 'no kernel in the object declares a wave size; wave32 assumed'
    return 32, f'the kernels of the object disagree on the wave size ({sorted(waves)}); wave32 assumed'


def function_inventory(elf: ELF, kernels: list[dict], issues: list[str] | None = None,
                       wave: int | None = None, unchecked: list[str] | None = None) -> list[dict]:
    """Every STT_FUNC symbol in an executable section that is not a kernel entry, as a kernel-shaped
    dict: name, symbol (= name), address, size, section, code_sha256, code_hex, metadata (only
    `.wavefront_size` from the object's kernels, the other counts 0), descriptor {}, issues,
    callee True.

    Aliases (same section, address and size) are listed once; a symbol inside a kernel's extent is
    not a function of its own. A symbol without a size has no extent to check: it is left out and
    a message goes to `issues` (if given) and its name to `unchecked` (if given), for the CALLS
    note. Bytes that cannot be read become an entry without `code_hex` and with the reason in its
    `issues`, as for kernels. `wave`: see `_wave`.
    """
    wave, why = _wave(kernels, wave)
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
            if unchecked is not None and sym.name not in unchecked:
                unchecked.append(sym.name)
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


# What binary.ELF.metadata / kernel_inventory raise for an object without kernels: an empty or
# missing `amdhsa.kernels`, or no metadata note at all (an assembled .s without .amdgpu_metadata).
_NO_KERNELS = re.compile(r'amdhsa\.kernels|expected one AMDGPU type-32 MessagePack metadata note')


def object_inventory(elf: ELF, wave: int | None = None,
                     unchecked: list[str] | None = None) -> tuple[list[dict], list[dict], list[str]]:
    """(kernels, functions, issues) of one gfx1201 code object: `binary.kernel_inventory` and
    `function_inventory`. An object that declares no kernel but holds functions (see the module
    docstring) yields ([], functions, issues), the functions with the wave size `wave` (wave32 and
    a note in their issues if None); a function without a symbol size goes to `issues` and
    `unchecked` as in function_inventory. An object with neither kernels nor functions re-raises the FormatError of kernel_inventory, which check.py
    reports as NO_KERNELS."""
    try:
        ks, issues = kernel_inventory(elf)
    except FormatError as ex:
        if not _NO_KERNELS.search(str(ex)) or elf.machine != 224 or elf.target != 'gfx1201':
            raise
        issues, missing = [], []
        fs = function_inventory(elf, [], issues, wave, missing)
        if unchecked is not None:
            unchecked += [m for m in missing if m not in unchecked]
        if not fs and not missing:
            raise
        return [], fs, issues
    return ks, function_inventory(elf, ks, issues, wave, unchecked), issues


def has_functions(elf: ELF) -> bool:
    """True if the object has a sized STT_FUNC symbol in an executable section: an assembled .s
    without .amdhsa_kernel is then checked through `object_inventory` with callee semantics
    instead of as one snippet (a snippet's labels are not sized function symbols)."""
    return any(sym.type == STT_FUNC and sym.size and 0 < sym.section < len(elf.sections)
               and elf.sections[sym.section].flags & SHF_EXECINSTR for sym in elf.symbols)


def relative_pc(unit: dict, pc: int) -> int:
    """`pc` as an offset inside its kernel or function (`unit` as in a report's 'kernels' list or an
    inventory entry): pcs in findings are addresses in the object, and a unit starts at
    unit['address'] (0 for a snippet)."""
    return pc - (unit.get('address') or 0)
