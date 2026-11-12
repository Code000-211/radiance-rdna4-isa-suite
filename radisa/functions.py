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
`call_target` resolves the function an s_swappc_b64 calls (check.py: NORETURN_CALL_RETURNS).
"""
from __future__ import annotations

import re
import struct

from .binary import ELF, FormatError, kernel_inventory, sha256, unpack
from .rules import operands

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


SHT_RELA = 4
R_AMDGPU_REL32_LO = 10   # S + A - P, low 32 bits (llvm/BinaryFormat/ELFRelocs/AMDGPU.def)
# What LLVM places between s_getpc_b64 and the s_add pair of a call sequence: the gfx12 s_getpc
# sign-extension workaround on the high half (SIInstrInfo: S_SEXT_I32_I16 after S_GETPC_B64 when
# hasGetPCZeroExtension is false), and the wait/delay/nop fillers of later passes.
_CALL_FILLER = {'s_wait_alu', 's_waitcnt_depctr', 's_delay_alu', 's_nop', 's_sext_i32_i16'}


def relocations(elf: ELF) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """(section index, offset) -> (relocation type, symbol value, symbol section, addend) of every
    SHT_RELA entry of a relocatable object ({} for a linked one, whose literals are resolved)."""
    out: dict = {}
    if elf.type != 1:
        return out
    for rs in elf.sections:
        if rs.type != SHT_RELA or rs.entry_size < 24 or not 0 < rs.link < len(elf.sections):
            continue
        st = elf.sections[rs.link]
        if st.entry_size < 24:
            continue
        for n in range(rs.size // rs.entry_size):
            off, info, addend = unpack('<QQq', elf.data, rs.offset + n * rs.entry_size)
            idx = info >> 32
            if idx * st.entry_size >= st.size:
                continue
            _, _, _, ndx, value, _ = unpack('<IBBHQQ', elf.data, st.offset + idx * st.entry_size)
            out[(rs.info, off)] = (info & 0xffffffff, value, ndx, addend)
    return out


def call_target(ins: list[dict], ix: int, relocs: dict, section: int | None) -> tuple[int, int | None] | None:
    """(address, section or None) of the function the call ins[ix] jumps to, or None if the machine
    code does not say. LLVM calls with (SIInstrInfo, SI_PC_ADD_REL_OFFSET / AMDGPUMCInstLower)
        s_getpc_b64 s[N:N+1]
        s_add_co_u32 sN, sN, callee@rel32@lo+A      ; A = literal address - address after s_getpc
        s_add_co_ci_u32 sN+1, sN+1, callee@rel32@hi+B
        s_swappc_b64 s[30:31], s[N:N+1]
    so the callee is (address after s_getpc_b64) + the 64-bit offset in the two literals. In a
    relocatable object the literals are 0 and the R_AMDGPU_REL32_LO relocation of the low one
    (S + A - P) names it: callee = S + A - (P - address after s_getpc_b64), which also holds for
    a relocation against a section symbol. Only this shape (with `_CALL_FILLER` in between) is
    resolved; `section`: the section of the calling code (relocatable objects)."""
    a = operands(ins[ix]['args'])
    if ins[ix]['op'] != 's_swappc_b64' or len(a) != 2:
        return None
    m = re.fullmatch(r's\[(\d+):(\d+)\]', a[1])
    if not m or int(m[2]) != int(m[1]) + 1:
        return None
    lo, hi = f's{m[1]}', f's{m[2]}'

    def filler(i: dict) -> bool:   # s_sext_i32_i16 only as the s_getpc workaround on the high half
        return i['op'] in _CALL_FILLER and (i['op'] != 's_sext_i32_i16' or operands(i['args']) == [hi, hi])
    j, found = ix - 1, {}
    for ops, reg in ((('s_add_co_ci_u32', 's_addc_u32'), hi), (('s_add_co_u32', 's_add_u32'), lo)):
        while j >= 0 and filler(ins[j]):
            j -= 1
        if j < 0 or ins[j]['op'] not in ops:
            return None
        o = operands(ins[j]['args'])
        if len(o) != 3 or o[0] != reg or o[1] != reg:
            return None
        found[reg] = (ins[j], o[2])
        j -= 1
    while j >= 0 and filler(ins[j]):
        j -= 1
    if j < 0 or ins[j]['op'] != 's_getpc_b64' or ins[j]['args'].strip() != a[1]:
        return None
    after_getpc = ins[j]['pc'] + ins[j]['size']
    add_lo, lit_lo = found[lo]
    if relocs:
        p = add_lo['pc'] + 4                  # the literal follows the 4-byte SOP2 word
        r = relocs.get((section, p))
        if r is None or r[0] != R_AMDGPU_REL32_LO:
            return None
        _, value, ndx, addend = r
        return value + addend - (p - after_getpc), ndx
    try:
        offset = (int(lit_lo, 0) & 0xffffffff) | (int(found[hi][1], 0) & 0xffffffff) << 32
    except ValueError:
        return None
    if offset >= 1 << 63:
        offset -= 1 << 64
    return after_getpc + offset, None
