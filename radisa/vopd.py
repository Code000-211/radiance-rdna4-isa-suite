"""Raw GFX12 VOPD decoder/validator.

Normative: supplied AMD RDNA4 ISA §§7.8, 15.3.7, 16.11.
VOPD destination parity is encoded, not an independently selectable illegal bit.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
from .binary import FormatError

OPS = {0: 'fmac_f32', 1: 'fmaak_f32', 2: 'fmamk_f32', 3: 'mul_f32',
       4: 'add_f32', 5: 'sub_f32', 6: 'subrev_f32', 7: 'mul_dx9_zero_f32',
       8: 'mov_b32', 9: 'cndmask_b32', 10: 'max_num_f32', 11: 'min_num_f32',
       12: 'dot2acc_f32_f16', 13: 'dot2acc_f32_bf16',
       16: 'add_nc_u32', 17: 'lshlrev_b32', 18: 'and_b32'}
ACCUM = {0, 12, 13}
DPMACC = {12, 13}
LITERAL_OPS = {1, 2}


def is_vopd(data: bytes) -> bool:
    return len(data) >= 4 and int.from_bytes(data[:4], 'little') >> 26 == 0b110010


@dataclass(frozen=True)
class Component:
    op: int
    dst: int
    src0: int
    src1: int

    @property
    def name(self) -> str:
        return 'v_dual_' + OPS.get(self.op, f'reserved_{self.op}')

    def reads(self) -> set[int]:
        v = {self.src0 - 256} if self.src0 >= 256 else set()
        if self.op != 8:
            v.add(self.src1)
        if self.op in ACCUM:
            v.add(self.dst)
        return v

    def ports(self, y_mov_reroute: bool = False) -> dict[int, int]:
        out = {}
        if self.src0 >= 256:
            out[2 if y_mov_reroute else 0] = self.src0 - 256
        if self.op != 8:
            out[2 if self.op == 2 else 1] = self.src1
        if self.op in ACCUM:
            out[2] = self.dst
        return out

    def scalar_sources(self) -> set[int | str]:
        # Inline constants do not consume the scalar bus. Aperture constants,
        # SCC, M0 and EXEC do. VCC is implicit for CNDMASK.
        value = set()
        # LLVM usesConstantBus explicitly excludes SGPR_NULL (encoding 124).
        if (self.src0 < 128 and self.src0 != 124) or self.src0 in (235, 236, 237, 238, 253):
            value.add(self.src0)
        if self.src0 == 255 or self.op in LITERAL_OPS:
            value.add('literal')
        if self.op == 9:
            value.add(106)
        return value


@dataclass(frozen=True)
class Pair:
    x: Component
    y: Component
    size: int
    literal: int | None


def decode(data: bytes) -> Pair:
    if not is_vopd(data):
        raise FormatError('not a VOPD instruction boundary')
    if len(data) < 8:
        raise FormatError('truncated VOPD pair')
    bits = int.from_bytes(data[:8], 'little')
    x = Component((bits >> 22) & 15, (bits >> 56) & 255, bits & 511, (bits >> 9) & 255)
    ydst = (((bits >> 49) & 127) << 1) | ((x.dst & 1) ^ 1)
    y = Component((bits >> 17) & 31, ydst, (bits >> 32) & 511, (bits >> 41) & 255)
    need_literal = any(c.src0 == 255 or c.op in LITERAL_OPS for c in (x, y))
    size = 12 if need_literal else 8
    if len(data) < size:
        raise FormatError('truncated VOPD literal DWORD')
    return Pair(x, y, size, int.from_bytes(data[8:12], 'little') if need_literal else None)


def validate(pair: Pair, wave: int, allocated: int | None = None) -> list[tuple[str, str]]:
    """Return constraint findings, not tuning advice.

    REGISTER_RANGE is a diagnostic of the ISA-defined out-of-range behavior,
    not a claim that the encoding necessarily raises an illegal-instruction trap.
    The reporting layer classifies that finding as review rather than error.
    """
    problems = []
    if wave != 32:
        problems.append(('WAVE32', 'VOPD is legal only in Wave32'))
    if pair.x.op not in range(14):
        problems.append(('OPCODE', f'reserved X opcode {pair.x.op}'))
    if pair.y.op not in OPS:
        problems.append(('OPCODE', f'reserved Y opcode {pair.y.op}'))
    for label, c in (('X', pair.x), ('Y', pair.y)):
        if c.src0 in (233, 234, 250):
            problems.append(('DPP', f'{label}: DPP is prohibited for VOPD'))
        elif c.src0 in set(range(209, 233)) | {239, 249, 251, 252, 254}:
            problems.append(('SOURCE', f'{label}: reserved source encoding {c.src0}'))
        elif c.src0 in (107, 127):
            # The source table of the Wave32 VALU does not list the high halves, yet
            # gfx1201 reads and writes them (tools/vcc_hi_probe.hip, dual_form_probe:
            # 3+7 = 10 in the Y slot of a real VOPD pair). Reported as review, not as
            # an error, because the encoding is functional on the hardware we have.
            problems.append(('SOURCE_HI', f'{label}: VCC_HI/EXEC_HI sourced by Wave32 VALU'))
        if allocated is not None and (c.dst >= allocated or any(r >= allocated for r in c.reads())):
            problems.append(('REGISTER_RANGE', f'{label}: references outside descriptor VGPR allocation'))
    if pair.x.op in range(14) and pair.y.op in OPS:
        px = pair.x.ports()
        py = pair.y.ports(pair.x.op == 8 and pair.y.op == 8)
        for port in sorted(px.keys() & py.keys()):
            a, b = px[port], py[port]
            if port in (0, 1) and a != b and a % 4 == b % 4:
                problems.append(('BANK', f'SRC{port} bank conflict: v{a} and v{b}'))
            if port == 2 and a % 2 == b % 2:
                problems.append(('SRC2', f'SRC2 inputs must have opposite parity: v{a}, v{b}'))
        if pair.x.dst in pair.y.reads():
            # §7.8 forbids this in the encoded word. LLVM's gcn-create-vopd
            # emits it deliberately when a program-order WAR pair is swapped
            # into X/Y, and SIInstrInfo::isVOPDAntidependencyAllowed permits
            # that for every non-DPMACC opX, so only DPMACC is an error.
            rule = 'RAW_DPMACC' if pair.x.op in DPMACC else 'ANTIDEP'
            problems.append((rule, f'X writes v{pair.x.dst}, which Y reads in the same pair'))
        # The converse (Y writes X source) is explicitly allowed by §7.8.
        scalar = pair.x.scalar_sources() | pair.y.scalar_sources()
        if len(scalar) > 2:
            problems.append(('SCALAR_BUS', f'pair uses {len(scalar)} distinct scalar/literal sources (limit 2)'))
    return problems


def source_text(n: int, literal: int | None) -> str:
    if n >= 256:
        return f'v{n-256}'
    if n <= 105:
        return f's{n}'
    if n == 255:
        return hex(literal or 0)
    specials = {106: 'vcc_lo', 107: 'vcc_hi', 124: 'null', 125: 'm0',
                126: 'exec_lo', 127: 'exec_hi', 253: 'scc'}
    if n in specials:
        return specials[n]
    if 108 <= n <= 123:
        return f'ttmp{n-108}'
    if 128 <= n <= 192:
        return str(n-128)
    if 193 <= n <= 208:
        return str(-(n-192))
    return f'inline_encoding_{n}'


def display(pair: Pair) -> str:
    pieces = []
    for c in (pair.x, pair.y):
        operands = [f'v{c.dst}', source_text(c.src0, pair.literal)]
        if c.op != 8:
            operands.append(f'v{c.src1}')
        if c.op in LITERAL_OPS:
            operands.append(hex(pair.literal or 0))
        pieces.append(c.name+' '+', '.join(operands))
    return ' :: '.join(pieces)
