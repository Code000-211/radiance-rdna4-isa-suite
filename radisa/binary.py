"""Bounded ELF64/LE, AMDHSA MessagePack, archive and offload-bundle readers.

No executable input is loaded into the GPU or host process. Source references:
AMD RDNA4 ISA (2025-04-07); LLVM AMDGPUUsage; ClangOffloadBundler.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any, Iterator

MAX_ENTRIES = 100_000
MAX_BYTES = 1024 * 1024 * 1024
BUNDLE_MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
# e_flags low byte to target name. The generic entries matter: ROCm ships one code
# object per generic family, and a name our table cannot resolve would look like a
# bundle/ELF target disagreement.
TARGET_FLAGS = {0x48: 'gfx1200', 0x4e: 'gfx1201',
                0x51: 'gfx9-generic', 0x52: 'gfx10-1-generic', 0x53: 'gfx10-3-generic',
                0x54: 'gfx11-generic', 0x59: 'gfx12-generic', 0x5f: 'gfx9-4-generic'}


class FormatError(ValueError):
    """Malformed or unsupported evidence. Never equivalent to a clean inspection."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def span(data: bytes, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or size > MAX_BYTES or offset > len(data) - size:
        raise FormatError(f'out-of-file range offset={offset}, size={size}, file={len(data)}')
    return data[offset:offset + size]


def unpack(fmt: str, data: bytes, offset: int = 0) -> tuple:
    return struct.unpack(fmt, span(data, offset, struct.calcsize(fmt)))


def cstring(data: bytes, offset: int) -> str:
    if not 0 <= offset < len(data):
        raise FormatError('string-table offset out of range')
    end = data.find(b'\0', offset)
    if end < 0:
        raise FormatError('unterminated string-table entry')
    try:
        return data[offset:end].decode('utf-8')
    except UnicodeDecodeError as ex:
        raise FormatError('non-UTF8 symbol/section name') from ex


class MessagePack:
    """Minimal, bounded MessagePack decoder; no optional PyYAML/msgpack dependency."""
    def __init__(self, data: bytes):
        self.data, self.pos, self.items = data, 0, 0

    def take(self, n: int) -> bytes:
        result = span(self.data, self.pos, n)
        self.pos += n
        return result

    def number(self, fmt: str) -> int | float:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def read(self, depth: int = 0) -> Any:
        self.items += 1
        if depth > 64 or self.items > 2_000_000:
            raise FormatError('MessagePack resource limit exceeded')
        tag = self.take(1)[0]
        if tag < 0x80:
            return tag
        if tag >= 0xe0:
            return tag - 256
        if 0xa0 <= tag <= 0xbf:
            return self.text(tag & 31)
        if 0x90 <= tag <= 0x9f:
            return self.array(tag & 15, depth)
        if 0x80 <= tag <= 0x8f:
            return self.mapping(tag & 15, depth)
        if tag in (0xc0, 0xc2, 0xc3):
            return {0xc0: None, 0xc2: False, 0xc3: True}[tag]
        nums = {0xcc: '>B', 0xcd: '>H', 0xce: '>I', 0xcf: '>Q',
                0xd0: '>b', 0xd1: '>h', 0xd2: '>i', 0xd3: '>q',
                0xca: '>f', 0xcb: '>d'}
        if tag in nums:
            return self.number(nums[tag])
        lengths = {0xd9: '>B', 0xda: '>H', 0xdb: '>I',
                   0xc4: '>B', 0xc5: '>H', 0xc6: '>I',
                   0xdc: '>H', 0xdd: '>I', 0xde: '>H', 0xdf: '>I'}
        if tag in lengths:
            n = int(self.number(lengths[tag]))
            if tag in (0xd9, 0xda, 0xdb):
                return self.text(n)
            if tag in (0xc4, 0xc5, 0xc6):
                return self.take(n)
            if tag in (0xdc, 0xdd):
                return self.array(n, depth)
            return self.mapping(n, depth)
        raise FormatError(f'unsupported MessagePack tag 0x{tag:02x}')

    def text(self, n: int) -> str:
        try:
            return self.take(n).decode('utf-8')
        except UnicodeDecodeError as ex:
            raise FormatError('invalid UTF8 in metadata') from ex

    def array(self, n: int, depth: int) -> list:
        if n > MAX_ENTRIES or n > len(self.data) - self.pos:
            raise FormatError('implausible MessagePack array length')
        return [self.read(depth + 1) for _ in range(n)]

    def mapping(self, n: int, depth: int) -> dict:
        if n > MAX_ENTRIES or 2*n > len(self.data) - self.pos:
            raise FormatError('implausible MessagePack map length')
        out = {}
        for _ in range(n):
            k = self.read(depth + 1)
            if not isinstance(k, (str, int)) or isinstance(k, bool) or k in out:
                raise FormatError('unsupported or duplicate metadata map key')
            out[k] = self.read(depth + 1)
        return out

    def decode(self) -> Any:
        out = self.read()
        if self.pos != len(self.data):
            raise FormatError('trailing bytes after MessagePack value')
        return out


@dataclass(frozen=True)
class Section:
    index: int
    name: str
    type: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    info: int
    alignment: int
    entry_size: int


@dataclass(frozen=True)
class Symbol:
    name: str
    value: int
    size: int
    type: int
    section: int


class ELF:
    def __init__(self, data: bytes):
        self.data = data
        if span(data, 0, 7) != b'\x7fELF\x02\x01\x01':
            raise FormatError('only ELF64 little-endian version 1 is supported')
        h = unpack('<HHIQQQIHHHHHH', data, 16)
        (self.type, self.machine, version, self.entry, phoff, shoff, self.flags,
         ehsize, phentsize, phnum, shentsize, shnum, shstrndx) = h
        self.osabi, self.abiversion = data[7], data[8]
        if version != 1 or ehsize < 64 or self.type not in (1, 2, 3):
            raise FormatError('unsupported ELF header')
        if not shoff or shentsize < 64:
            raise FormatError('missing section table: symbol/byte coverage unavailable')
        s0 = unpack('<IIQQQQIIQQ', data, shoff)
        shnum = shnum or s0[5]
        if shstrndx == 0xffff:
            shstrndx = s0[6]
        if phnum == 0xffff:
            phnum = s0[7]
        if not (0 < shnum <= MAX_ENTRIES and 0 <= shstrndx < shnum and phnum <= MAX_ENTRIES):
            raise FormatError('invalid ELF table counts')
        span(data, shoff, shnum*shentsize)
        headers = [unpack('<IIQQQQIIQQ', data, shoff+i*shentsize) for i in range(shnum)]
        st = headers[shstrndx]
        strings = span(data, st[4], st[5])
        self.sections = []
        for i, s in enumerate(headers):
            name = cstring(strings, s[0])
            if s[1] != 8 and i != 0:
                span(data, s[4], s[5])
            self.sections.append(Section(i, name, *s[1:]))
        self.segments = []
        if phnum:
            if phentsize < 56:
                raise FormatError('invalid program header size')
            span(data, phoff, phnum*phentsize)
            for i in range(phnum):
                p = unpack('<IIQQQQQQ', data, phoff+i*phentsize)
                if p[5] > p[6]:
                    raise FormatError('ELF segment file size exceeds memory size')
                span(data, p[2], p[5])
                self.segments.append(p)
        self.symbols: list[Symbol] = []
        seen = set()
        for s in self.sections:
            if s.type not in (2, 11):
                continue
            if s.entry_size < 24 or s.size % s.entry_size or s.size//s.entry_size > MAX_ENTRIES:
                raise FormatError('invalid symbol-table size')
            if not 0 <= s.link < len(self.sections):
                raise FormatError('invalid symbol string-table link')
            ss = self.section_bytes(self.sections[s.link])
            for i in range(s.size//s.entry_size):
                n, info, other, ndx, value, size = unpack('<IBBHQQ', data, s.offset+i*s.entry_size)
                sym = Symbol(cstring(ss, n), value, size, info & 15, ndx)
                if sym not in seen:
                    self.symbols.append(sym)
                    seen.add(sym)
        self.target = TARGET_FLAGS.get(self.flags & 255, f'unknown-eflags-0x{self.flags & 255:02x}')

    def section_bytes(self, s: Section) -> bytes:
        if s.type == 8:
            raise FormatError('NOBITS section has no file evidence')
        return span(self.data, s.offset, s.size)

    def symbol_bytes(self, sym: Symbol, count: int | None = None) -> bytes:
        if not 0 < sym.section < len(self.sections):
            raise FormatError(f'undefined/unsupported symbol section: {sym.name}')
        s = self.sections[sym.section]
        rel = sym.value if self.type == 1 else sym.value - s.address
        return span(self.section_bytes(s), rel, sym.size if count is None else count)

    def named_symbol(self, name: str) -> Symbol:
        found = [s for s in self.symbols if s.name == name and 0 < s.section < len(self.sections)]
        if len(found) != 1:
            raise FormatError(f'expected one defined symbol {name!r}, found {len(found)}')
        return found[0]

    def notes(self) -> Iterator[tuple[str, int, bytes]]:
        ranges = {(s.offset, s.size) for s in self.sections if s.type == 7}
        ranges.update((p[2], p[5]) for p in self.segments if p[0] == 4)
        seen = set()
        for off, length in sorted(ranges):
            region = span(self.data, off, length)
            pos = 0
            while pos < len(region):
                if not any(region[pos:]):
                    break
                namesz, descsz, typ = unpack('<III', region, pos)
                if namesz > 1024 or descsz > 64*1024*1024:
                    raise FormatError('implausible ELF note length')
                pos += 12
                name = span(region, pos, namesz).rstrip(b'\0')
                pos += (namesz+3) & ~3
                desc = span(region, pos, descsz)
                pos += (descsz+3) & ~3
                key = (name, typ, desc)
                if key not in seen:
                    seen.add(key)
                    yield name.decode('ascii', errors='replace'), typ, desc

    def metadata(self) -> dict:
        found = [MessagePack(d).decode() for n, t, d in self.notes() if n == 'AMDGPU' and t == 32]
        if len(found) != 1 or not isinstance(found[0], dict):
            raise FormatError('expected one AMDGPU type-32 MessagePack metadata note')
        md = found[0]
        version = md.get('amdhsa.version')
        if not isinstance(version, list) or len(version) != 2 or version[0] != 1:
            raise FormatError(f'unsupported AMDHSA metadata version {version!r}')
        kernels = md.get('amdhsa.kernels')
        if not isinstance(kernels, list) or not kernels:
            raise FormatError('empty or missing amdhsa.kernels')
        return md

    def mapped_executable(self, address: int, size: int) -> bool:
        # File-backed PT_LOAD proof, stronger than merely observing .text size.
        return any(p[0] == 1 and p[1] & 1 and p[3] <= address and
                   address + size <= p[3] + p[5] for p in self.segments)


def target_processor(declared: str) -> str | None:
    """The processor of an amdhsa.target string, `<arch>-<vendor>-<os>-<environment>-<processor>[:features]`.

    The environment is empty up to LLVM 20 (`amdgcn-amd-amdhsa--gfx1201`) and `unknown` from
    LLVM 21 on (`amdgcn-amd-amdhsa-unknown-gfx1201`); a processor may itself hold dashes
    (`gfx10-1-generic`)."""
    parts = declared.split(':')[0].split('-', 4)
    return parts[4] if len(parts) == 5 else None


def kernel_inventory(elf: ELF) -> tuple[list[dict], list[str]]:
    """Enumerate expected kernels BEFORE disassembly, cross-check descriptor symbols."""
    if elf.machine != 224:
        raise FormatError('not EM_AMDGPU')
    if elf.osabi != 64:
        raise FormatError('not AMDHSA ELF OSABI 64')
    if elf.abiversion not in (1, 2, 3, 4):
        raise FormatError(f'unsupported AMDHSA code-object ABI byte {elf.abiversion}')
    md = elf.metadata()
    declared = md.get('amdhsa.target')
    if declared and not (isinstance(declared, str) and target_processor(declared) == elf.target):
        raise FormatError('metadata target disagrees with ELF e_flags')
    out, issues, names, descriptors = [], [], set(), set()
    for raw in md['amdhsa.kernels']:
        if not isinstance(raw, dict):
            raise FormatError('kernel metadata entry is not a map')
        name, symbol = raw.get('.name'), raw.get('.symbol')
        if not isinstance(name, str) or not isinstance(symbol, str) or not name or not symbol:
            raise FormatError('kernel has no usable .name/.symbol')
        if name in names or symbol in descriptors:
            raise FormatError('duplicate name or descriptor in one code object')
        names.add(name); descriptors.add(symbol)
        k = {'name': name, 'symbol': symbol, 'metadata': raw, 'issues': []}
        try:
            for f in ('.wavefront_size', '.vgpr_count', '.sgpr_count',
                      '.group_segment_fixed_size', '.private_segment_fixed_size'):
                if type(raw.get(f)) is not int or raw[f] < 0:
                    raise FormatError(f'missing/invalid {f}')
            if raw['.wavefront_size'] not in (32, 64):
                raise FormatError('invalid wavefront size')
            if raw['.vgpr_count'] > 256:
                raise FormatError('logical VGPR count exceeds gfx1201 range')
            kd = elf.named_symbol(symbol)
            if kd.size != 64:
                raise FormatError(f'kernel descriptor size {kd.size}, expected 64')
            desc = elf.symbol_bytes(kd)
            group, private, kernarg = unpack('<III', desc)
            entry_off = unpack('<q', desc, 16)[0]
            rsrc3, rsrc1, rsrc2, props = unpack('<IIIH', desc, 44)
            preload = unpack('<H', desc, 58)[0]  # KERNARG_PRELOAD_SPEC: LENGTH 6:0, OFFSET 15:7
            wave = 32 if props & (1 << 10) else 64
            if wave != raw['.wavefront_size']:
                raise FormatError('descriptor and metadata wave sizes disagree')
            if group != raw['.group_segment_fixed_size'] or private != raw['.private_segment_fixed_size']:
                raise FormatError('descriptor and metadata LDS/scratch sizes disagree')
            gran = 8 if wave == 32 else 4  # Descriptor ENCODING, not physical granularity.
            encoded = ((rsrc1 & 63) + 1)*gran
            if encoded < max(1, raw['.vgpr_count']):
                raise FormatError('descriptor allocates fewer registers than metadata requires')
            fn = elf.named_symbol(name)
            if fn.type != 2 or not fn.size or not (elf.sections[fn.section].flags & 4):
                raise FormatError('missing finite executable STT_FUNC extent')
            if fn.size % 4:
                raise FormatError('kernel code size is not a DWORD multiple')
            if elf.type == 1:
                k['issues'].append('relocatable ELF: entry relocation and load mapping are unresolved')
                entry = fn.value
            else:
                entry = kd.value + entry_off
                if entry != fn.value:
                    raise FormatError('descriptor entry differs from named function address')
                if entry % 256:
                    raise FormatError('kernel entry is not 256-byte aligned')
                if kd.value % 64:
                    raise FormatError('kernel descriptor is not 64-byte aligned')
            if elf.type != 1 and not elf.mapped_executable(entry, fn.size):
                raise FormatError('kernel bytes are not covered by an executable PT_LOAD')
            code = elf.symbol_bytes(fn)
            # The lookahead past a function end is only meaningful up to the end of the
            # executable segment: at or near that end there is provably nothing left to
            # read, so the proof covers min(256, bytes actually left in the segment) and
            # the number proved is recorded instead of claiming a full 256.
            seg_end = max((p[3] + p[5] for p in elf.segments
                           if p[0] == 1 and p[1] & 1 and p[3] <= entry < p[3] + p[5]), default=0)
            lookahead = max(0, min(256, seg_end - (entry + fn.size))) if seg_end else 0
            proved = lookahead == 0 or elf.mapped_executable(entry + fn.size, lookahead)
            k.update(address=entry, size=fn.size, section=fn.section,
                     code_sha256=sha256(code), code_hex=code.hex(),
                     descriptor={'wave': wave, 'encoded_vgprs': encoded,
                                 'rsrc1': rsrc1, 'rsrc2': rsrc2, 'rsrc3': rsrc3,
                                 'properties': props, 'kernarg': kernarg,
                                 'kernarg_preload': preload},
                     padding_file_backed=proved,
                     padding_bytes_proved=lookahead if proved else 0)
            if not proved:
                k['issues'].append('256 bytes after function end not proved file-backed executable PT_LOAD')
        except (FormatError, struct.error) as ex:
            k['issues'].append(str(ex))
        out.append(k)
    kd_symbols = {s.name for s in elf.symbols if s.name.endswith('.kd') and
                  s.type == 1 and 0 < s.section < len(elf.sections)}
    for extra in sorted(kd_symbols - descriptors):
        issues.append(f'descriptor {extra} has no kernel metadata; enumeration is incomplete')
    return out, issues


def unpack_bundle(data: bytes) -> tuple[list[tuple[str, bytes]], int]:
    if not data.startswith(BUNDLE_MAGIC):
        raise FormatError('not an uncompressed clang offload bundle')
    n = unpack('<Q', data, 24)[0]
    if not 0 < n <= 4096:
        raise FormatError('invalid offload bundle entry count')
    entries, pos = [], 32
    ids = set()
    for _ in range(n):
        off, size, length = unpack('<QQQ', data, pos)
        pos += 24
        if not 0 < length <= 4096:
            raise FormatError('invalid offload bundle target length')
        try:
            name = span(data, pos, length).decode('utf-8')
        except UnicodeDecodeError as ex:
            raise FormatError('invalid bundle ID') from ex
        pos += length
        if name in ids:
            raise FormatError('duplicate offload bundle ID')
        ids.add(name)
        entries.append((name, off, size))
    ranges = sorted((off, off+size) for _, off, size in entries if size)
    for i, (start, end) in enumerate(ranges):
        if start < pos or end > len(data) or (i and ranges[i-1][1] > start):
            raise FormatError('overlapping/truncated offload entry')
    return [(n, span(data, off, size)) for n, off, size in entries], max([pos]+[b for _, b in ranges])


def ar_members(data: bytes) -> Iterator[tuple[str, bytes]]:
    if not data.startswith(b'!<arch>\n'):
        raise FormatError('not a regular ar archive; thin archives require materialization')
    pos, longnames, count = 8, b'', 0
    while pos < len(data):
        count += 1
        if count > MAX_ENTRIES:
            raise FormatError('archive member limit exceeded')
        h = span(data, pos, 60)
        if h[58:60] != b'`\n':
            raise FormatError('invalid ar member header')
        try:
            n = int(h[48:58].decode('ascii').strip())
            name = h[:16].decode('ascii').strip()
        except ValueError as ex:
            raise FormatError('invalid ar length/name') from ex
        content = span(data, pos+60, n)
        pos += 60+n+(n & 1)
        if name == '//':
            longnames = content
            continue
        if name in ('/', '/SYM64/') or name.startswith('__.SYMDEF'):
            continue
        if name.startswith('#1/'):
            try:
                size = int(name[3:])
                name = span(content, 0, size).rstrip(b'\0').decode('utf-8')
                content = content[size:]
            except ValueError as ex:
                raise FormatError('bad BSD ar name') from ex
        elif name.startswith('/'):
            try:
                off = int(name[1:])
            except ValueError as ex:
                raise FormatError('bad GNU ar name offset') from ex
            if not 0 <= off < len(longnames):
                raise FormatError('GNU ar name offset out of range')
            end = longnames.find(b'/\n', off)
            if end < 0:
                raise FormatError('unterminated GNU ar name')
            name = longnames[off:end].decode('utf-8')
        else:
            name = name.rstrip('/')
        yield name, content
    if pos > len(data)+1:
        raise FormatError('truncated ar padding')


def extract_code_objects(data: bytes, origin: str, depth: int = 0) -> tuple[list[dict], list[dict]]:
    """Return all embedded AMDGPU objects and explicit skipped/error evidence.

    Host ELF, regular ar, standalone AMDGPU ELF and concatenated .hip_fatbin
    bundles are supported. Compression/LLVM bitcode/opaque offload containers
    are NOT silently accepted.
    """
    if depth > 8 or len(data) > MAX_BYTES:
        raise FormatError('container resource limit exceeded')
    objects, events = [], []
    if data.startswith(b'!<arch>\n'):
        for index, (name, payload) in enumerate(ar_members(data)):
            a, b = extract_code_objects(payload, f'{origin}!ar[{index}]/{name}', depth+1)
            objects.extend(a); events.extend(b)
    elif data.startswith(b'\x7fELF'):
        elf = ELF(data)
        if elf.machine == 224:
            objects.append({'origin': origin, 'target': elf.target, 'data': data, 'sha256': sha256(data)})
        else:
            found = False
            for s in elf.sections:
                if s.name == '.hip_fatbin' or s.name.startswith('__CLANG_OFFLOAD_BUNDLE__'):
                    found = True
                    a, b = extract_code_objects(elf.section_bytes(s), f'{origin}!{s.name}', depth+1)
                    objects.extend(a); events.extend(b)
                elif s.name.startswith(('.llvm.offloading', '.llvm.lto')):
                    found = True
                    events.append({'origin': origin+'!'+s.name, 'status': 'unsupported',
                                   'reason': 'offload/LTO container not supported; provide final uncompressed code object'})
            if not found:
                events.append({'origin': origin, 'status': 'host_only'})
    elif data.startswith(BUNDLE_MAGIC):
        pos, index = 0, 0
        while pos < len(data):
            # Linked .hip_fatbin sections may concatenate bundles with alignment zeros.
            while pos < len(data) and data[pos] == 0:
                pos += 1
            if pos == len(data):
                break
            entries, consumed = unpack_bundle(data[pos:])
            for name, payload in entries:
                if not payload or name.startswith('host-'):
                    events.append({'origin': f'{origin}!bundle[{index}]/{name}', 'status': 'host_bundle'})
                    continue
                a, b = extract_code_objects(payload, f'{origin}!bundle[{index}]/{name}', depth+1)
                # A triple/ELF target disagreement must not be concealed by filtering.
                # A generic family label is the one legitimate exception: ROCm's device
                # libraries ship one code object for "gfx12" (and gfx10, gfx11, gfx9),
                # whose ELF names the concrete target gfx1200.
                at = name.find('amdgcn-')
                label = target_processor(name[at:]) if at >= 0 else None
                if not (label or '').startswith('gfx'):
                    label = None
                for obj in a:
                    if label and label != obj['target'] and not obj['target'].startswith(label):
                        raise FormatError('bundle ID disagrees with device ELF target')
                objects.extend(a); events.extend(b)
            pos += consumed; index += 1
    else:
        label = ('compressed clang bundle' if data.startswith(b'CCOB') else
                 'thin archive' if data.startswith(b'!<thin>') else
                 'LLVM bitcode' if data.startswith((b'BC\xc0\xde', b'\xde\xc0\x17\x0b')) else
                 'unrecognized binary container')
        events.append({'origin': origin, 'status': 'unsupported', 'reason': label})
    return objects, events
