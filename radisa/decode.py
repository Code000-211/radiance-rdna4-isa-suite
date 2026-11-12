"""LLVM-backed binary decode. Native LLVM runs in a timed child process.

VOPD boundaries are also decoded independently from the AMD bitfields. Other
instruction boundaries come from LLVM, never from regex guesses over bytes.
"""
from __future__ import annotations
import ctypes as C
import ctypes.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from .binary import FormatError
from . import vopd


def select_backend(library: str | None = None, objdump: str | None = None) -> dict:
    if library and objdump:
        raise FormatError('choose either --llvm-library or --objdump')
    if library:
        return {'kind': 'library', 'path': library}
    if objdump:
        path = shutil.which(objdump) if os.sep not in objdump else objdump
        if not path or not Path(path).is_file():
            raise FormatError(f'objdump not found: {objdump}')
        return {'kind': 'objdump', 'path': str(Path(path).resolve())}
    for base in (Path(os.environ.get('ROCM_PATH', '/opt/rocm'))/'llvm/lib',
                 Path('/opt/rocm/lib')):
        for p in sorted(base.glob('libLLVM.so*')):
            return {'kind': 'library', 'path': str(p)}
    for version in range(24, 18, -1):
        p = ctypes.util.find_library(f'LLVM-{version}')
        if p:
            return {'kind': 'library', 'path': p}
    for exe in ('/opt/rocm/llvm/bin/llvm-objdump', 'llvm-objdump'):
        p = shutil.which(exe)
        if p:
            return {'kind': 'objdump', 'path': p}
    raise FormatError('no LLVM AMDGPU decoder found; supply --llvm-library or --objdump')


class NativeDecoder:
    def __init__(self, library: str, wave: int):
        self.lib = C.CDLL(library)
        for suffix in ('TargetInfo', 'Target', 'TargetMC', 'Disassembler'):
            fn = getattr(self.lib, 'LLVMInitializeAMDGPU'+suffix)
            fn.argtypes = []; fn.restype = None; fn()
        self.lib.LLVMCreateDisasmCPUFeatures.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p,
                                                       C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
        self.lib.LLVMCreateDisasmCPUFeatures.restype = C.c_void_p
        self.lib.LLVMDisasmInstruction.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_uint64,
                                                 C.c_uint64, C.c_char_p, C.c_size_t]
        self.lib.LLVMDisasmInstruction.restype = C.c_size_t
        self.lib.LLVMDisasmDispose.argtypes = [C.c_void_p]
        self.lib.LLVMDisasmDispose.restype = None
        features = b'+wavefrontsize64,-wavefrontsize32' if wave == 64 else b'+wavefrontsize32,-wavefrontsize64'
        self.ctx = self.lib.LLVMCreateDisasmCPUFeatures(b'amdgcn-amd-amdhsa', b'gfx1201',
                                                       features, None, 0, None, None)
        if not self.ctx:
            raise FormatError('LLVM did not create a gfx1201 decoder')

    def close(self) -> None:
        if self.ctx:
            self.lib.LLVMDisasmDispose(self.ctx)
            self.ctx = None

    def one(self, data: bytes, pc: int) -> tuple[int, str]:
        b = (C.c_uint8*len(data)).from_buffer_copy(data)
        out = C.create_string_buffer(4096)
        size = int(self.lib.LLVMDisasmInstruction(self.ctx, b, len(data), pc, out, len(out)))
        return size, out.value.decode('utf-8', errors='replace').strip()


def instruction(pc: int, raw: bytes, text: str, independent: bool = False) -> dict:
    text = text.strip()
    fields = text.split(None, 1)
    return {'pc': pc, 'size': len(raw), 'raw': raw.hex(),
            'op': fields[0].lower() if fields else '<unknown>',
            'args': fields[1] if len(fields) > 1 else '',
            'text': text, 'vopd_independent_decode': independent}


def native_worker(request: dict) -> dict:
    results, decoders = {}, {}
    try:
        for k in request['kernels']:
            key, code, pc = k['symbol'], bytes.fromhex(k['code_hex']), k['address']
            wave = k['metadata']['.wavefront_size']
            if wave not in decoders:
                decoders[wave] = NativeDecoder(request['backend']['path'], wave)
            decoder = decoders[wave]
            decoded, issues, pos = [], [], 0
            while pos < len(code):
                chunk = code[pos:pos+24]
                size, text = decoder.one(chunk, pc+pos)
                if vopd.is_vopd(chunk):
                    try:
                        pair = vopd.decode(chunk)
                    except FormatError as ex:
                        issues.append({'pc': pc+pos, 'message': str(ex)})
                        break
                    if size and size != pair.size:
                        issues.append({'pc': pc+pos, 'message': 'LLVM/raw-VOPD instruction lengths disagree'})
                        break
                    size = pair.size
                    # Independent display avoids trusting LLVM's rendering of an illegal pair.
                    text = vopd.display(pair)
                    decoded.append(instruction(pc+pos, code[pos:pos+size], text, True))
                else:
                    if not size or size % 4 or size > len(code)-pos or size > 24 or not text:
                        issues.append({'pc': pc+pos, 'message': 'undecodable/truncated bytes; not assumed legal'})
                        break
                    decoded.append(instruction(pc+pos, code[pos:pos+size], text))
                pos += size
            results[key] = {'instructions': decoded, 'issues': issues, 'decoded_bytes': pos}
    finally:
        for decoder in decoders.values():
            decoder.close()
    return results


def parse_objdump(text: str) -> dict[int, tuple[bytes, str]]:
    """Support LLVM AMDGPU trailing-address and conventional leading-address output.

    This is not a symbol recognizer. Kernel boundaries come exclusively from ELF.
    Duplicate addresses with different bytes/text are a collection error.
    """
    out = {}
    # A branch carries its target after the words: `s_cbranch_scc0 7 // 000000000000: BFA10007 <.text+0x20>`.
    amd = re.compile(r'^\s*(.*?)\s*//\s*([0-9a-fA-F]+):\s*((?:[0-9a-fA-F]{8}(?:\s+|$))+)(?:<.*>\s*)?$')
    generic = re.compile(r'^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)(\S.*)$')
    for line in text.splitlines():
        a = amd.match(line)
        g = generic.match(line)
        if a:
            ins, pc, words = a.groups()
            raw = b''.join(int(w, 16).to_bytes(4, 'little') for w in words.split())
            pc = int(pc, 16)
        elif g:
            pc, encoded, ins = g.groups(); pc = int(pc, 16); raw = bytes.fromhex(encoded)
        else:
            continue
        value = (raw, ins.strip())
        if pc in out and out[pc] != value:
            raise FormatError('ambiguous objdump addresses (multiple sections or inconsistent output)')
        out[pc] = value
    return out


def decode_object(path: Path, kernels: list[dict], backend: dict, timeout: float = 120) -> dict:
    if backend['kind'] == 'library':
        payload = {'backend': backend, 'kernels': kernels}
        cmd = [sys.executable, '-m', 'radisa.decode']
        env = dict(os.environ)
        root = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        try:
            p = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True,
                               timeout=timeout, env=env, check=False)
        except (OSError, subprocess.TimeoutExpired) as ex:
            raise FormatError(f'isolated LLVM worker failed: {ex}') from ex
        if p.returncode:
            raise FormatError(f'isolated LLVM worker exit {p.returncode}: {p.stderr[-4000:]}')
        if p.stderr.strip():
            raise FormatError(f'LLVM diagnostic prevents clean decode: {p.stderr[-4000:]}')
        try:
            value = json.loads(p.stdout)
        except ValueError as ex:
            raise FormatError('invalid LLVM worker output') from ex
        if not isinstance(value, dict) or 'results' not in value:
            raise FormatError('missing decoder results envelope')
        backend['llvm_version'] = value.get('llvm_version', 'unknown')
        return value['results']
    cmd = [backend['path'], '--disassemble', '--mcpu=gfx1201', '--disassemble-zeroes', str(path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise FormatError(f'objdump failed: {ex}') from ex
    if p.returncode or p.stderr.strip():
        raise FormatError(f'objdump exit {p.returncode}: {p.stderr[-4000:]}')
    records = parse_objdump(p.stdout)
    results = {}
    for k in kernels:
        code, pos, decoded, issues = bytes.fromhex(k['code_hex']), 0, [], []
        while pos < len(code):
            pc = k['address']+pos
            if vopd.is_vopd(code[pos:]):
                try:
                    pair = vopd.decode(code[pos:])
                except FormatError as ex:
                    issues.append({'pc': pc, 'message': str(ex)}); break
                raw = code[pos:pos+pair.size]
                text = vopd.display(pair)
                size = pair.size
                own = True
            else:
                if pc not in records:
                    issues.append({'pc': pc, 'message': 'instruction absent from objdump output'}); break
                raw, text = records[pc]
                size, own = len(raw), False
                if not size or code[pos:pos+size] != raw:
                    issues.append({'pc': pc, 'message': 'objdump raw bytes disagree with ELF'}); break
                if text.startswith('.') or '<unknown>' in text:
                    issues.append({'pc': pc, 'message': 'objdump did not decode instruction'}); break
            decoded.append(instruction(pc, raw, text, own)); pos += size
        results[k['symbol']] = {'instructions': decoded, 'issues': issues, 'decoded_bytes': pos}
    return results


def worker_main() -> int:
    try:
        request = json.load(sys.stdin)
        results = native_worker(request)
        lib = C.CDLL(request['backend']['path'])
        version = 'unknown'
        if hasattr(lib, 'LLVMGetVersion'):
            fn = lib.LLVMGetVersion
            fn.argtypes = [C.POINTER(C.c_uint)]*3
            fn.restype = None
            major, minor, patch = C.c_uint(), C.c_uint(), C.c_uint()
            fn(C.byref(major), C.byref(minor), C.byref(patch))
            version = f'{major.value}.{minor.value}.{patch.value}'
        json.dump({'results': results, 'llvm_version': version}, sys.stdout)
        return 0
    except Exception as ex:
        print(f'{type(ex).__name__}: {ex}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(worker_main())
