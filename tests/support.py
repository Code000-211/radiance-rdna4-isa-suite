from pathlib import Path
import struct
ROOT=Path(__file__).resolve().parent.parent
FIXTURE=(ROOT/'fixtures/smoke.hsaco').read_bytes()

def pack(value):
    """Tiny independent test encoder. Runtime suite has no msgpack dependency."""
    if value is None:return b'\xc0'
    if type(value)is bool:return b'\xc3' if value else b'\xc2'
    if isinstance(value,int):
        if 0<=value<128:return bytes([value])
        if -32<=value<0:return bytes([value&255])
        if value>=0:return b'\xcf'+struct.pack('>Q',value)
        return b'\xd3'+struct.pack('>q',value)
    if isinstance(value,str):
        b=value.encode();return b'\xdb'+struct.pack('>I',len(b))+b
    if isinstance(value,list):return b'\xdd'+struct.pack('>I',len(value))+b''.join(pack(x)for x in value)
    if isinstance(value,dict):return b'\xdf'+struct.pack('>I',len(value))+b''.join(pack(k)+pack(v)for k,v in value.items())
    raise TypeError(type(value))

def bundle(entries):
    magic=b'__CLANG_OFFLOAD_BUNDLE__'
    assert len(magic)==24
    ids=[s.encode()for s,_ in entries]
    offset=32+sum(24+len(i)for i in ids)
    table=b'';payload=b''
    for (name,b),i in zip(entries,ids):
        table+=struct.pack('<QQQ',offset,len(b),len(i))+i;payload+=b;offset+=len(b)
    return magic+struct.pack('<Q',len(entries))+table+payload

def archive(members):
    result=b'!<arch>\n'
    for n,b in members:
        assert len(n)<=15
        result+=(n+'/').ljust(16).encode()+b'0'.ljust(12)+b'0'.ljust(6)+b'0'.ljust(6)+b'100644'.ljust(8)+str(len(b)).ljust(10).encode()+b'`\n'+b
        if len(b)%2:result+=b'\n'
    return result

def vopd_bytes(x=4,y=4,dx=6,dy=7,sx=260,sy=261,vx=0,vy=1,literal=None):
    assert (dx^dy)&1
    word=sx|(vx<<9)|(y<<17)|(x<<22)|(0b110010<<26)|(sy<<32)|(vy<<41)|((dy>>1)<<49)|(dx<<56)
    return word.to_bytes(8,'little')+(struct.pack('<I',literal)if literal is not None else b'')

def instruction(op,args='',pc=0,size=4,imm=0):
    # Rules-only fixture, not asserted to be a complete machine encoding.
    return {'pc':pc,'size':size,'raw':(imm&65535).to_bytes(2,'little').hex()+'00'*(size-2),
            'op':op,'args':args,'text':op+' '+args,'vopd_independent_decode':False}

def seq(*spec):
    return [instruction(x[0],x[1]if len(x)>1 else'',pc=i*4,imm=x[2]if len(x)>2 else 0) for i,x in enumerate(spec)]

def replace_function(data,newcode):
    from radisa.binary import ELF
    e=ELF(data);s=e.named_symbol('smoke');sec=e.sections[s.section]
    assert len(newcode)==s.size
    start=sec.offset+s.value-sec.address
    return data[:start]+newcode+data[start+s.size:]

# ---- toolchain capabilities and whole-kernel assembly helpers (radisa check tests) ----
import functools, io, json, os, subprocess, tempfile, unittest
from contextlib import redirect_stdout
CHECK_FIX=ROOT/'fixtures'/'check'

def require_llvm():
    """RAD_ISA_REQUIRE_LLVM=1 (check.sh): a missing toolchain capability is a failure, never a skip."""
    return os.environ.get('RAD_ISA_REQUIRE_LLVM')=='1'

@functools.lru_cache(maxsize=None)
def llvm_mc():
    """Path of the llvm-mc `radisa check` would use, or None."""
    from radisa import hipsrc
    try:return hipsrc.find_tool('llvm-mc',None)
    except Exception:return None

@functools.lru_cache(maxsize=None)
def llvm_mc_version():
    """(major, minor, patch) of llvm_mc(), or None."""
    import re
    if not llvm_mc():return None
    p=subprocess.run([llvm_mc(),'--version'],capture_output=True,text=True,check=False)
    m=re.search(r'LLVM version (\d+)\.(\d+)\.(\d+)',p.stdout)
    return tuple(map(int,m.groups()))if m else None

@functools.lru_cache(maxsize=None)
def assembles(text):
    """'' if llvm-mc assembles `text` for gfx1201 (as `radisa check` invokes it), else why not."""
    if not llvm_mc():return 'llvm-mc not found'
    with tempfile.TemporaryDirectory()as d:
        s=os.path.join(d,'probe.s');Path(s).write_text(text)
        p=subprocess.run([llvm_mc(),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx1201','-filetype=obj',s,'-o',os.path.join(d,'probe.o')],
                         capture_output=True,text=True,check=False)
    return '' if p.returncode==0 else (p.stderr.strip().splitlines() or ['llvm-mc failed'])[0]

def _need(why,what):
    if not why or require_llvm():return lambda f:f
    return unittest.skip(f'{what}: {why} (llvm-mc {".".join(map(str,llvm_mc_version() or ()))or"?"}; '
                         'set RAD_ISA_REQUIRE_LLVM=1 to require it)')

def needs_llvm_mc(f):
    """Skip unless llvm-mc assembles gfx1201 code (never skipped with RAD_ISA_REQUIRE_LLVM=1)."""
    return _need('' if llvm_mc() else 'llvm-mc not found','needs llvm-mc')(f)

def needs_kernel_asm(f):
    """Skip unless llvm-mc assembles compiler-emitted kernels of AMD clang 20+ (fixtures/check/kernel_reduce.s:
    `instprefsize`, the 4-component target id); LLVM 18 does not. Never skipped with RAD_ISA_REQUIRE_LLVM=1."""
    return _need(assembles((CHECK_FIX/'kernel_reduce.s').read_text()),'needs a newer llvm-mc (kernel_reduce.s)')(f)

def needs_minimal_kernel(f):
    """Skip unless llvm-mc assembles kernel_asm() output (LLVM 18 does)."""
    return _need(assembles(kernel_asm('s_endpgm')),'needs llvm-mc for kernel_asm()')(f)

@functools.lru_cache(maxsize=None)
def hip_compiles(cxx,source):
    """'' if `cxx` compiles the HIP file `source` for gfx1201 (device only, -nogpuinc -nogpulib), else why not."""
    if not cxx:return 'no HIP-capable clang'
    with tempfile.TemporaryDirectory()as d:
        p=subprocess.run([cxx,'-x','hip','--offload-arch=gfx1201','--offload-device-only','-nogpuinc','-nogpulib',
                          '-O1','-S',str(source),'-o',os.path.join(d,'probe.s')],capture_output=True,text=True,check=False)
    if p.returncode==0:return ''
    errs=[l for l in p.stderr.splitlines() if 'error:' in l]
    return (errs or p.stderr.strip().splitlines() or ['compile failed'])[0]

def needs_hip(cxx,source):
    """Skip unless `cxx` compiles `source` for gfx1201 (e.g. clang 18 lacks the 3-argument
    __builtin_amdgcn_fence). Never skipped with RAD_ISA_REQUIRE_LLVM=1."""
    return _need(hip_compiles(cxx,str(source)),f'needs a clang that compiles {Path(source).name}')

def kernel_asm(body,name='k',vgprs=8,lds=0,extra_functions='',sgprs=16,kernarg_size=16):
    """Text of one complete wave32 gfx1201 kernel (code object v5) that llvm-mc 18 and newer assemble
    and radisa.binary.kernel_inventory accepts: s[0:1] = kernarg pointer, s2 = workgroup id x,
    v0 = workitem id x. `body` holds instructions, one per line (end it with s_endpgm yourself);
    `vgprs`/`lds` go to both the descriptor and the metadata (the descriptor allocates VGPRs in
    granules of 8); `extra_functions` is assembly placed after the kernel in .text (callees).
    The target id is the 3-component form `amdgcn-amd-amdhsa--gfx1201`: LLVM 18 accepts only
    that one, newer LLVM checks compatibility of the triple instead of string equality."""
    code=''.join(f'\t{l.strip()}\n' for l in body.strip('\n').splitlines() if l.strip())
    return f'''\t.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
\t.amdhsa_code_object_version 5
\t.text
\t.globl\t{name}
\t.p2align\t8
\t.type\t{name},@function
{name}:
{code}.Lfunc_end_{name}:
\t.size\t{name}, .Lfunc_end_{name}-{name}
{extra_functions.rstrip()+chr(10) if extra_functions.strip() else ''}\t.rodata
\t.p2align\t6
\t.amdhsa_kernel {name}
\t\t.amdhsa_group_segment_fixed_size {lds}
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size {kernarg_size}
\t\t.amdhsa_user_sgpr_count 2
\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1
\t\t.amdhsa_wavefront_size32 1
\t\t.amdhsa_system_sgpr_workgroup_id_x 1
\t\t.amdhsa_system_vgpr_workitem_id 0
\t\t.amdhsa_next_free_vgpr {vgprs}
\t\t.amdhsa_next_free_sgpr {sgprs}
\t.end_amdhsa_kernel
\t.text
\t.amdgpu_metadata
---
amdhsa.kernels:
  - .name:           {name}
    .symbol:         {name}.kd
    .group_segment_fixed_size: {lds}
    .private_segment_fixed_size: 0
    .kernarg_segment_size: {kernarg_size}
    .kernarg_segment_align: 8
    .max_flat_workgroup_size: 256
    .sgpr_count:     {sgprs}
    .sgpr_spill_count: 0
    .vgpr_count:     {vgprs}
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...
\t.end_amdgpu_metadata
'''

def run_check(path,*argv):
    """`radisa check path *argv --json ...` in process: (exit code, report dict); stdout is discarded."""
    from radisa.cli import main
    with tempfile.TemporaryDirectory()as d,redirect_stdout(io.StringIO()):
        j=os.path.join(d,'r.json')
        code=main(['check',str(path),*map(str,argv),'--json',j])
        report=json.loads(Path(j).read_text())
    return code,report

def check_text(text,*argv,suffix='.s'):
    """run_check on `text` written to a temporary file."""
    with tempfile.TemporaryDirectory()as d:
        p=Path(d,'input'+suffix);p.write_text(text)
        return run_check(p,*argv)

def all_findings(report):
    return report['findings']+[f for k in report['kernels'] for f in k['findings']]

def check_args(path,*argv):
    """The argparse namespace `radisa check path *argv` runs with (for check.analyze)."""
    import argparse
    from radisa.check import add_parser
    p=argparse.ArgumentParser();add_parser(p.add_subparsers(dest='command'))
    return p.parse_args(['check',str(path),*map(str,argv)])
