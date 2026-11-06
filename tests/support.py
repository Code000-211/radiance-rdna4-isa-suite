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
