"""Small independent CPU oracles. No imports from Radiance/libref, no NumPy.
These define f32 dense math only; quantized packing/rounding needs a separate adapter.
"""
import math
import struct
from .format import require

def f32(x: float) -> float:
    return struct.unpack('<f',struct.pack('<f',x))[0]

def pack_f32(values):
    return b''.join(struct.pack('<f',x) for x in values)

def gemm_nt(a, b, m: int, n: int, k: int):
    require(all(type(x) is int and x>0 for x in (m,n,k)),'invalid oracle shape')
    require(len(a)==m*k and len(b)==n*k,'oracle storage/shape mismatch')
    require(all(math.isfinite(x) for x in (*a,*b)),'finite-input oracle only')
    # Higher-precision accumulation, rounding once to f32: tolerance must account
    # for the different reduction algorithm of the device implementation.
    return [f32(math.fsum(float(a[i*k+t])*float(b[j*k+t]) for t in range(k)))
            for i in range(m) for j in range(n)]

def boundaries(tile: int, *, multiples=(1,2), maximum: int | None = None):
    require(type(tile) is int and tile>0,'positive tile required')
    values={1}
    for multiple in multiples:
        require(type(multiple) is int and multiple>0,'positive tile multiple required')
        for delta in (-1,0,1):
            x=tile*multiple+delta
            if x>0 and (maximum is None or x<=maximum):values.add(x)
    return sorted(values)
