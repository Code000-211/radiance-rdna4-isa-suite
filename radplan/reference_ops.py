"""Small independent CPU mathematical oracles, not GPU emulation or timing models."""
from __future__ import annotations
import math
from .common import need


def _shape(a):
    need(isinstance(a,list) and a and all(isinstance(r,list) and r for r in a),'nonempty matrix required')
    n=len(a[0]);need(all(len(r)==n for r in a),'ragged matrix')
    need(all(type(x) in (float,int) and math.isfinite(x) for r in a for x in r),'finite values required')
    return len(a),n


def attention_reference(q,k,v,visible,*,chunk_size=None):
    """One head. visible[q] contains exact token indices (including tail)."""
    nq,d=_shape(q);nk,dk=_shape(k);nv,dv=_shape(v)
    need(d==dk and nk==nv and len(visible)==nq,'attention dimensions')
    if chunk_size is not None:need(type(chunk_size) is int and chunk_size>0,'positive chunk size')
    result=[]
    for query,ids in zip(q,visible):
        need(isinstance(ids,list) and ids and len(ids)==len(set(ids)),'unique nonempty attention visibility')
        need(all(type(i) is int and 0<=i<nk for i in ids),'attention token index')
        scores=[math.fsum(a*b for a,b in zip(query,k[i]))/math.sqrt(d) for i in ids]
        if chunk_size is None:
            m=max(scores);weights=[math.exp(x-m) for x in scores];den=math.fsum(weights)
            result.append([math.fsum(w*v[i][j] for i,w in zip(ids,weights))/den for j in range(dv)])
        else:
            m=-math.inf;den=0.;acc=[0.]*dv
            for z in range(0,len(ids),chunk_size):
                ix=ids[z:z+chunk_size];sc=scores[z:z+chunk_size];nm=max(m,max(sc));old=math.exp(m-nm)
                ws=[math.exp(x-nm) for x in sc]
                acc=[old*acc[j]+math.fsum(w*v[i][j] for i,w in zip(ix,ws)) for j in range(dv)]
                den=old*den+math.fsum(ws);m=nm
            result.append([x/den for x in acc])
    return result


def gated_expert(x,gate,up,down):
    nt,h=_shape(x);f,hg=_shape(gate);fu,hu=_shape(up);ho,fd=_shape(down)
    need(h==hg==hu==ho and f==fu==fd,'expert dimensions')
    outs=[]
    for row in x:
        a=[math.fsum(a*b for a,b in zip(row,g)) for g in gate]
        u=[math.fsum(a*b for a,b in zip(row,g)) for g in up]
        def silu(z):
            return z/(1+math.exp(-z)) if z>=0 else z*math.exp(z)/(1+math.exp(z))
        z=[silu(t)*v for t,v in zip(a,u)]
        outs.append([math.fsum(a*b for a,b in zip(z,r)) for r in down])
    return outs


def sharded_expert(x,gate,up,down,slices):
    need(slices and sorted(j for lo,hi in slices for j in range(lo,hi))==list(range(len(gate))),'partition must cover each column once')
    parts=[gated_expert(x,gate[lo:hi],up[lo:hi],[r[lo:hi] for r in down]) for lo,hi in slices]
    return [[math.fsum(p[t][j] for p in parts) for j in range(len(down))] for t in range(len(x))]
