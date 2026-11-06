#!/usr/bin/env python3
"""Executable HOST-only GEMM boundary corpus; not invented GPU or production evidence.
One-axis-at-a-time shapes keep this independent-reference corpus small.  Use a
kernel-specific adapter for actual GPU geometry constraints/quantized layouts.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from radreplay.builder import Package
from radreplay.corpus import expect
from radreplay.oracles import boundaries, gemm_nt, pack_f32
from radreplay.format import require, file_hash

POLICY={'schema':'radiance-numeric-policy/1','version':'dense-f32-1',
        'scope':'finite dense f32 GEMM; independent Python fsum oracle; not quantization accuracy',
        'op':'gemm_nt','dtype':1,'atol':2e-5,'rtol':2e-5,'rel_l2':2e-5,'nonfinite':'reject'}

def build(root: Path, plugin: Path):
    root=Path(root);require(not root.exists(),'corpus output already exists');root.mkdir(parents=True)
    shapes=set()
    for tile in (16,32,64,96,128):
        for x in boundaries(tile):shapes.add((x,3,5))
    for x in boundaries(16)+boundaries(32):shapes.add((2,x,5));shapes.add((2,3,x))
    # Adversarial but valid input distributions; each expected output is independent.
    cases=[];index=[]
    for m,n,k in sorted(shapes):
        name=f'gemm-m{m}-n{n}-k{k}'
        a=[float((i*7)%9-4) for i in range(m*k)]
        b=[float((i*5)%7-3) for i in range(n*k)]
        path=make(root/name,name,plugin,m,n,k,a,b)
        cases.append(path);index.append({'manifest':str(path.relative_to(root)),'M':m,'N':n,'K':k,'distribution':'signed-exact'})
    for kind in ('zeros','cancellation','outlier','rounding'):
        m,n,k=2,3,33
        if kind=='zeros':a=[0.0]*(m*k);b=[1.0]*(n*k)
        elif kind=='cancellation':a=[(-1.0 if i%2 else 1.0) for i in range(m*k)];b=[1.0]*(n*k)
        elif kind=='outlier':a=[0.125]*(m*k);a[0]=1024.0;b=[0.25]*(n*k)
        else:a=[(i%5-2)*0.125 for i in range(m*k)];b=[(i%7-3)*0.0625 for i in range(n*k)]
        name='gemm-'+kind;path=make(root/name,name,plugin,m,n,k,a,b);cases.append(path)
        index.append({'manifest':str(path.relative_to(root)),'M':m,'N':n,'K':k,'distribution':kind})
    (root/'numeric-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    expect(cases,root/'expected.json','HOST-only dense f32 reference corpus, one-axis boundaries and finite value patterns; not all Radiance kernels')
    (root/'matrix.json').write_text(json.dumps({'schema':'radiance-boundary-corpus/1','cases':index,
        'numeric_policy_sha256':file_hash(root/'numeric-policy.json'),'all_kernel_coverage':False},indent=2)+'\n')
    return cases

def make(root,name,plugin,m,n,k,a,b):
    p=Package(root,name,note='Independent dense-f32-1 Python oracle; synthetic finite host inputs.')
    p.plugin('libref',plugin);p.stream()
    for key,data,shape,mut in [('a',a,[m,k],False),('b',b,[n,k],False),('y',[0.0]*(m*n),[m,n],True)]:
        raw=pack_f32(data);p.allocation(key,raw,mutable=mut);p.view(key,key,nbytes=len(raw),dtype=1,shape=shape)
    p.kernel('gemm','p0','ref_gemm_nt','gemm_nt',['a','b','y'],[('a','r'),('b','r'),('y','w')],params={'M':m,'N':n,'K':k,'dtype':'f32'})
    p.check('y',pack_f32(gemm_nt(a,b,m,n,k)),mode='float',**{x:POLICY[x] for x in ('atol','rtol','rel_l2','nonfinite')})
    return p.finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--plugin',required=True)
    a=p.parse_args();print(f'Created {len(build(Path(a.out),Path(a.plugin)))} explicit host fixtures')
