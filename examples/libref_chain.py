#!/usr/bin/env python3
"""A synthetic, exactly representable add->mul chain using real Radiance libref.
Not a GPU benchmark or production-captured input. No NumPy dependency.
"""
import argparse
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from radreplay.builder import Package

def build(root, plugin):
    p=Package(Path(root),'libref-add-mul',note='Synthetic integers; independently computed exact float32 oracle.')
    p.plugin('libref',Path(plugin));p.stream()
    # An alias view of y is reused in-place by the second operation.
    values={'a':[1,2,3,4], 'b':[4,3,2,1], 'y':[0,0,0,0]}
    for name,vals in values.items():
        p.allocation(name,struct.pack('<4f',*vals),mutable=name=='y')
        p.view(name,name,nbytes=16,dtype=1,shape=[1,4])
    p.view('y_alias','y',nbytes=16,dtype=1,shape=[1,4])
    params={'M':1,'n':4,'dtype':'f32'}
    p.kernel('add','p0','ref_add','add',['a','b','y'],[('a','r'),('b','r'),('y','w')],params=params,phase='prefix')
    p.kernel('mul','p0','ref_mul','mul',['y_alias','a','y'],[('y_alias','r'),('a','r'),('y','w')],params=params)
    p.check('y',struct.pack('<4f',5,10,15,20))
    return p.finish()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plugin',required=True);a.add_argument('--out',required=True)
    n=a.parse_args();print(build(n.out,n.plugin))
