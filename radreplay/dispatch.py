"""Bind observed *diagnostic* dispatches to an already strictly checked code inventory.
A clean result concerns the declared ordered-launcher route, not all HIP/tape/SDMA
or background-thread paths. The opt-in backend hook refuses unapproved identities
before queue reservation; CPU unit tests do not certify that GPU integration.
"""
from pathlib import Path
import math
from .format import require,read_json,file_hash,HASH


def allowed_identities(isa_manifest:dict)->set[str]:
    return {o['sha256']+':'+k['symbol'] for o in isa_manifest['objects'] for k in o['kernels']}


def bind(trace_path:Path,fixture:dict,fixture_hash:str,isa_manifest:dict,device:int)->dict:
    t=read_json(trace_path)
    require(t.get('schema')=='radiance-dispatch-trace/1','wrong dispatch trace schema')
    require(t.get('fixture_sha256')==fixture_hash and t.get('completed') is True and not t.get('error'),'incomplete/stale dispatch trace')
    allowed=allowed_identities(isa_manifest);require(allowed,'empty static whitelist')
    entries=t.get('entries');require(isinstance(entries,list) and 0<len(entries)<=1000000,'empty/oversized dispatch trace')
    nodes={n['id']:n for n in fixture['nodes']};seen=set();observed=set();support=set()
    for e in entries:
        require(e.get('identity') in allowed,'observed unapproved code object/symbol')
        require(type(e.get('device')) is int and e['device']==device,'dispatch device mismatch')
        require(type(e.get('queue')) is int and 0<=e['queue']<=2**32-1,'invalid queue id')
        for key in ('grid','group'):
            v=e.get(key);require(isinstance(v,list) and len(v)==3 and all(type(x) is int and 0<x<=(65535 if key=='group' else 2**32-1) for x in v),'invalid dispatch geometry')
        require(math.prod(e['group'])<=1024,'invalid workgroup size')
        require(type(e.get('group_bytes')) is int and 0<=e['group_bytes']<=65536,'invalid LDS extent')
        require(type(e.get('private_bytes')) is int and 0<=e['private_bytes']<=2**32-1,'invalid private extent')
        phase=e.get('phase');node=e.get('node');pattern=e.get('pattern')
        require(phase in ('setup','diagnostic-node','diagnostic-support'),'unmodeled dispatch route/phase')
        require(node=='' or node in nodes,'trace has unknown node')
        require(type(pattern) is int and (pattern==-1 if phase=='setup' else pattern in (0,1,2)),'missing diagnostic pass')
        observed.add(e['identity'])
        if phase=='diagnostic-node':
            require(node in nodes,'dispatch node absent')
            if nodes[node]['kind']=='kernel':seen.add((node,pattern))
        else:support.add(e['identity'])
    expected={(n['id'],p) for n in fixture['nodes'] if n['kind']=='kernel' for p in range(3)}
    require(expected and expected<=seen,'some kernel nodes/passes have no observed dispatch')
    return {'status':'DISPATCH_BINDING_PASS','trace_sha256':file_hash(trace_path),'entries':len(entries),
            'observed_identities':sorted(observed),'support_identities':sorted(support),
            'kernel_node_passes_observed':len(seen),'scope':'observed diagnostic ordered-launcher dispatches only',
            'timed_dispatches_observed':False,'all_kernel_coverage':False,'raw_tape_covered':False}
