"""Checked host paged-attention fixtures through genuine Radiance libref.
No GPU emulation, no fitting of GPU rates, no comparison of CPU and GPU latencies.
"""
from __future__ import annotations
import contextlib,io,json,math,struct
from pathlib import Path
from .common import need,new_dir,write,file_hash,digest
from .reference_ops import attention_reference


def specs():
    cases=[]
    for context in (1,3,4,5,17,33):
        for window in (0,4):
            cases.append({'id':f'c{context}.window{window}','contexts':[context],'query_counts':[1],
                          'head_dim':8,'kv_heads':2,'gqa':3,'block_size':4,'causal':1,'window':window})
    for queries in (2,3,5):
        cases.append({'id':f'verify{queries}','contexts':[33],'query_counts':[queries],
                      'head_dim':16,'kv_heads':2,'gqa':3,'block_size':4,'causal':1,'window':0})
    cases.append({'id':'ragged.batch','contexts':[5,17],'query_counts':[1,3],
                  'head_dim':8,'kv_heads':2,'gqa':2,'block_size':4,'causal':1,'window':0})
    cases.append({'id':'ragged.window','contexts':[5,17],'query_counts':[1,3],
                  'head_dim':8,'kv_heads':2,'gqa':2,'block_size':4,'causal':1,'window':4})
    cases.append({'id':'noncausal','contexts':[17],'query_counts':[3],
                  'head_dim':8,'kv_heads':1,'gqa':2,'block_size':4,'causal':0,'window':0})
    for ratio in (6,12):
        cases.append({'id':f'h256.gqa{ratio}.verify5','contexts':[9],'query_counts':[5],
                      'head_dim':256,'kv_heads':1,'gqa':ratio,'block_size':4,'causal':1,'window':0})
    return cases


def fixture(root:Path,spec:dict,plugin:Path):
    from radreplay.builder import Package
    from radreplay.oracles import pack_f32
    bs=spec['block_size'];d=spec['head_dim'];kh=spec['kv_heads'];qh=kh*spec['gqa']
    ctxs=spec['contexts'];qs=spec['query_counts'];nq=sum(qs);nb=[(n+bs-1)//bs for n in ctxs]
    total_blocks=sum(nb);physical=list(reversed(range(total_blocks)));max_blocks=max(nb)
    bt=[];keys=[];vals=[];cache=[0.]*(total_blocks*kh*bs*2*d);cursor=0
    for seq,(ctx,nblocks) in enumerate(zip(ctxs,nb)):
        ids=physical[cursor:cursor+nblocks];cursor+=nblocks;bt+=ids+[0]*(max_blocks-nblocks)
        ks=[];vs=[]
        for h in range(kh):
            k=[[((seq*31+h*7+t*3+j)%13-6)/16 for j in range(d)] for t in range(ctx)]
            v=[[((seq*17+h*5+t*7+j*3)%17-8)/8 for j in range(d)] for t in range(ctx)]
            ks.append(k);vs.append(v)
            for t in range(ctx):
                off=((ids[t//bs]*kh+h)*bs+t%bs)*2*d
                cache[off:off+d]=k[t];cache[off+d:off+2*d]=v[t]
        keys.append(ks);vals.append(vs)
    queries=[[[((t*7+h*3+j)%11-5)/16 for j in range(d)] for h in range(qh)] for t in range(nq)]
    expected=[[[0.]*d for _ in range(qh)] for _ in range(nq)];cursor=0
    for seq,(ctx,count) in enumerate(zip(ctxs,qs)):
        visibility=[]
        for t in range(count):
            pos=ctx-count+t;end=pos+1 if spec['causal'] else ctx
            visibility.append([i for i in range(end) if not spec['window'] or (pos-i<spec['window'] if spec['causal'] else abs(pos-i)<spec['window'])])
        for h in range(qh):
            out=attention_reference([queries[cursor+t][h] for t in range(count)],keys[seq][h//spec['gqa']],vals[seq][h//spec['gqa']],visibility)
            # Online chunked softmax is a second math implementation only; it is
            # not a compiled Radiance GPU kernel and is never used for GPU timing.
            chunked=attention_reference([queries[cursor+t][h] for t in range(count)],keys[seq][h//spec['gqa']],vals[seq][h//spec['gqa']],visibility,chunk_size=4)
            need(all(abs(x-y)<1e-12 for a,b in zip(out,chunked) for x,y in zip(a,b)),'independent oracle disagreement')
            for t in range(count):expected[cursor+t][h]=out[t]
        cursor+=count
    flatten=lambda a:[x for t in a for h in t for x in h]
    p=Package(root,spec['id'],note='Synthetic finite f32 paged-attention host fixture; permuted pages; independent Python fsum/stable-softmax oracle. Not production activations.')
    p.plugin('libref',plugin);p.stream()
    items=[('q',pack_f32(flatten(queries)),1,[nq,qh,d],False),
           ('kv',pack_f32(cache),1,[total_blocks,kh,bs,2*d],False),
           ('bt',struct.pack('<'+'i'*len(bt),*bt),11,[len(qs),max_blocks],False),
           ('su',struct.pack('<'+'i'*len(ctxs),*ctxs),11,[len(ctxs)],False),
           ('out',bytes(nq*qh*d*4),1,[nq,qh,d],True)]
    cu=None
    if len(set(qs))>1:
        offsets=[0]
        for q in qs:offsets.append(offsets[-1]+q)
        cu='cu';items.append(('cu',struct.pack('<'+'i'*len(offsets),*offsets),11,[len(offsets)],False))
    for name,data,dt,shape,mutable in items:
        p.allocation(name,data,mutable=mutable);p.view(name,name,nbytes=len(data),dtype=dt,shape=shape)
    accesses=[(name,'w' if name=='out' else 'r') for name,_,_,_,_ in items]
    p.kernel('attention','p0','ref_attn_paged','attn_paged',['q','kv','bt','su',None,None,cu,'out'],accesses,
        params={'q_len':max(qs),'head_dim':d,'gqa':spec['gqa'],'block_size':bs,'causal':spec['causal'],
                'window':spec['window'],'q_dtype':'f32','kv_dtype':'f32','scale':1/math.sqrt(d)})
    p.check('out',pack_f32(flatten(expected)),mode='float',atol=3e-5,rtol=3e-5,rel_l2=3e-5)
    return p.finish()


def run(out:Path,worker:Path,libref:Path,reps:int=5):
    from radreplay.cli import main as replay_main
    need(worker.is_file() and libref.is_file(),'host worker/libref prerequisites missing')
    need(type(reps) is int and reps>=3,'at least 3 repetitions')
    root=new_dir(out);index=[]
    for sp in specs():
        fp=fixture(root/'fixtures'/sp['id'],sp,libref);dest=root/'runs'/sp['id']
        with contextlib.redirect_stdout(io.StringIO()):
            rc=replay_main(['run',str(fp),'--out',str(dest),'--worker',str(worker),'--plugin','p0='+str(libref),
                           '--execute-trusted-code','--warmups','1','--reps',str(reps),'--timeout','30','--diagnostics','contracts'])
        index.append({'case':sp,'fixture_sha256':file_hash(fp),'report_sha256':file_hash(dest/'report.json'),
                      'report':str(dest.relative_to(root)/'report.json'),'exit_code':rc})
        if rc:write(root/'FAILED_RUNS.json',index);raise RuntimeError('host attention stopped on first failure: '+sp['id'])
    summary={'schema':'radiance-v06-host-attention/1','checked_cases':len(index),'warmups_per_case':1,'timed_repetitions_per_case':reps,
             'all_checks_passed':True,'row':'ref_attn_paged','worker_sha256':file_hash(worker),'plugin_sha256':file_hash(libref),
             'oracle':'independent Python fsum + stable softmax; chunked recurrence cross-check',
             'gpu_execution':False,'gpu_model_calibration':False,'hardware_noise_controlled':False,
             'report_provenance':'actual genuine-source Radiance host replay executions, not synthetic timing data'}
    write(root/'run-index.json',index);write(root/'summary.json',summary);return summary
