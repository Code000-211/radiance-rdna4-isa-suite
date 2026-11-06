"""Interleaved, stop-on-failure A/B experiments. No throughput or production claims.
This compares separate supervised processes in ABBA order. It is not simultaneous
multi-GPU testing; environmental drift within a block can remain.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time
from .format import require,read_json,validate,file_hash,digest,all_blobs
from .corpus import write


def workload_identity(path:Path)->str:
    validate(path);m=copy.deepcopy(read_json(path))
    # Only binary identities and deliberate candidate lineage differ; row names,
    # frozen tuning, inputs, expected bytes and tolerances must remain identical.
    m.pop('id');m.pop('lineage',None)
    for p in m['plugins']:p.pop('sha256')
    for b in all_blobs(m):b.pop('path')
    return digest(json.dumps(m,sort_keys=True,separators=(',',':')).encode())


def summarize(blocks:list[dict],seed:int=0)->dict:
    require(len(blocks)>=3,'at least three complete ABBA blocks required')
    ratios=[]
    for b in blocks:
        require(len(b['a_us'])==len(b['b_us'])==2,'incomplete ABBA block')
        require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in b['a_us']+b['b_us']),'invalid per-run median')
        ratios.append(statistics.mean(b['b_us'])/statistics.mean(b['a_us']))
    rng=random.Random(seed)
    estimates=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(4000))
    return {'estimator':'median of paired block mean ratios (candidate / baseline)',
            'ratio':statistics.median(ratios),'block_ratios':ratios,
            'bootstrap_interval_95':[estimates[100],estimates[3899]],
            'bootstrap_unit':'whole ABBA block','bootstrap_seed':seed,
            'interval_warning':'Exploratory block bootstrap; very few blocks have weak resolution. Not a significance or production guarantee.',
            'individual_dispatch_percentiles':False,'tokens_per_second_estimated':False}


def experiment(args, invoke=None):
    from .cli import main
    if invoke is None:invoke=main
    out=Path(args.out).resolve();require(not out.exists(),'A/B output exists')
    require(args.execute_trusted_code,'A/B execution requires --execute-trusted-code')
    require(3<=args.blocks<=1000 and 1<=args.warmups<=1000 and 3<=args.reps<=10000,'use >=3 blocks, >=1 warmup, >=3 repetitions')
    a=Path(args.baseline).resolve();b=Path(args.candidate).resolve()
    identity=workload_identity(a);require(identity==workload_identity(b),'A/B workloads, row/tuning or tolerances differ')
    ah=file_hash(a);bh=file_hash(b)
    require(read_json(a)['target']==read_json(b)['target'],'A/B target mismatch')
    out.mkdir(parents=True,mode=0o700)
    result={'schema':'radiance-replay-ab/1','status':'INCOMPLETE','order':'ABBA',
            'workload_sha256':identity,'baseline_manifest_sha256':ah,'candidate_manifest_sha256':bh,
            'requested_blocks':args.blocks,'blocks':[],'runs':[], 'production_equivalence_proved':False,
            'baseline_worker_sha256':file_hash(Path(args.baseline_worker)),
            'candidate_worker_sha256':file_hash(Path(args.candidate_worker)),
            'started_monotonic':time.monotonic()}
    code=2;environment=None
    try:
        for bi in range(args.blocks):
            block={'index':bi,'a_us':[],'b_us':[]}
            for si,arm in enumerate('abba'):
                require(file_hash(a)==ah and file_hash(b)==bh,'fixture changed during A/B')
                require(file_hash(Path(args.baseline_worker))==result['baseline_worker_sha256'] and file_hash(Path(args.candidate_worker))==result['candidate_worker_sha256'],'worker changed during A/B')
                prefix='baseline' if arm=='a' else 'candidate';path=a if arm=='a' else b
                runout=out/f'{bi:04d}-{si}-{arm}'
                cmd=['run',str(path),'--out',str(runout),'--worker',str(getattr(args,prefix+'_worker')),
                     '--execute-trusted-code','--diagnostics','contracts','--warmups',str(args.warmups),
                     '--reps',str(args.reps),'--timeout',str(args.timeout),'--device',str(args.device)]
                for bind in getattr(args,prefix+'_plugin'):cmd+=['--plugin',bind]
                for flag in ('isa_manifest','isa_report'):
                    v=getattr(args,prefix+'_'+flag,None)
                    if v:cmd+=['--'+flag.replace('_','-'),str(v)]
                if args.gate:cmd+=['--gate']
                rc=invoke(cmd);report=read_json(runout/'report.json');rt=report.get('runtime',{})
                result['runs'].append({'block':bi,'slot':si,'arm':arm,'path':str(runout.relative_to(out)/'report.json'),
                                       'sha256':file_hash(runout/'report.json'),'status':report.get('status'),'exit_code':rc})
                if rc!=0 or report.get('status')!='RUNTIME_CHECKS_PASS':
                    code=1 if rc==1 or report.get('status')=='FAIL' else 2
                    result['status']='FAIL' if code==1 else 'INCOMPLETE'
                    result['error']='Stopped on first nonpassing run; no subsequent GPU submission';return code,result
                require(report.get('exit_code')==0 and report.get('runtime_executed') is True,'pass without execution')
                require(rt.get('fixture_sha256')==file_hash(path),'A/B report fixture mismatch')
                require(report.get('worker_sha256')==result[prefix+'_worker_sha256'],'A/B worker hash mismatch')
                require(rt.get('checked_iterations')==args.warmups+args.reps,'unchecked A/B iteration')
                require(rt.get('diagnostics',{}).get('status')=='PASS','missing A/B diagnostics')
                require(not rt.get('worker_build',{}).get('host_asan') and not report.get('options',{}).get('diagnostic_only'),'instrumented/diagnostic run cannot produce benchmark evidence')
                values=rt.get('samples_us',[])
                require(len(values)==args.reps and all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in values),'missing/nonpositive A/B samples')
                dev=rt.get('device',{})
                env=(report.get('host'),rt.get('backend'),rt.get('measurement'),{k:dev.get(k) for k in ('arch','name','ordinal','pci_address','runtime_version')})
                if environment is None:environment=env
                require(environment==env,'A/B environment/device/measurement mismatch')
                block[arm+'_us'].append(statistics.median(values))
            result['blocks'].append(block)
        result['summary']=summarize(result['blocks']);result['status']='AB_MEASURED';code=0
        return code,result
    except (ValueError,OSError,KeyError,TypeError) as e:
        result['error']=str(e);return 2,result
    finally:
        result['exit_code']=code;write(out/'ab.json',result)
        (out/'ab.md').write_text('# Interleaved kernel-window comparison\n\n**'+result['status']+'**\n\n```json\n'+json.dumps(result.get('summary',{}),indent=2)+'\n```\n\nAll attempts retained. Serial contract diagnostics precede each uninstrumented measurement. Neither driver clock state nor production equivalence is proved.\n')
