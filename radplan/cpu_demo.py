"""Real Radiance libref + CPU candidate benchmarking through rad-kbench-replay.
All original report JSONs remain in the work directory. Never times GPU code on CPU.
"""
from __future__ import annotations
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import random
import statistics
from .common import need,digest,file_hash,write,new_dir
from .calibrate import fit
from .planner import plan,markdown
from .importer import import_replay

CONTRACT='finite-f32-contiguous-nt-independent-oracle'


def catalog(libref:Path,plugin:Path)->dict:
    items=[]
    for name,nb,path,row in [('libref',1,libref,'ref_gemm_nt'),('panel1',1,plugin,'cpu_panel1'),('panel4',4,plugin,'cpu_panel4'),('panel16',16,plugin,'cpu_panel16')]:
        items.append({'id':name,'family':name,'backend':'host','arch':'host','contract':CONTRACT,'compute_path':'cpu_f32',
            'tile':[1,nb,1],'wave_size':32,'waves_per_group':1,'stages':1,'lds_bytes':0,
            'tail':{'M':True,'N':True,'K':True},'multiples':{'M':1,'N':1,'K':1},'work_policy':'exact',
            'identity':{'kind':'plugin','sha256':file_hash(path),'provenance':'compiled CPU code, not GPU emulation'},
            'launch':{'row':row,'op':'gemm_nt','params':{'dtype':'f32'}},'unpack_ops_per_weight':0.,'matrix_alu_overlap':True,
            'format':{'a_bits':32,'b_bits':32,'out_bits':32,'weight_group':1,'weight_scale_bytes':0,'activation_scale_bytes':0},
            'notes':'CPU panel tile is explicit. libref tile=[1,1,1] is a feature proxy, not its actual internal tiling.',
            'input_layout':'f32-contiguous','output_layout':'f32-contiguous'})
    return {'schema':'radiance-planner-catalog/1','candidates':items,'notes':'CPU experiment only; libref is genuine source, panel candidates are new diagnostic implementations'}


def make_fixture(root:Path,w:dict,c:dict,plugin:Path,a:list,b:list,expected:bytes)->Path:
    from radreplay.builder import Package
    from radreplay.oracles import pack_f32
    m,n,k=w['M'],w['N'],w['K']
    package=Package(root,w['id']+'-'+c['id'],note='CPU-only planner comparison; independent fsum oracle; not production activation snapshot.')
    package.plugin('libref' if c['id']=='libref' else 'planner_cpu',plugin);package.stream()
    for name,data,shape,mutable in [('a',pack_f32(a),[m,k],False),('b',pack_f32(b),[n,k],False),('y',bytes(4*m*n),[m,n],True)]:
        package.allocation(name,data,mutable=mutable);package.view(name,name,nbytes=len(data),dtype=1,shape=shape)
    package.kernel('gemm','p0',c['launch']['row'],'gemm_nt',['a','b','y'],[('a','r'),('b','r'),('y','w')],params={'M':m,'N':n,'K':k,'dtype':'f32'})
    package.check('y',expected,mode='float',atol=2e-5,rtol=2e-5,rel_l2=2e-5)
    return package.finish()


def run(out:Path,worker:Path,libref:Path,plugin:Path,*,seed:int=7,reps:int=15)->dict:
    from radreplay.cli import main as replay_main
    from radreplay.oracles import gemm_nt,pack_f32
    need(reps>=3,'at least 3 repetitions required')
    for f in (worker,libref,plugin):need(f.is_file(),'compiled prerequisite missing: '+str(f))
    root=new_dir(out);cat=catalog(libref,plugin)
    # Forty distinct shape groups, not a random split of repetitions of the same shape.
    shapes=[]
    for idx,m in enumerate((1,2,5,8,16,31,32,33,63,64)):
        for j,n in enumerate((16,32,64,96)):
            shapes.append((m,n,(32,64,96,128)[(idx+j)%4]))
    rng=random.Random(seed);rng.shuffle(shapes)
    records=[];runlog=[];p=None
    for si,(m,n,k) in enumerate(shapes):
        w={'id':f'cpu-M{m}-N{n}-K{k}','op':'gemm_nt','M':m,'N':n,'K':k,'a_bits':32,'b_bits':32,'out_bits':32,
           'contract':CONTRACT,'cache_state':'warm','note':'restore copies precede timing; warmed CPU inputs, not an asserted GPU cache state'}
        a=[((i*7)%9-4)*.125 for i in range(m*k)];b=[((i*5)%7-3)*.25 for i in range(n*k)]
        expected=pack_f32(gemm_nt(a,b,m,n,k))
        order=list(cat['candidates']);rng.shuffle(order)
        for c in order:
            binary=libref if c['id']=='libref' else plugin
            fixture=make_fixture(root/'fixtures'/f"{w['id']}-{c['id']}",w,c,binary,a,b,expected)
            dest=root/'runs'/f"{w['id']}-{c['id']}"
            args=['run',str(fixture),'--out',str(dest),'--worker',str(worker),'--plugin','p0='+str(binary),'--execute-trusted-code',
                  '--warmups','2','--reps',str(reps),'--timeout','30','--diagnostics','contracts']
            with contextlib.redirect_stdout(io.StringIO()):rc=replay_main(args)
            r=json.loads((dest/'report.json').read_text());rt=r.get('runtime',{})
            runlog.append({'fixture_sha256':file_hash(fixture),'report_sha256':file_hash(dest/'report.json'),'shape':[m,n,k],'candidate':c['id'],'exit_code':rc})
            if rc:
                write(root/'FAILED_RUNS.json',runlog);raise RuntimeError('CPU experiment stopped on first nonpassing run: '+str(dest))
            if p is None:
                p={'schema':'radiance-planner-profile/1','id':'authoring-host-only','backend':'host','arch':'host',
                   'environment':{'device_name':rt['device']['name'],'worker_sha256':r['worker_sha256'],'measurement_mode':rt['measurement'],
                       'platform':platform.platform(),'compiler':rt['worker_build']['compiler'],'clock_power':'not controlled; authoring container',
                       'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [],'cpu_threading':'single-threaded kernels; no OpenMP in libref build'},
                   'spec':{'peak_gops':{'cpu_f32':10.0},'bandwidth_gbs':10.,'compute_units':1,'simds_per_cu':1,'max_threads':1024,'max_lds_per_group':65536},
                   'assumptions':{'launch_us':.05,'compute_efficiency':1.,'memory_efficiency':1.,'lds_gbs':1.,'alu_gops':1.,'reuse_fraction':1.,'latency_hiding_waves':1},
                   'notes':'CPU peak numbers are illustrative baseline coefficients, NOT hardware measurements. Only empirical held-out model evaluates this machine. No GPU transfer.'}
            records.append(import_replay(dest/'report.json',fixture,w,c,p))
    dataset={'schema':'radiance-planner-measurements/1','profile_sha256':digest(p),'records':records,
             'notes':'Randomized shape/candidate order; repeated single-operation windows. Shared authoring CPU, no power/frequency control. No GPU timings.'}
    calibration=fit(dataset,cat,p,seed=seed,ncal=10,ntest=10)
    # Compare with the simplest theoretical CPU baseline trained ONLY on train groups:
    # per-candidate time proportional to useful operation count (no intercept).
    training=set(calibration['split']['train']);testing=set(calibration['split']['test'])
    from .model import shape_key
    slopes={c['id']:statistics.median(statistics.median(r['samples_us'])/(2*r['workload']['M']*r['workload']['N']*r['workload']['K']) for r in records if r['candidate_id']==c['id'] and shape_key(r['workload']) in training) for c in cat['candidates']}
    errs=[];regrets=[]
    for g in testing:
        rs=[r for r in records if shape_key(r['workload'])==g]
        pred=lambda r:slopes[r['candidate_id']]*2*r['workload']['M']*r['workload']['N']*r['workload']['K']
        actual=lambda r:statistics.median(r['samples_us'])
        chosen=min(rs,key=pred);regrets.append(actual(chosen)/min(map(actual,rs))-1)
        errs += [abs(pred(r)/actual(r)-1) for r in rs]
    comparison={'schema':'radiance-planner-cpu-comparison/1','shape_groups':40,'candidate_count':4,'replay_executions':len(records),
        'warmups_per_run':2,'timed_repetitions_per_run':reps,'all_correctness_checks_passed':True,
        'simple_ops_only_baseline':{'median_relative_latency_error':statistics.median(errs),'median_selection_regret':statistics.median(regrets),'max_selection_regret':max(regrets)},
        'hybrid_surrogate_held_out':calibration['validation'],'gpu_execution':False,'cpu_to_gpu_conversion':False,'host_noise_controlled':False}
    heldout={shape_key(r['workload']):r['workload'] for r in records if shape_key(r['workload']) in testing}
    report=plan(cat,{'schema':'radiance-planner-workloads/1','workloads':list(heldout.values())},p,calibration)
    for name,obj in [('catalog.json',cat),('profile.json',p),('measurements.json',dataset),('calibration.json',calibration),('comparison.json',comparison),('plan.json',report),('run-index.json',runlog)]:write(root/name,obj)
    (root/'plan.md').write_text(markdown(report))
    return comparison
