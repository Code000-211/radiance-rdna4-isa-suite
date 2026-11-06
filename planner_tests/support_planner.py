"""Synthetic unit-test inputs. These are NOT benchmark evidence."""
from pathlib import Path
import copy
from radplan.common import read,digest
ROOT=Path(__file__).resolve().parents[1]

def gpu_profile():return read(ROOT/'planner/profiles/r9700-unmeasured.json')
def gpu_catalog():return read(ROOT/'planner/examples/radiance-w4a8-catalog.json')
def gpu_work(m=128,n=2560,k=640):
    w=read(ROOT/'planner/examples/w4a8-workloads.json')['workloads'][0]
    w.update(M=m,N=n,K=k,id=f'm{m}n{n}k{k}');return w

def cpu_profile():
    p=gpu_profile();p.update(id='synthetic-host',backend='host',arch='host',environment={'test':'synthetic, not measured'})
    p['spec']['peak_gops']={'cpu_f32':10};return p

def cpu_candidate(name='cpu',nb=4):
    c=copy.deepcopy(gpu_catalog()['candidates'][0])
    c.update(id=name,family=name,backend='host',arch='host',contract='f32',compute_path='cpu_f32',tile=[1,nb,1],
        wave_size=32,waves_per_group=1,stages=1,lds_bytes=0,tail={a:True for a in 'MNK'},multiples={a:1 for a in 'MNK'},
        work_policy='exact',identity={'kind':'plugin','sha256':'a'*64},
        launch={'row':name,'op':'gemm_nt','params':{'dtype':'f32'}},
        format={'a_bits':32,'b_bits':32,'out_bits':32,'weight_group':1,'weight_scale_bytes':0,'activation_scale_bytes':0},
        known_issues=[],unmodeled=[],unpack_ops_per_weight=0,lds_bytes_per_kblock=0)
    return c

def cpu_work(m=8,n=16,k=32):
    return {'id':f'm{m}n{n}k{k}','op':'gemm_nt','M':m,'N':n,'K':k,'a_bits':32,'b_bits':32,'out_bits':32,'contract':'f32','cache_state':'warm'}

def synthetic_data(groups=40):
    from radplan.model import features
    p=cpu_profile();cs=[cpu_candidate('a',4),cpu_candidate('b',8)]
    cat={'schema':'radiance-planner-catalog/1','candidates':cs};rows=[]
    for i in range(groups):
        w=cpu_work(i+1,16+(i%4)*16,16+(i%3)*16)
        for j,c in enumerate(cs):
            f=features(w,c,p)
            y=(1+j)+f['features']['executed_gops']*(10000 if j else 18000)+f['features']['output_melements']*300
            rows.append({'workload':copy.deepcopy(w),'candidate_id':c['id'],'candidate_sha256':digest(c),'backend':'host','arch':'host',
                'environment_sha256':digest(p['environment']),'metric':'window_us','samples_us':[y*.99,y,y*1.01],
                'correctness_status':'RUNTIME_CHECKS_PASS','evidence':{'origin':'radreplay-verified-import','report_sha256':'0'*64,
                 'fixture_sha256':'1'*64,'worker_sha256':'2'*64,'plugin_sha256':'a'*64}})
    return p,cat,{'schema':'radiance-planner-measurements/1','profile_sha256':digest(p),'records':rows}
