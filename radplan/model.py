"""Inspectable dense GEMM cost features and a deliberately uncalibrated roofline.

Counts are conditional on the declared tile/traffic model, not inferred dynamically
from arbitrary HIP code. CPU timings never calibrate an AMDGPU profile.
"""
from __future__ import annotations
import math
from .common import PlanError, need, fields, integer, number, digest, ceildiv, hash_string

# Public feature order is versioned. Coefficients fitted to these features are not
# estimates of physical bandwidth/throughput: they are nonnegative surrogate terms.
FEATURES = ('intercept', 'executed_gops', 'operand_mb', 'tile_read_mb',
            'block_k_million', 'output_melements')
MODEL_VERSION='gemm-cost/1'


def workload(w: dict) -> dict:
    fields(w, {'id','op','M','N','K','a_bits','b_bits','out_bits','contract','cache_state'},
           {'weight_group','weight_scale_bytes','activation_scale_bytes','note','frequency'}, 'workload')
    need(w['op']=='gemm_nt', 'unsupported operation; do not substitute GEMM for attention/MoE/collectives')
    need(isinstance(w['id'],str) and w['id'], 'workload id required')
    for k in ('M','N','K'): integer(w[k], k, 1, 2**24)
    need(w['M']*w['N']*w['K'] <= 2**60, 'workload product too large')
    for k in ('a_bits','b_bits','out_bits'): need(w[k] in (4,8,16,32,64) and type(w[k]) is int, 'invalid storage bits')
    need(isinstance(w['contract'],str) and bool(w['contract']), 'explicit numerical/layout contract required')
    need(w['cache_state'] in ('cold','warm','unknown'), 'cache_state must be cold/warm/unknown')
    integer(w.get('weight_group',1),'weight_group')
    number(w.get('weight_scale_bytes',0),'weight_scale_bytes')
    number(w.get('activation_scale_bytes',0),'activation_scale_bytes')
    if 'frequency' in w: integer(w['frequency'],'frequency')
    return w


def shape_key(w: dict) -> str:
    workload(w)
    # All candidates for one shape/state stay in a single fit/calibration/test group.
    return digest({k:v for k,v in w.items() if k not in ('id','note','frequency')})


def candidate(c: dict) -> dict:
    required={'id','family','backend','arch','contract','compute_path','tile','wave_size',
              'waves_per_group','stages','lds_bytes','tail','multiples','work_policy',
              'identity','launch','unpack_ops_per_weight','matrix_alu_overlap','format'}
    optional={'resources','known_issues','notes','split_k','lds_bytes_per_kblock','unmodeled','input_layout','output_layout'}
    fields(c,required,optional,'candidate')
    for k in ('id','family','contract','compute_path'): need(isinstance(c[k],str) and c[k], f'candidate {k} required')
    need(c['backend'] in ('host','aql') and c['arch']==('host' if c['backend']=='host' else 'gfx1201'), 'backend/arch mismatch')
    need(isinstance(c['tile'],list) and len(c['tile'])==3,'tile requires [BM,BN,BK]')
    for v in c['tile']: integer(v,'tile',1,4096)
    need(c['wave_size'] in (32,64) and type(c['wave_size']) is int,'wave size must be 32/64')
    integer(c['waves_per_group'],'waves_per_group',1,32)
    integer(c['stages'],'stages',1,16); integer(c['lds_bytes'],'lds_bytes',0,2**24)
    fields(c['tail'], {'M','N','K'},set(),'tail')
    fields(c['multiples'], {'M','N','K'},set(),'multiples')
    for k in ('M','N','K'):
        need(type(c['tail'][k]) is bool,'tail flags must be boolean'); integer(c['multiples'][k],k+' multiple')
    need(c['work_policy'] in ('full_tiles','exact'),'work_policy must be full_tiles or exact')
    need(type(c['matrix_alu_overlap']) is bool,'matrix_alu_overlap must be boolean')
    number(c['unpack_ops_per_weight'],'unpack ops')
    integer(c.get('split_k',1),'split_k',1,64)
    number(c.get('lds_bytes_per_kblock',0),'lds traffic')
    fields(c['identity'], {'kind','sha256'}, {'object_sha256','plugin_sha256','symbol','provenance'}, 'identity')
    need(c['identity']['kind'] in ('source','plugin','code_object','hypothesis'),'identity kind unsupported')
    hash_string(c['identity']['sha256'],'identity hash')
    for key in ('object_sha256','plugin_sha256'):
        if key in c['identity']:hash_string(c['identity'][key],key)
    fields(c['launch'], {'row','op','params'},set(),'launch')
    need(c['launch']['op'] in ('gemm_nt','gemm_nt_q'),'GEMM adapter only')
    fields(c['format'],{'a_bits','b_bits','out_bits','weight_group','weight_scale_bytes','activation_scale_bytes'},set(),'format')
    for key in ('a_bits','b_bits','out_bits'):
        need(type(c['format'][key]) is int and c['format'][key] in (4,8,16,32,64), 'invalid format bits')
    integer(c['format']['weight_group'],'format group')
    for key in ('weight_scale_bytes','activation_scale_bytes'):number(c['format'][key],key)
    need(isinstance(c['launch']['row'],str) and isinstance(c['launch']['params'],dict),'invalid launcher')
    if 'resources' in c:
        r=c['resources'];fields(r,{'vgprs','descriptor_vgprs','scratch_bytes','origin'}, {'object_sha256','symbol','report_sha256'},'resources')
        for k in ('vgprs','descriptor_vgprs'):integer(r[k],k,0,256)
        integer(r['scratch_bytes'],'scratch',0,2**30)
        need(r['origin'] in ('compiler_metadata','estimate'), 'resource provenance missing')
    return c


def profile(p: dict) -> dict:
    fields(p,{'schema','id','backend','arch','environment','spec','assumptions'}, {'sources','notes'},'profile')
    need(p['schema']=='radiance-planner-profile/1','unsupported profile')
    need(p['backend'] in ('host','aql') and p['arch']==('host' if p['backend']=='host' else 'gfx1201'),'profile backend/arch mismatch')
    need(isinstance(p['environment'],dict) and p['environment'], 'environment fingerprint required')
    s=p['spec'];fields(s,{'peak_gops','bandwidth_gbs','compute_units','simds_per_cu','max_threads','max_lds_per_group'},set(),'spec')
    need(isinstance(s['peak_gops'],dict),'peak_gops required')
    for path,value in s['peak_gops'].items():number(value,path,positive=True)
    number(s['bandwidth_gbs'],'bandwidth',positive=True)
    for k in ('compute_units','simds_per_cu','max_threads','max_lds_per_group'):integer(s[k],k)
    a=p['assumptions'];fields(a,{'launch_us','compute_efficiency','memory_efficiency','lds_gbs','alu_gops','reuse_fraction','latency_hiding_waves'},set(),'assumptions')
    for k in a: number(a[k],k,positive=k!='launch_us' and k!='reuse_fraction')
    for k in ('compute_efficiency','memory_efficiency'): need(a[k]<=1,'efficiency exceeds one')
    need(a['reuse_fraction']<=1,'reuse exceeds one')
    return p


def features(w: dict, c: dict, p: dict) -> dict:
    workload(w);candidate(c);profile(p)
    need(c['backend']==p['backend'] and c['arch']==p['arch'],'candidate/profile target mismatch')
    need(w['contract']==c['contract'],'incompatible numerical/layout contract')
    for key,val in c['format'].items():
        default=1 if key=='weight_group' else 0
        need(w.get(key,default)==val, 'format mismatch: '+key)
    m,n,k=w['M'],w['N'],w['K'];bm,bn,bk=c['tile'];split=c.get('split_k',1)
    for axis,t in zip(('M','N','K'),c['tile']):
        need(w[axis]%c['multiples'][axis]==0, axis+' violates launcher multiple')
        need(c['tail'][axis] or w[axis]%t==0, axis+' tail not supported')
    need(k%split==0,'K does not divide split_k')
    need((k//split)%w.get('weight_group',1)==0, 'split-K crosses quantization group; adapter required')
    for issue in c.get('known_issues',[]):
        if issue=='scalar_scale_tail' and m%bm: raise PlanError('source-known scalar Ascale tail read: candidate excluded for this M')
        need(issue=='scalar_scale_tail','unhandled known issue: '+str(issue))
    s=p['spec'];g_m=ceildiv(m,bm);g_n=ceildiv(n,bn);g_k=ceildiv(k//split,bk)
    groups=g_m*g_n*split
    if c['backend']=='aql':
        need(c['wave_size']*c['waves_per_group']<=s['max_threads'],'workgroup exceeds thread limit')
        need(c['lds_bytes']<=s['max_lds_per_group'],'LDS exceeds workgroup limit')
    mp,np,kp=g_m*bm,g_n*bn,g_k*bk*split
    operations=2*m*n*k
    executed=2*mp*np*kp if c['work_policy']=='full_tiles' else operations
    a_bytes=ceildiv(m*k*w['a_bits'],8);b_bytes=ceildiv(n*k*w['b_bits'],8)
    out_bytes=ceildiv(m*n*w['out_bits'],8)
    bscale=n*ceildiv(k,w.get('weight_group',1))*w.get('weight_scale_bytes',0)
    ascale=m*w.get('activation_scale_bytes',0)
    operand=a_bytes+b_bytes+bscale+ascale+out_bytes
    # Global tile requests are separated from unique bytes. Bounds describe THIS
    # tiling model, not arbitrary spill/prefetch/cache traffic of a compiled kernel.
    if c['work_policy']=='full_tiles':
        tile_a=groups*bm*(g_k*bk)*w['a_bits']/8
        tile_b=groups*bn*(g_k*bk)*w['b_bits']/8
    else:
        tile_a=a_bytes*g_n;tile_b=b_bytes*g_m
    read_tile=tile_a+tile_b+bscale*g_m+ascale*g_n*split
    unique_read=a_bytes+b_bytes+bscale+ascale
    partial_bytes=4*m*n*split if split>1 else 0
    reduction_bytes=partial_bytes+out_bytes if split>1 else 0
    # Main split kernels write partials instead of final C; reduction reads them.
    main_store=partial_bytes if split>1 else out_bytes
    q_request=read_tile+main_store
    q_cold_min=unique_read+main_store
    reuse=p['assumptions']['reuse_fraction']
    q_scenario=q_cold_min+(q_request-q_cold_min)*(1-reuse)
    # Warm-cache VRAM misses are unknowable from shape alone; do not call cold
    # bytes a universal DRAM lower bound.
    warnings=[]
    if w['cache_state']!='cold':warnings.append('warm/unknown cache: VRAM traffic is a cold-style scenario, not a lower bound')
    if c.get('unmodeled'):warnings+=c['unmodeled']
    regs=None;register_slots=None
    if c['backend']=='aql' and 'resources' in c:
        r=c['resources'];gran,cap=(24,1536) if c['wave_size']==32 else (12,768)
        regs=ceildiv(max(1,r['vgprs'],r['descriptor_vgprs']),gran)*gran
        register_slots=min(16,cap//regs)
        if r['scratch_bytes']:warnings.append('nonzero scratch: traffic not inferred from scratch allocation size')
        if r['origin']=='estimate':warnings.append('register count is estimated, not compiled')
    elif c['backend']=='aql':warnings.append('missing compiler register counts; occupancy not known')
    # This is a loose resource upper bound across a WGP; placement/fragmentation
    # is not modeled, so it is NEVER reported as achieved occupancy.
    resident=None
    if c['backend']=='aql' and register_slots is not None:
        by_regs=(register_slots*s['simds_per_cu']*2)//c['waves_per_group']
        by_lds=(128*1024)//(ceildiv(c['lds_bytes'],1024)*1024) if c['lds_bytes'] else 32
        resident=min(32,by_regs,by_lds)
    acc_est=ceildiv(bm*bn,c['wave_size']*c['waves_per_group'])
    grid_util=min(1,groups/max(1,s['compute_units']/2)) if c['backend']=='aql' else 1
    matrix_peak=s['peak_gops'].get(c['compute_path'])
    need(matrix_peak is not None,'profile has no throughput for actual compute instruction path')
    t_compute=executed/(matrix_peak*1000)
    t_mem=q_scenario/(s['bandwidth_gbs']*1000)
    unpack=2*tile_b*8/w['b_bits']*c['unpack_ops_per_weight']/2
    t_unpack=unpack/(p['assumptions']['alu_gops']*1000)
    lds_bytes=groups*g_k*c.get('lds_bytes_per_kblock',0)
    t_lds=lds_bytes/(p['assumptions']['lds_gbs']*1000)
    hiding=1.0
    if register_slots is not None:
        hiding=min(1,register_slots/p['assumptions']['latency_hiding_waves'])
    effective_compute=t_compute/(p['assumptions']['compute_efficiency']*max(grid_util,1e-12))
    effective_mem=t_mem/(p['assumptions']['memory_efficiency']*max(hiding,1e-12))
    issue=max(effective_compute,t_unpack) if c['matrix_alu_overlap'] else effective_compute+t_unpack
    terms={'matrix_and_unpack_us':issue,'memory_us':effective_mem,'lds_us':t_lds}
    reduce_us=(p['assumptions']['launch_us']+reduction_bytes/(s['bandwidth_gbs']*1000*p['assumptions']['memory_efficiency'])) if split>1 else 0
    scenario=p['assumptions']['launch_us']+max(terms.values())+reduce_us
    lower=max(t_compute,q_cold_min/(s['bandwidth_gbs']*1000)) if w['cache_state']=='cold' and c['backend']=='aql' else None
    vector=[1.,executed/1e9,operand/1e6,q_request/1e6,groups*g_k/1e6,m*n/1e6]
    return {'model':MODEL_VERSION,'features':dict(zip(FEATURES,vector)),
        'useful_ops':operations,'executed_ops':executed,'tile_efficiency':operations/executed,
        'groups':groups,'k_blocks_per_group':g_k,'unique_operand_bytes':operand,
        'tile_requested_bytes':q_request,'scenario_memory_bytes':q_scenario,
        'split_reduction_bytes':reduction_bytes,'lds_traffic_bytes':lds_bytes,
        'physical_vgpr_allocation':regs,'register_limited_waves':register_slots,
        'resident_groups_per_wgp_upper_bound':resident,'accumulator_register_estimate':acc_est,
        'cold_resource_floor_us':lower,'scenario_us':scenario,'scenario_terms_us':terms,
        'scenario_bottleneck':max(terms,key=terms.get),'warnings':warnings,
        'estimate_kind':'UNCALIBRATED_SCENARIO','confidence_interval':None,
        'scratch_traffic_modeled':False,'achieved_occupancy_proved':False}


def catalog_check(cat: dict) -> dict:
    fields(cat,{'schema','candidates'}, {'source','notes'},'catalog')
    need(cat['schema']=='radiance-planner-catalog/1','unsupported catalog')
    need(isinstance(cat['candidates'],list) and 0<len(cat['candidates'])<=10000,'empty/oversized candidate catalog')
    for c in cat['candidates']:candidate(c)
    ids=[c['id'] for c in cat['candidates']];need(len(ids)==len(set(ids)),'duplicate candidate IDs')
    return cat


def workloads_check(doc: dict) -> dict:
    fields(doc,{'schema','workloads'}, {'notes'},'workloads')
    need(doc['schema']=='radiance-planner-workloads/1','unsupported workloads')
    need(isinstance(doc['workloads'],list) and 0<len(doc['workloads'])<=10000,'empty/oversized workloads')
    for w in doc['workloads']:workload(w)
    ids=[w['id'] for w in doc['workloads']];need(len(ids)==len(set(ids)),'duplicate workload IDs')
    return doc
