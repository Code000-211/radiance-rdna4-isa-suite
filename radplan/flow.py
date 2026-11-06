"""Dependency/resource simulator. It executes NO native code or GPU work.

A deterministic, non-preemptive admission policy and bottleneck-limited equal
sharing are explicit assumptions, not a model of the AMD scheduler. Quantities
are service units (bytes, operations); capacities are service units/microsecond.
"""
from __future__ import annotations
from collections import Counter
import math
from .common import fields, need, number, integer, digest

MAX_PHASES = 20000

def text(x, name):
    need(isinstance(x, str) and 0 < len(x) <= 512, name + ': nonempty string required')
    return x

AM_KEYS={'a_us','scan_floor_us','us_per_token','scan_batch_exponent','us_per_query_ctx'}


def MODEL_KEYS(where: str, m) -> None:
    """One analytic form, five measured constants, no per-geometry entry."""
    need(isinstance(m,dict) and set(m)==AM_KEYS,
         f'{where}: attention_model needs exactly {sorted(AM_KEYS)}')
    for k,v in m.items(): number(v,k)


def system_check(s: dict) -> dict:
    fields(s, {'schema','id','environment','devices','links','shared_resources'}, {'notes','operator_rates'}, 'flow system')
    need(s['schema']=='radiance-flow-system/1', 'unsupported system schema')
    text(s['id'], 'system id')
    need(isinstance(s['environment'],dict) and bool(s['environment']), 'explicit environment required')
    need(isinstance(s['devices'],list) and 0 < len(s['devices']) <= 16, '1..16 devices required')
    seen=set()
    for d in s['devices']:
        fields(d, {'id','arch','dram_bytes_per_us','matrix_ops_per_us','vector_ops_per_us',
                   'exp_ops_per_us','launch_us','memory_bytes','kernel_slots','provenance'},
               {'attention_model'},'device')
        if 'attention_model' in d:
            MODEL_KEYS(d['id'],d['attention_model'])
        text(d['id'],'device id'); need(d['id'] not in seen,'duplicate device'); seen.add(d['id'])
        need(d['arch'] in ('gfx1201','host'),'unsupported target')
        for k in ('dram_bytes_per_us','vector_ops_per_us','exp_ops_per_us'):
            number(d[k],k,positive=True)
        number(d['launch_us'],'launch_us')
        integer(d['memory_bytes'],'memory_bytes',1,2**60); integer(d['kernel_slots'],'kernel_slots',1,64)
        need(isinstance(d['matrix_ops_per_us'],dict) and bool(d['matrix_ops_per_us']),'matrix rates required')
        for k,v in d['matrix_ops_per_us'].items(): text(k,'matrix path'); number(v,k,positive=True)
        need(d['provenance'] in ('assumed','external-measured-profile'),'profile provenance required')
    # A measured profile may override service rates for one operator family: kernel
    # families reach very different fractions of peak, so one device rate is not enough.
    if 'operator_rates' in s:
        need(isinstance(s['operator_rates'],dict) and 0<len(s['operator_rates'])<=64,'operator_rates must be a non-empty object')
        for fam,ov in s['operator_rates'].items():
            text(fam,'operator family')
            need(isinstance(ov,dict) and ov,'operator rate override must be a non-empty object')
            allowed={'dram_bytes_per_us','vector_ops_per_us','exp_ops_per_us','launch_us',
                     'matrix_ops_per_us','attention_model'}
            need(set(ov)<=allowed, f'{fam}: unknown operator rate override {sorted(set(ov)-allowed)}')
            if 'attention_model' in ov:
                MODEL_KEYS(fam,ov['attention_model'])
            for k in ('dram_bytes_per_us','vector_ops_per_us','exp_ops_per_us'):
                if k in ov: number(ov[k],k,positive=True)
            if 'launch_us' in ov: number(ov['launch_us'],'launch_us')
            if 'matrix_ops_per_us' in ov:
                need(isinstance(ov['matrix_ops_per_us'],dict) and ov['matrix_ops_per_us'],f'{fam}: matrix override must be a non-empty object')
                for k,v in ov['matrix_ops_per_us'].items(): text(k,'matrix path'); number(v,k,positive=True)
    need(isinstance(s['shared_resources'],dict),'shared_resources required')
    for k,v in s['shared_resources'].items(): text(k,'shared resource'); number(v,k,positive=True)
    need(isinstance(s['links'],list) and len(s['links'])<=256,'invalid links')
    ids=set()
    for l in s['links']:
        fields(l,{'id','source','target','access','bytes_per_us','latency_us','shared_resources','provenance'},set(),'link')
        text(l['id'],'link id');need(l['id'] not in ids,'duplicate link'); ids.add(l['id'])
        need(l['source'] in seen and l['target'] in seen and l['source']!=l['target'],'invalid link endpoints')
        need(l['access'] in ('push','pull'),'link access must be explicit push/pull')
        number(l['bytes_per_us'],'link bandwidth',positive=True); number(l['latency_us'],'link latency')
        need(isinstance(l['shared_resources'],list) and len(l['shared_resources'])==len(set(l['shared_resources'])),'duplicate/shared link resources')
        need(set(l['shared_resources'])<=set(s['shared_resources']),'unknown shared resource')
        need(l['provenance'] in ('assumed','external-measured-profile'),'link provenance required')
    return s

def resources(s: dict) -> tuple[dict,dict]:
    system_check(s)
    rates={'shared:'+k:float(v) for k,v in s['shared_resources'].items()}
    slots={}
    for d in s['devices']:
        p=d['id']+':'
        rates.update({p+'dram':d['dram_bytes_per_us'],p+'vector':d['vector_ops_per_us'],p+'exp':d['exp_ops_per_us'],p+'matrix_issue':1.0})
        for path,rate in d['matrix_ops_per_us'].items(): rates[p+'matrix:'+path]=rate
        slots[p+'kernels']=d['kernel_slots']
    for l in s['links']: rates['link:'+l['id']]=l['bytes_per_us']
    return rates,slots

def phase(pid:str, deps:list[str], demands:dict, *, fixed_us:float=0, reservations:dict|None=None, label:str='') -> dict:
    return {'id':pid,'deps':deps,'demands':{k:v for k,v in demands.items() if v>0},
            'fixed_us':fixed_us,'reservations':reservations or {},'label':label}

def validate_phases(ps: list, rates:dict, slots:dict) -> list[str]:
    need(isinstance(ps,list) and 0 < len(ps) <= MAX_PHASES,'empty/oversized phase graph')
    for k,v in rates.items():text(k,'rate');number(v,k,positive=True)
    for k,v in slots.items():text(k,'slot');integer(v,k,1,2**60)
    ids=[]
    for t in ps:
        fields(t, {'id','deps','demands','fixed_us','reservations','label'},set(),'phase')
        text(t['id'],'phase id'); ids.append(t['id'])
        need(isinstance(t['deps'],list) and all(isinstance(v,str) for v in t['deps']),'dependency list required')
        need(len(t['deps'])==len(set(t['deps'])),'duplicate dependency')
        need(isinstance(t['demands'],dict) and isinstance(t['reservations'],dict),'resource maps required')
        number(t['fixed_us'],'fixed latency')
        for k,v in t['demands'].items(): need(k in rates,'unknown service resource: '+k);number(v,k)
        for k,v in t['reservations'].items():need(k in slots,'unknown slot resource: '+k);integer(v,k,0,slots[k])
    need(len(ids)==len(set(ids)),'duplicate phase ids')
    known=set(ids)
    for t in ps:need(set(t['deps'])<=known and t['id'] not in t['deps'],'missing/self dependency')
    # Kahn ordering preserves manifest priority, without recursive stack limits.
    pending={t['id']:set(t['deps']) for t in ps}; ordered=[]
    while pending:
        ready=[i for i in ids if i in pending and not pending[i]]
        need(bool(ready),'dependency cycle')
        for i in ready: del pending[i];ordered.append(i)
        for ds in pending.values(): ds.difference_update(ready)
    return ordered

def simulate(ps:list, rates:dict, slots:dict, *, mode:str='shared', retain_segments:bool=False) -> dict:
    """Simulate normalized-progress work. Each active phase gets an equal share
    of each service resource it actually uses; its slowest demand determines
    progress. This can leave bandwidth unused and does NOT prove an upper bound.
    Slot reservations enforce declared kernel admission limits.
    """
    need(mode in ('shared','serial'),'unknown scheduler mode')
    order=validate_phases(ps,rates,slots); by={t['id']:t for t in ps}
    pending=set(order);done=set();active={};starts={};ends={};t=0.;segments=[];iterations=0
    critical={}
    for i in order:
        p=by[i];iso=max([p['fixed_us']]+[v/rates[k] for k,v in p['demands'].items()])
        critical[i]=max((critical[x] for x in p['deps']), default=0)+iso
    total={k:sum(p['demands'].get(k,0) for p in ps) for k in rates}
    # Resource-volume + dependency-chain floor within THIS declared fluid model.
    lower=max([max(critical.values())]+[v/rates[k] for k,v in total.items()])
    while pending or active:
        iterations+=1;need(iterations<=4*len(ps)+10,'scheduler made no progress')
        held=Counter()
        for i in active:held.update(by[i]['reservations'])
        for i in order:
            if i not in pending:continue
            p=by[i]
            if not set(p['deps'])<=done:continue
            if mode=='serial' and active:break
            if any(held[k]+v>slots[k] for k,v in p['reservations'].items()):continue
            pending.remove(i);starts[i]=t
            if not any(p['demands'].values()) and p['fixed_us']==0:
                ends[i]=t;done.add(i);continue
            active[i]=1.;held.update(p['reservations'])
        if not active:
            need(not pending or any(set(by[i]['deps'])<=done for i in pending),'deadlock in declared graph')
            continue
        users=Counter(k for i in active for k,v in by[i]['demands'].items() if v>0)
        speed={}
        for i in active:
            p=by[i];lim=[rates[k]/(users[k]*v) for k,v in p['demands'].items() if v>0]
            if p['fixed_us']>0:lim.append(1/p['fixed_us'])
            speed[i]=min(lim);need(math.isfinite(speed[i]) and speed[i]>0,'invalid progress rate')
        dt=min(active[i]/speed[i] for i in active)
        need(math.isfinite(dt) and dt>0,'invalid scheduler step')
        if retain_segments:
            need(len(segments)<MAX_PHASES*2,'too many segments')
            segments.append({'start_us':t,'end_us':t+dt,'active':sorted(active),
                             'rates':{k:sum(by[i]['demands'].get(k,0)*speed[i] for i in active) for k in users}})
        t+=dt
        finished=[]
        for i in active:
            active[i]-=dt*speed[i]
            if active[i]<=1e-10:finished.append(i)
        need(bool(finished),'no phase completed')
        for i in finished:del active[i];done.add(i);ends[i]=t
    timeline=[{'id':i,'start_us':starts[i],'end_us':ends[i],'label':by[i]['label']} for i in order]
    return {'makespan_us':t,'model_floor_us':lower,'timeline':timeline,'segments':segments,
            'resource_service':total,'model':'bottleneck-limited equal-share fluid, fixed manifest admission',
            'hardware_scheduler_simulated':False,'production_latency_proved':False}

def memory_peak(buffers:list, node_times:dict, devices:dict, *, resident:dict|None=None)->dict:
    """Explicit allocation lifetimes. Aliases must name ONE allocation, not copies.
    Producer starts allocate, final consumer completion frees. Unknown/zero-lifetime
    buffers are rejected. Over-budget schedules are rejected, not silently spilled.
    """
    resident=resident or {}
    need(set(resident)<=set(devices),'unknown resident device')
    events={d:[] for d in devices};seen=set()
    for d,v in resident.items():integer(v,'resident bytes',0,2**60)
    for b in buffers:
        fields(b,{'id','device','bytes','producer','consumers'},set(),'buffer')
        text(b['id'],'allocation id');need(b['id'] not in seen,'duplicate allocation (encode aliases once)');seen.add(b['id'])
        need(b['device'] in devices,'unknown allocation device');integer(b['bytes'],'allocation bytes',1,2**60)
        need(b['producer'] is None or b['producer'] in node_times,'missing producer')
        need(isinstance(b['consumers'],list) and bool(b['consumers']) and len(set(b['consumers']))==len(b['consumers']),'nonempty unique consumers required')
        need(all(c in node_times for c in b['consumers']),'missing consumer')
        start=0. if b['producer'] is None else node_times[b['producer']][0]
        end=max(node_times[c][1] for c in b['consumers'])
        if b['producer'] is not None:
            ready=node_times[b['producer']][1]
            need(all(c==b['producer'] or node_times[c][0]>=ready-1e-8 for c in b['consumers']), 'consumer precedes producer completion')
        need(end>=start,'invalid lifetime')
        events[b['device']].extend([(start,1,b['bytes']),(end,0,-b['bytes'])])
    peaks={}
    for d,ev in events.items():
        cur=resident.get(d,0);peak=cur
        # At one timestamp charge acquisitions before releases: conservative alias-free peak.
        for _,_,delta in sorted(ev,key=lambda v:(v[0],-v[1])):cur+=delta;peak=max(peak,cur)
        peaks[d]={'peak_bytes':peak,'budget_bytes':devices[d]['memory_bytes'],'fits':peak<=devices[d]['memory_bytes']}
    return peaks
