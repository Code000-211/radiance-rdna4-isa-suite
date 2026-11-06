"""Finite candidate search over an explicit multi-device dependency graph.

The result is optimal only within an exhaustively enumerated candidate set AND
the declared fluid/admission model. Beam truncation is always reported. Neither
mode modifies Radiance dispatch or promises an optimal hardware schedule.
"""
from __future__ import annotations
import itertools
import math
from collections import defaultdict
from .common import fields, need, integer, number, digest, PlanError
from .flow import phase, simulate, resources, memory_peak, text
from .operators import op_plan


def validate_graph(spec,report,s):
    fields(spec,{'schema','nodes','transitions','buffers','resident_bytes','search'}, {'scope','notes'},'graph')
    need(spec['schema']=='radiance-planner-graph/1','unsupported graph schema')
    need(isinstance(spec['nodes'],list) and 0<len(spec['nodes'])<=1024,'1..1024 graph nodes')
    table={w['workload_id']:w for w in report['workloads']};known={n['id'] for n in spec['nodes']}
    need(len(known)==len(spec['nodes']),'duplicate graph node')
    node_by={};prev_queue={};deps={};options={}
    for n in spec['nodes']:
        fields(n,{'id','workload_id','allowed_candidates','deps','queue'},set(),'graph node')
        text(n['id'],'node id');text(n['queue'],'queue');need(n['workload_id'] in table,'workload missing')
        need(isinstance(n['deps'],list),'node dependencies required')
        ds={}
        for e in n['deps']:
            fields(e,{'node','kind'},set(),'edge');need(e['node'] in known and e['node']!=n['id'],'bad edge')
            need(e['kind'] in ('data','control'),'edge kind');need(e['node'] not in ds,'duplicate edge');ds[e['node']]=e['kind']
        if n['queue'] in prev_queue:ds.setdefault(prev_queue[n['queue']],'control')
        prev_queue[n['queue']]=n['id'];deps[n['id']]=ds;node_by[n['id']]=n
        rows={r['candidate_id']:r for r in table[n['workload_id']]['ranked']}
        ac=n['allowed_candidates'];need(isinstance(ac,list) and ac and len(ac)==len(set(ac)),'allowed candidate list empty/duplicated')
        need(set(ac)<=set(rows),'unknown/rejected candidate in graph')
        options[n['id']]=[rows[x] for x in ac]
    # Same-layout data is not automatically accessible on another GPU. Data
    # crossing ranks requires an explicit collective node, never an implicit copy.
    for nid, ds in deps.items():
        dst=table[node_by[nid]['workload_id']]['workload']
        for parent,kind in ds.items():
            if kind!='data':continue
            src=table[node_by[parent]['workload_id']]['workload']
            srcdev=set(src['ranks']) if src['op']=='collective' else {src['device']}
            dstdev=set(dst['ranks']) if dst['op']=='collective' else {dst['device']}
            need(bool(srcdev & dstdev),'cross-device data edge needs an explicit transfer/collective node')
    order=[];pending=set(known)
    while pending:
        ready=[n['id'] for n in spec['nodes'] if n['id'] in pending and set(deps[n['id']])<=set(order)]
        need(ready,'graph/queue cycle')
        order+=ready;pending.difference_update(ready)
    trans={};devs={d['id']:d for d in s['devices']}
    for t in spec['transitions']:
        fields(t,{'from','to','device','bytes','fixed_us','provenance'},set(),'layout transition')
        text(t['from'],'from layout');text(t['to'],'to layout');need(t['device'] in devs,'conversion device missing')
        integer(t['bytes'],'conversion bytes',0,2**48);number(t['fixed_us'],'conversion cost')
        need(t['provenance']=='assumed','transitions currently accept explicit assumptions, not unattested measurements')
        key=(t['from'],t['to'],t['device']);need(key not in trans,'duplicate transition');trans[key]=t
    fields(spec['search'],{'max_combinations','beam_width'},set(),'search')
    integer(spec['search']['max_combinations'],'max combinations',1,4096);integer(spec['search']['beam_width'],'beam width',1,128)
    need(isinstance(spec['buffers'],list) and len(spec['buffers'])<=10000,'buffer list')
    need(isinstance(spec['resident_bytes'],dict),'resident bytes map')
    ancestors={}
    for nid in order:
        ancestors[nid]=set(deps[nid])
        for parent in deps[nid]:ancestors[nid].update(ancestors[parent])
    for buf in spec['buffers']:
        fields(buf,{'id','device','bytes','producer','consumers'},set(),'buffer')
        need(buf['producer'] is None or buf['producer'] in known,'buffer producer missing')
        need(isinstance(buf['consumers'],list) and all(c in known for c in buf['consumers']),'buffer consumer missing')
        if buf['producer'] is not None:
            for con in buf['consumers']:
                need(buf['producer']==con or buf['producer'] in ancestors[con], 'buffer read lacks a dependency path')
    return order,deps,options,trans,node_by,table


def expand(chosen,order,deps,trans,node_by,table):
    ps=[];roots={};leaves={};node_phases={}
    for nid in order:
        if nid not in chosen:continue
        row=chosen[nid];old=row['phases'];prefix=nid+'/'
        used={x for p in old for x in p['deps']}
        roots[nid]=[prefix+p['id'] for p in old if not p['deps']]
        leaves[nid]=[prefix+p['id'] for p in old if p['id'] not in used]
        prior=[]
        for other,kind in deps[nid].items():
            need(other in chosen,'partial graph not dependency closed')
            incoming=list(leaves[other])
            if kind=='data' and chosen[other]['output_layout']!=row['input_layout']:
                w=table[node_by[nid]['workload_id']]['workload']
                device=w.get('device')
                # Cross-rank data movement must be an explicit collective/transfer,
                # not disguised as a same-device layout conversion.
                need(device is not None,'collective layout mismatch needs an explicit conversion node')
                t=trans.get((chosen[other]['output_layout'],row['input_layout'],device))
                need(t is not None,'missing layout transition')
                tid=f'{nid}/convert-from/{other}'
                ps.append(phase(tid,incoming,{device+':dram':2*t['bytes']},fixed_us=t['fixed_us'],
                                reservations={device+':kernels':1},label='declared layout conversion'))
                incoming=[tid]
            prior.extend(incoming)
        node_phases[nid]=[]
        for p in old:
            ds=[prefix+x for x in p['deps']]
            if not ds:ds+=prior
            q=phase(prefix+p['id'],sorted(set(ds)),p['demands'],fixed_us=p['fixed_us'],reservations=p['reservations'],label=p['label'])
            ps.append(q);node_phases[nid].append(q['id'])
    return ps,node_phases


def evaluate(chosen,order,deps,trans,node_by,table,s,spec,full=True,segments=False):
    ps,node_phases=expand(chosen,order,deps,trans,node_by,table)
    rates,slots=resources(s);out=simulate(ps,rates,slots,retain_segments=segments)
    times={t['id']:(t['start_us'],t['end_us']) for t in out['timeline']}
    nt={nid:(min(times[x][0] for x in ids),max(times[x][1] for x in ids)) for nid,ids in node_phases.items()}
    if full:
        buffers=list(spec['buffers'])
        for nid,row in chosen.items():
            w=table[node_by[nid]['workload_id']]['workload'];ds=w['ranks'] if w['op']=='collective' else [w['device']]
            if row['workspace_bytes']:
                for d in ds:
                    buffers.append({'id':'$planner_workspace/'+nid+'/'+d,'device':d,'bytes':math.ceil(row['workspace_bytes']/len(ds)),
                                    'producer':nid,'consumers':[nid]})
        mem=memory_peak(buffers,nt,{d['id']:d for d in s['devices']},resident=spec['resident_bytes'])
        need(all(x['fits'] for x in mem.values()),'scheduled allocation highwater exceeds device memory budget')
        out['memory']=mem
    out['node_times']=nt
    out['chosen']={nid:r['candidate_id'] for nid,r in chosen.items()}
    return out


def graph_plan(cat,workloads,s,spec):
    report=op_plan(cat,workloads,s)
    order,deps,options,trans,node_by,table=validate_graph(spec,report,s)
    combos=math.prod(len(options[x]) for x in order);limit=spec['search']['max_combinations'];count=0;rejected=[]
    finalists=[];exhaustive=combos<=limit
    if exhaustive:
        for seq in itertools.product(*(options[x] for x in order)):
            chosen=dict(zip(order,seq));count+=1
            try:r=evaluate(chosen,order,deps,trans,node_by,table,s,spec)
            except PlanError as e:rejected.append({'chosen':{k:v['candidate_id'] for k,v in chosen.items()},'reason':str(e)});continue
            finalists.append((r['makespan_us'],chosen,r))
    else:
        beam=[{}]
        for idx,nid in enumerate(order):
            next_beam=[]
            for partial in beam:
                for row in options[nid]:
                    chosen=dict(partial);chosen[nid]=row;count+=1
                    need(count<=20000,'beam evaluation budget exhausted; reduce graph/catalog')
                    try:r=evaluate(chosen,order,deps,trans,node_by,table,s,spec,full=idx==len(order)-1)
                    except PlanError as e:
                        if len(rejected)<1000:rejected.append({'chosen':{k:v['candidate_id'] for k,v in chosen.items()},'reason':str(e)})
                        continue
                    next_beam.append((r['makespan_us'],chosen,r))
            need(next_beam,'no feasible candidate assignment survived beam')
            next_beam.sort(key=lambda x:(x[0],digest(x[2]['chosen'])))
            finalists=next_beam[:spec['search']['beam_width']];beam=[x[1] for x in finalists]
    need(finalists,'no feasible graph assignment')
    finalists.sort(key=lambda x:(x[0],digest(x[2]['chosen'])))
    chosen=finalists[0][1]
    selected=evaluate(chosen,order,deps,trans,node_by,table,s,spec,segments=True)
    ps,_=expand(chosen,order,deps,trans,node_by,table);rates,slots=resources(s)
    serial=simulate(ps,rates,slots,mode='serial')
    return {'schema':'radiance-graph-plan/1','status':'NEEDS_MEASUREMENT','system_sha256':digest(s),
            'catalog_sha256':digest(cat),'workloads_sha256':digest(workloads),'graph_sha256':digest(spec),
            'candidate_combinations':combos,'evaluations':count,'search_complete':exhaustive,
            'search':'exhaustive fixed-policy scenarios' if exhaustive else 'beam heuristic; discarded alternatives not proved worse',
            'selected':selected,'serialized_scenario_us':serial['makespan_us'],'rejected':rejected,
            'alternatives':[{'scenario_us':r[0],'chosen':r[2]['chosen']} for r in finalists[:32]],
            'scope':spec.get('scope',{'complete_model':False}),
            'gpu_executed':False,'tp2_runtime_executed':False,'production_latency_proved':False,
            'automatic_dispatch_changed':False,'global_hardware_optimum_proved':False,
            'measurement_requests':[{'node':nid,'workload_id':node_by[nid]['workload_id'],'candidate_ids':[x['candidate_id'] for x in options[nid]],
                                     'reason':'new operator/overlap model uncalibrated'} for nid in order],
            'limitations':['Fixed manifest/queue admission, equal resource sharing, no preemption or register-placement simulation.',
                           'Resource floor is conditional on model and profile, not a hardware latency bound.',
                           'Allocation sizes/lifetimes must be complete; opaque allocator or engine buffers are not inferred.',
                           'Concurrent timings must be validated as graph windows, not summed isolated kernels.']}
