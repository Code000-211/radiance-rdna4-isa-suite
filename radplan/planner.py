"""Rank candidates, request measurements when uncertain, never certify correctness."""
from __future__ import annotations
from .common import PlanError, need, digest, number, fields
from .model import catalog_check,workloads_check,profile,features,shape_key
from .calibrate import calibrated_prediction


def plan(cat:dict, workloads:dict, p:dict, calibration:dict|None=None)->dict:
    catalog_check(cat);workloads_check(workloads);profile(p)
    results=[]
    for w in workloads['workloads']:
        ranked=[];rejected=[]
        for c in cat['candidates']:
            try:f=features(w,c,p)
            except PlanError as e:
                rejected.append({'candidate_id':c['id'],'reason':str(e)});continue
            pred=calibrated_prediction(calibration,w,c,p,f) if calibration else {'status':'UNCALIBRATED','prediction_us':None,'interval_us':None}
            ranked.append({'candidate_id':c['id'],'candidate_sha256':digest(c),'tile':c['tile'],'launch':c['launch'],
                           'identity':c['identity'],'cost':f,'prediction':pred,
                           'input_layout':c.get('input_layout',c['contract']),
                           'output_layout':c.get('output_layout',c['contract'])})
        # Comparable metric only: never compare microsecond scenario values with
        # measured-fit values when some candidates have no valid fit.
        comparable=bool(ranked) and all(r['prediction']['interval_us'] is not None for r in ranked)
        for r in ranked:
            r['ranking_score_us']=r['prediction']['interval_us'][1] if comparable else r['cost']['scenario_us']
        ranked.sort(key=lambda r:(r['ranking_score_us'],r['candidate_id']))
        separated=comparable and len(ranked)>1 and ranked[0]['prediction']['interval_us'][1]<min(r['prediction']['interval_us'][0] for r in ranked[1:])
        status='NO_SUPPORTED_CANDIDATE' if not ranked else 'CALIBRATED_SEPARATION' if separated else 'NEEDS_MEASUREMENT'
        reason='ranking intervals overlap or only one measured candidate' if comparable else 'missing calibration, changed candidate or out-of-domain prediction'
        # Prioritize those that can still beat the selected candidate; the rest
        # need not be remeasured for THIS performance decision.
        if comparable:
            threshold=ranked[0]['prediction']['interval_us'][1]
            queue=[r['candidate_id'] for r in ranked if r['prediction']['interval_us'][0]<=threshold] if not separated else []
        else:queue=[r['candidate_id'] for r in ranked]
        results.append({'workload_id':w['id'],'shape_key':shape_key(w),'shape':{a:w[a] for a in ('M','N','K')},
            'status':status,'ranking_basis':'calibrated upper envelope' if comparable else 'uncalibrated analytic scenario',
            'suggested_candidate':ranked[0]['candidate_id'] if ranked else None,'decision_reason':reason if not separated else 'separated within stated calibration scope',
            'ranked':ranked,'rejected':rejected,'measurement_queue':queue,
            'correctness_validation_still_required':True,'automatic_dispatch_changed':False})
    return {'schema':'radiance-offline-plan/1','profile_sha256':digest(p),'catalog_sha256':digest(cat),'workloads_sha256':digest(workloads),
        'calibration_sha256':digest(calibration) if calibration else None,'backend':p['backend'],'arch':p['arch'],
        'status':'OFFLINE_PLAN_CREATED','offline':True,'gpu_executed':False,'cpu_timings_are_gpu_timings':False,
        'full_model_speed_predicted':False,'deployment_approved':False,'workloads':results}


def markdown(r:dict)->str:
    lines=['# Offline kernel plan','',f"Target: `{r['arch']}`. No GPU executed. No correctness certification.",'',
           '| Workload | Suggested candidate | Basis | Decision |','|---|---|---|---|']
    for w in r['workloads']:
        lines.append(f"| {w['workload_id']} | {w['suggested_candidate']} | {w['ranking_basis']} | {w['status']} |")
    for w in r['workloads']:
        lines+=['',f"## {w['workload_id']}",'','| Candidate | Tile | Score (us) | Padding efficiency | Groups |','|---|---|---:|---:|---:|']
        for x in w['ranked']:
            f=x['cost'];lines.append(f"| {x['candidate_id']} | {x['tile']} | {x['ranking_score_us']:.4f} | {f['tile_efficiency']:.4f} | {f['groups']} |")
        lines+=['','Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.',
                '',f"Measurements requested: {', '.join(w['measurement_queue']) or 'none for ranking; numerical validation still required'}"]
        if w['rejected']:
            lines+=['','Rejected:']+[f"- `{x['candidate_id']}`: {x['reason']}" for x in w['rejected']]
    return '\n'.join(lines)+'\n'


def serial_path(report:dict,spec:dict)->dict:
    """Dynamic programming over explicitly serial stage alternatives/layout costs.
    No concurrency, TP2, hidden fusion or automatic inference of tensor lifetimes.
    """
    fields(spec,{'schema','execution','stages','transitions'},set(),'path')
    need(spec['schema']=='radiance-planner-path/1' and spec['execution']=='serial_no_overlap','only declared serial, no-overlap paths supported')
    need(report.get('schema')=='radiance-offline-plan/1','unsupported plan report')
    need(isinstance(spec['stages'],list) and 0<len(spec['stages'])<=1024,'empty/oversized stages')
    table={w['workload_id']:w for w in report['workloads']}
    transitions={}
    for t in spec['transitions']:
        fields(t,{'from','to','cost_us','provenance'},set(),'transition')
        number(t['cost_us'],'transition cost');need(t['provenance'] in ('measured','assumed'),'transition provenance required')
        key=(t['from'],t['to']);need(key not in transitions,'duplicate layout transition');transitions[key]=t
    # Do not mix calibrated and scenario units across stages.
    selected=[]
    for stage in spec['stages']:
        fields(stage,{'workload_id','allowed_candidates'},set(),'stage')
        need(stage['workload_id'] in table,'unknown stage workload')
        rows=table[stage['workload_id']]['ranked'];known={x['candidate_id'] for x in rows}
        need(isinstance(stage['allowed_candidates'],list) and bool(stage['allowed_candidates']),'explicit nonempty allowed candidates required')
        need(set(stage['allowed_candidates'])<=known,'stage names rejected/unknown candidates')
        selected.append([x for x in rows if x['candidate_id'] in stage['allowed_candidates']])
    calibrated=all(x['prediction']['interval_us'] is not None for rows in selected for x in rows)
    def cost(x):return x['prediction']['interval_us'][1] if calibrated else x['cost']['scenario_us']
    dp={}
    for i,rows in enumerate(selected):
        nxt={}
        for x in rows:
            possibilities=[]
            if i==0:possibilities=[(cost(x),[],[])]
            else:
                for old in dp.values():
                    key=(old['layout'],x['input_layout'])
                    t=transitions.get(key)
                    if key[0]==key[1]:t={'cost_us':0,'provenance':'assumed','from':key[0],'to':key[1]}
                    if t is None:continue
                    possibilities.append((old['cost_us']+t['cost_us']+cost(x),old['candidates'],old['transitions']+[t]))
            if possibilities:
                val,history,ts=min(possibilities,key=lambda v:v[0]);nxt[x['candidate_id']]={'cost_us':val,'candidates':history+[x['candidate_id']],'layout':x['output_layout'],'transitions':ts}
        need(bool(nxt),'no compatible path; missing explicit layout transition cost')
        dp=nxt
    best=min(dp.values(),key=lambda x:x['cost_us'])
    return {'schema':'radiance-planner-path-result/1','status':'SERIAL_PATH_SCENARIO','basis':'calibrated upper envelopes' if calibrated else 'analytic scenarios',
            'best_path':best,'plan_sha256':digest(report),'spec_sha256':digest(spec),'overlap_modeled':False,'production_latency_proved':False}
