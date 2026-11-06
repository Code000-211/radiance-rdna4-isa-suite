"""Finite assumption stress tests, not confidence intervals or probabilities.
The profile set is supplied explicitly. Stability across this finite set is not
proof of accuracy, completeness, measured optimality or a production guarantee.
"""
from __future__ import annotations
from .common import fields,need,digest
from .flow import system_check
from .operators import op_plan


def analyze(cat,workloads,profiles):
    fields(profiles,{'schema','systems'}, {'notes'},'profile scenarios')
    need(profiles['schema']=='radiance-system-scenarios/1','unsupported scenario set')
    systems=profiles['systems'];need(isinstance(systems,list) and 2<=len(systems)<=64,'2..64 explicit profiles required')
    for s in systems:system_check(s)
    need(len({s['id'] for s in systems})==len(systems),'duplicate scenario IDs')
    def topology(s):
        return {'devices':[(d['id'],d['arch'],d['memory_bytes'],d['kernel_slots'],sorted(d['matrix_ops_per_us'])) for d in s['devices']],
                'links':[(l['id'],l['source'],l['target'],l['access'],l['shared_resources']) for l in s['links']],
                'shared_resources':sorted(s['shared_resources'])}
    need(all(topology(s)==topology(systems[0]) for s in systems),'different topology/layout-capacity is not a rate sensitivity test')
    reports=[op_plan(cat,workloads,s) for s in systems];rows=[]
    for idx,w in enumerate(workloads['workloads']):
        panels=[r['workloads'][idx] for r in reports]
        ids=set(x['candidate_id'] for x in panels[0]['ranked'])
        need(ids and all({x['candidate_id'] for x in p['ranked']}==ids for p in panels),'incomplete candidate panel across scenarios')
        times=[{x['candidate_id']:x['scenario_us'] for x in p['ranked']} for p in panels]
        need(all(t>=0 for ts in times for t in ts.values()),'negative modeled time')
        winners=[min(ts,key=lambda c:(ts[c],c)) for ts in times]
        candidates=[]
        for cid in sorted(ids):
            ts=[p[cid] for p in times];regrets=[]
            for panel in times:
                best=min(panel.values())
                # Zero-cost no-op alternatives must all be zero; no invented ratio.
                need(best>0 or all(x==0 for x in panel.values()),'mixed zero/nonzero timing ratios unsupported')
                regrets.append(0. if best==0 else panel[cid]/best-1)
            candidates.append({'candidate_id':cid,'min_scenario_us':min(ts),'max_scenario_us':max(ts),
                               'max_relative_regret_within_profile_set':max(regrets),
                               'winner_count':winners.count(cid),'scenario_times_us':ts})
        candidates.sort(key=lambda r:(r['max_relative_regret_within_profile_set'],r['max_scenario_us'],r['candidate_id']))
        rows.append({'workload_id':w['id'],'minimax_profile_set_candidate':candidates[0]['candidate_id'],
                     'same_winner_in_all_scenarios':len(set(winners))==1,'scenario_winners':winners,'candidates':candidates,
                     'status':'NEEDS_MEASUREMENT','measurement_priority':'ranking_changes_with_assumptions' if len(set(winners))>1 else 'validate_profile_and_cost_model'})
    return {'schema':'radiance-planner-sensitivity/1','status':'ASSUMPTION_STRESS_TEST_ONLY',
            'profile_ids':[s['id'] for s in systems],'profile_sha256':[digest(s) for s in systems],
            'catalog_sha256':digest(cat),'workloads_sha256':digest(workloads),'workloads':rows,
            'probability_interpretation':False,'confidence_interval':False,'gpu_executed':False,'deployment_approved':False,
            'scope':'Finite supplied rate/latency scenarios on one topology, not every feasible machine state.'}
