"""Compare external GPU observations, never CPU-to-GPU time conversion.

This is an evidence comparison protocol, not automatic instrumentation. Records
are trusted local reports with hash identities, not authenticated attestations.
No result auto-calibrates unobserved overlap or silently approves deployment.
"""
from __future__ import annotations
import statistics
from .common import fields,need,number,hash_string,digest
from .flow import system_check


def compare_graph(report:dict,observations:dict,system:dict):
    system_check(system)
    need(report.get('schema')=='radiance-graph-plan/1','graph plan required')
    need(report['system_sha256']==digest(system),'system profile changed')
    fields(observations,{'schema','graph_plan_sha256','environment_sha256','backend','devices','samples_us',
                         'correctness','instrumented','evidence_sha256'},set(),'graph observations')
    need(observations['schema']=='radiance-graph-observations/1','observation schema')
    need(observations['graph_plan_sha256']==digest(report),'different graph/candidate plan')
    need(observations['environment_sha256']==digest(system['environment']),'different environment/compiler/power profile')
    need(observations['backend']=='aql','CPU timing cannot validate a GPU graph')
    expected=[{'id':d['id'],'arch':d['arch']} for d in system['devices']]
    need(observations['devices']==expected and all(d['arch']=='gfx1201' for d in expected),'different device set')
    need(observations['correctness']=='PASS' and observations['instrumented'] is False,'failed/instrumented graph rejected')
    hash_string(observations['evidence_sha256'],'observation evidence')
    samples=observations['samples_us'];need(isinstance(samples,list) and 3<=len(samples)<=100000,'3..100000 samples required')
    for v in samples:number(v,'sample us',positive=True)
    observed=statistics.median(samples);pred=number(report['selected']['makespan_us'],'prediction',positive=True)
    return {'schema':'radiance-graph-comparison/1','status':'COMPARISON_ONLY','predicted_us':pred,'observed_median_us':observed,
            'relative_error':pred/observed-1,'absolute_relative_error':abs(pred/observed-1),'sample_count':len(samples),
            'observations_sha256':digest(observations),'plan_sha256':digest(report),
            'sample_min_us':min(samples),'sample_max_us':max(samples),'deployment_approved':False,
            'unmeasured_candidates_validated':False,'profile_automatically_modified':False}


def measurement_batch(report:dict):
    need(report.get('schema')=='radiance-operator-plan/1','operator plan required')
    requests=[]
    for w in report['workloads']:
        for row in w['ranked']:
            requests.append({'workload_id':w['workload_id'],'candidate_id':row['candidate_id'],
                             'workload_sha256':row['workload_sha256'],'candidate_sha256':row['candidate_sha256'],
                             'system_sha256':report['system_sha256'],'state':'NOT_EXECUTED',
                             'required_evidence':['compiled mapping + ISA check','independent correctness/reference',
                                                  'uninstrumented samples','pinned environment','actual dispatch trace'],
                             'can_launch_automatically':False})
    return {'schema':'radiance-operator-measurement-batch/1','requests':requests,'gpu_executed':False,
            'note':'Requested measurements, not measured results. New operator-to-launcher mappings are not guessed.'}
