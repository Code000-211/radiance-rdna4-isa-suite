"""Import only checked, uninstrumented, single-operation replay measurements."""
from __future__ import annotations
from pathlib import Path
from .common import need, digest, file_hash, read
from .model import workload, candidate, profile, features


def import_replay(report_path:Path,fixture_path:Path,w:dict,c:dict,p:dict)->dict:
    from radreplay.format import validate
    workload(w);candidate(c);profile(p);features(w,c,p)
    info=validate(fixture_path);m=read(fixture_path);r=read(report_path);rt=r.get('runtime',{})
    need(r.get('status')=='RUNTIME_CHECKS_PASS' and r.get('runtime_executed') is True and r.get('exit_code')==0,'replay did not pass and execute')
    need(rt.get('status')=='RUNTIME_CHECKS_PASS','native result did not pass')
    need(rt.get('fixture_sha256')==file_hash(fixture_path)==info['manifest_sha256'],'fixture identity mismatch')
    need(not rt.get('worker_build',{}).get('host_asan') and not r.get('options',{}).get('diagnostic_only'),'sanitized/diagnostic timings forbidden')
    need(rt.get('diagnostics',{}).get('status')=='PASS','required correctness diagnostics absent')
    need(len(rt.get('samples_us',[]))==r['options']['repetitions'],'missing timing samples')
    need(rt.get('checked_iterations')==r['options']['warmups']+r['options']['repetitions'],'unchecked measurement iterations')
    need(rt.get('backend')==m['target']['backend']==c['backend']==p['backend'],'backend mismatch; CPU is not a GPU simulator')
    need(rt.get('device',{}).get('arch')==p['arch'],'device architecture mismatch')
    need(p['environment'].get('measurement_mode')==rt.get('measurement'),'timing mode not pinned in profile')
    need(p['environment'].get('worker_sha256')==r.get('worker_sha256'),'worker/build fingerprint mismatch')
    need(p['environment'].get('device_name')==rt.get('device',{}).get('name'),'actual device name differs')
    if p['backend']=='aql':
        need(r.get('isa',{}).get('status')=='STATIC_CHECKS_PASS','GPU timing requires bound static evidence')
        need(c['resources']['report_sha256']==r['isa'].get('report_sha256'),'compiled resources and runtime ISA evidence differ')
        need(p['environment'].get('device_identity')==rt.get('device'),'GPU physical/runtime identity differs')
        need(p['environment'].get('clock_power') not in (None,'UNMEASURED'),'declare the actual power/clock profile; importer cannot measure it')
    need(len(m['nodes'])==1 and m['nodes'][0]['kind']=='kernel' and m['nodes'][0]['phase']=='measure','single kernel operation only; do not assign window time to one of several nodes')
    node=m['nodes'][0];need(node['row']==c['launch']['row'] and node['op']==c['launch']['op'],'launcher differs from planned candidate')
    params={x['key']:x['value'] for x in node['params']}
    for k in ('M','N','K'):need(params.get(k)==w[k], 'measured '+k+' differs from workload')
    for k,v in c['launch']['params'].items():need(params.get(k)==v,'measured tuning differs: '+k)
    need(len(m['plugins'])==1,'multi-plugin measurements unsupported')
    plugin=m['plugins'][0]
    if c['identity']['kind']=='plugin':need(plugin['sha256']==c['identity']['sha256'],'binary differs from planned candidate')
    if p['backend']=='aql':need(plugin['sha256']==c['identity'].get('plugin_sha256'),'GPU binary not bound to candidate')
    # Dtype/layout interpretation is owned by the reviewed adapter, not guessed from raw quant bits.
    return {'workload':w,'candidate_id':c['id'],'candidate_sha256':digest(c),
        'backend':p['backend'],'arch':p['arch'],'environment_sha256':digest(p['environment']),
        'metric':'window_us','samples_us':rt['samples_us'],'correctness_status':'RUNTIME_CHECKS_PASS',
        'evidence':{'origin':'radreplay-verified-import','report_sha256':file_hash(report_path),
            'fixture_sha256':file_hash(fixture_path),'worker_sha256':r['worker_sha256'],
            'plugin_sha256':plugin['sha256'],'path':str(report_path)}}


def profile_from_replay(base:dict,report_path:Path,clock_power:str,compiler_id:str)->dict:
    """Capture observed identity; user-declared clock profile is NOT measured telemetry."""
    import copy
    profile(base);r=read(report_path);rt=r.get('runtime',{})
    need(r.get('status')=='RUNTIME_CHECKS_PASS' and r.get('runtime_executed') is True,'passing replay required')
    need(rt.get('backend')==base['backend'] and rt.get('device',{}).get('arch')==base['arch'],'profile/device target mismatch')
    need(not rt.get('worker_build',{}).get('host_asan') and rt.get('samples_us'),'need uninstrumented measured run')
    need(isinstance(clock_power,str) and clock_power.strip() and clock_power!='UNMEASURED','explicit clock/power description required')
    need(isinstance(compiler_id,str) and compiler_id.strip(),'compiler/build description required')
    p=copy.deepcopy(base);p['id']=base['id']+'-observed-environment'
    p['environment']={'device_name':rt['device']['name'],'device_identity':rt['device'],
        'worker_sha256':r['worker_sha256'],'measurement_mode':rt['measurement'],'platform':r['host']['platform'],
        'clock_power':clock_power,'compiler_id':compiler_id,'profile_declarations_verified':False}
    p['notes']=p.get('notes','')+' Observed identity only: this command does not calibrate throughput or prove clock stability.'
    return profile(p)
