"""Scoped acceptance, independent of a convenient green runtime status.
No boolean supplied by a caller can enable unimplemented TP2 or GPU read checks.
JSON reports are engineering evidence, not authenticated remote attestations.
"""
from pathlib import Path
import json
import math
from .format import require,read_json,validate,file_hash,keys,all_blobs,safe_file
from .corpus import write,coverage

def diagnostic_evidence(r, node_count):
    d=r.get('runtime',{}).get('diagnostics',{})
    return (d.get('status')=='PASS' and d.get('completed_passes')==3
            and d.get('checked_nodes')==node_count and d.get('per_operation_write_contracts') is True
            and d.get('output_poison_variants')==2 and d.get('scratch_contracts') is True)

def numeric_policy(m, policy):
    """Check a versioned, pre-approved policy; unknown ops/dtypes never inherit a catch-all."""
    require(policy.get('schema')=='radiance-numeric-policy/1' and policy.get('version'),'missing versioned numeric policy')
    rules=policy.get('rules',[policy] if 'op' in policy else [])
    require(isinstance(rules,list) and rules,'empty policy rules')
    views={v['id']:v for v in m['views']}
    for check in m['checks']+[c for cp in m.get('checkpoints',[]) for c in cp['checks']]:
        if check['mode']=='bytes':continue
        v=views[check['view']];writers=set()
        for n in m['nodes']:
            if n['kind']!='kernel':continue
            for a in n['accesses']:
                u=views[a['view']]
                if ('w' in a['mode'] and u['allocation']==v['allocation'] and
                    max(u['offset'],v['offset'])<min(u['offset']+u['bytes'],v['offset']+v['bytes'])):writers.add(n['op'])
        require(writers,'numeric view has no declared kernel writer')
        for op in writers:
            matches=[x for x in rules if x.get('op')==op and x.get('dtype')==v['dtype']]
            require(len(matches)==1,'missing/ambiguous numeric policy for '+op)
            p=matches[0]
            for key in ('atol','rtol','rel_l2'):
                x=p.get(key);require(type(x) in (float,int) and math.isfinite(x) and 0<=x<=1e6,'invalid policy tolerance')
                require(check[key]<=x,'fixture relaxes pre-approved '+key+' for '+op)
            require(check['nonfinite']==p.get('nonfinite'),'nonfinite policy mismatch')

def qualify(expected:Path,reports:list[Path],out:Path,*,profile:str,policy:Path|None=None):
    require(profile in ('host-contracts','single-gpu','production-tp2'),'unknown qualification profile')
    require(not out.exists(),'qualification output exists');out.mkdir(parents=True,mode=0o700)
    code,cov=coverage(expected,reports,out/'corpus')
    result={'schema':'radiance-qualification/1','profile':profile,'status':'INCOMPLETE',
            'corpus_sha256':file_hash(expected),'requirements':[], 'all_kernel_coverage':False,
            'full_memory_safety_proved':False,'production_equivalence_proved':False}
    checks=result['requirements']
    def item(name,passed,detail):checks.append({'name':name,'status':'PASS' if passed else 'MISSING','detail':detail})
    item('explicit-corpus',code==0,'All predeclared cases, no missing/duplicate/stale reports.')
    m=read_json(expected)
    by_hash={read_json(p).get('fixture',{}).get('manifest_sha256'):read_json(p) for p in reports}
    pol=read_json(policy) if policy else None
    if policy:result['numeric_policy_sha256']=file_hash(policy)
    for c in m.get('cases',[]):
        try:
            path=safe_file(expected.parent,c['manifest']);info=validate(path)
            require(info['manifest_sha256']==c['sha256'] and info['id']==c['id'],'stale approved fixture')
            fixture=read_json(path);r=by_hash.get(c['sha256'],{})
        except (ValueError,OSError,KeyError,TypeError) as e:
            item(str(c.get('id','invalid'))+':fixture',False,str(e));continue
        item(c['id']+':contracts',diagnostic_evidence(r,info['node_count']),'Serial contract diagnostics, not race-freedom evidence.')
        item(c['id']+':independent-reference',fixture['provenance']['reference']=='independent','Captured baselines are regression oracles only.')
        try:
            if any(x['mode']=='float' for x in fixture['checks']+[x for cp in fixture.get('checkpoints',[]) for x in cp['checks']]):
                require(pol is not None,'numeric policy not supplied');numeric_policy(fixture,pol)
            item(c['id']+':numeric-policy',True,'Exact bytes, or thresholds within pinned explicit policy.')
        except ValueError as e:item(c['id']+':numeric-policy',False,str(e))
        backend=r.get('runtime',{}).get('backend')
        item(c['id']+':backend',backend==('host' if profile=='host-contracts' else 'aql'),'Actual worker backend must match the requested profile.')
        if profile!='host-contracts':
            item(c['id']+':isa',r.get('isa',{}).get('status')=='STATIC_CHECKS_PASS','Strict hash-bound static gate.')
            item(c['id']+':dispatch-trace',r.get('dispatch_binding',{}).get('status')=='DISPATCH_BINDING_PASS','Actual ordered-launcher dispatch identities, including observed helpers.')
            # This release has no certified gfx1201 memory-read instrumentation.
            item(c['id']+':gpu-read-instrumentation',False,'NOT_IMPLEMENTED_FOR_GFX1201_AQL; host ASan cannot satisfy this requirement.')
        if profile=='production-tp2':
            item(c['id']+':production-input',fixture['provenance']['kind']=='production-snapshot','Seeded/model-weight fixtures are not captured production activations.')
    if profile=='production-tp2':
        item('tp2-protocol-tests',False,'Multi-rank/P2P runner is not implemented in v0.4.')
        item('raw-tape-equivalence',False,'Ordered launcher replay is not device-chain/raw-tape equivalence.')
        item('engine-window-calibration',False,'No supplied target-host engine-window evidence; do not infer tokens/s.')
    if code==1:result['status']='FAIL';exit_code=1
    elif all(x['status']=='PASS' for x in checks):result['status']='SCOPED_QUALIFICATION_PASS';exit_code=0
    else:exit_code=2
    result['exit_code']=exit_code;write(out/'qualification.json',result)
    lines=['# Scoped qualification',f"\n**{result['status']}** — {profile}",'',
           '| Requirement | Status | Evidence boundary |','|---|---|---|']
    lines += [f"| {x['name']} | {x['status']} | {x['detail']} |" for x in checks]
    (out/'qualification.md').write_text('\n'.join(lines)+'\n')
    return exit_code,result

def row_coverage(inventory:Path, manifests:list[Path], reports:list[Path],out:Path):
    inv=read_json(inventory);require(inv.get('schema')=='radiance-declared-rows/1','wrong row inventory schema')
    require(not out.exists(),'row coverage output exists');out.mkdir(parents=True,mode=0o700)
    h=inv['plugin_sha256'];declared={r['name']:r for r in inv['rows']}
    require(len(declared)==len(inv['rows'])==inv['declared_row_count'] and declared,'invalid/duplicate row inventory')
    infos={};cases={}
    for p in manifests:
        i=validate(p);require(i['manifest_sha256'] not in cases,'duplicate case');cases[i['manifest_sha256']]=read_json(p);infos[i['manifest_sha256']]=i
    seen={};errors=[];tested={name:[] for name in declared}
    for p in reports:
        r=read_json(p);fh=r.get('fixture',{}).get('manifest_sha256')
        require(fh in cases and fh not in seen,'unexpected/duplicate report');seen[fh]=r
        m=cases[fh]
        rt=r.get('runtime',{});opts=r.get('options',{})
        try:
            require(r.get('status')=='RUNTIME_CHECKS_PASS' and r.get('runtime_executed') is True and r.get('exit_code')==0,'not a completed runtime pass')
            require(rt.get('status')=='RUNTIME_CHECKS_PASS' and rt.get('fixture_sha256')==fh and rt.get('backend')==m['target']['backend'],'runtime binding mismatch')
            require(diagnostic_evidence(r,infos[fh]['node_count']),'missing contract diagnostics')
            require(opts.get('diagnostics')=='contracts','diagnostics not requested')
            require(type(opts.get('warmups')) is int and opts['warmups']>=0 and type(opts.get('repetitions')) is int and opts['repetitions']>0,'missing iteration contract')
            n=0 if opts.get('diagnostic_only') else opts['warmups']+opts['repetitions']
            require(rt.get('checked_iterations')==n and len(rt.get('samples_us',[]))==(0 if opts.get('diagnostic_only') else opts['repetitions']),'partial runtime evidence')
            require(r.get('isa',{}).get('status')==('NOT_APPLICABLE_HOST' if m['target']['backend']=='host' else 'STATIC_CHECKS_PASS'),'missing static evidence')
        except (ValueError,TypeError,KeyError) as e:
            errors.append(f"{fh}: {e}");continue
        plugins={p['id']:p for p in m['plugins']}
        for node in m['nodes']:
            if node['kind']!='kernel' or plugins[node['plugin']]['sha256']!=h:continue
            require(node['row'] in declared,'fixture row missing from inventory')
            tested[node['row']].append({'manifest_sha256':fh,'node':node['id'],'params':node['params']})
    for fh in cases:
        if fh not in seen:errors.append('missing runtime report: '+fh)
    rows=[{'name':n,'identity':h+':'+n,'status':'TESTED' if tested[n] else 'NOT_TESTED','observations':tested[n],
           'tuning_axes':declared[n]['tunables'],'all_tuning_combinations_tested':False} for n in declared]
    result={'schema':'radiance-row-coverage/1','status':'SCOPED_ROWS_COVERED' if not errors and all(tested.values()) else 'INCOMPLETE',
            'inventory_sha256':file_hash(inventory),'declared_rows':len(rows),'tested_rows':sum(bool(x) for x in tested.values()),
            'rows':rows,'errors':errors,'all_templates_tested':False}
    write(out/'rows.json',result);return (0 if result['status']=='SCOPED_ROWS_COVERED' else 2),result
