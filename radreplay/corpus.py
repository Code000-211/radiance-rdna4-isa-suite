"""Explicit candidate rebinding and scoped, hash-pinned corpus coverage.
Reports are engineering evidence, not authenticated attestations.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from .format import require, read_json, file_hash, validate, safe_file, keys, HASH, digest, all_blobs

def write(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');path.chmod(0o600)

def candidate(source: Path, out: Path, bindings: list[str]) -> Path:
    info=validate(source);m=read_json(source);require(not out.exists(),'candidate output exists')
    changes={}
    for binding in bindings:
        key,sep,path=binding.partition('=');require(sep and key not in changes,'use unique ID=/path/plugin.so bindings')
        changes[key]=file_hash(Path(path).resolve(strict=True))
    require(changes and changes.keys()<={p['id'] for p in m['plugins']},'unknown/empty candidate plugin bindings')
    lineage=[]
    for p in m['plugins']:
        if p['id'] in changes:
            require(p['sha256']!=changes[p['id']],'candidate plugin is byte-identical')
            lineage.append({'id':p['id'],'before':p['sha256'],'after':changes[p['id']]});p['sha256']=changes[p['id']]
    m['lineage']={'purpose':'explicit-candidate-rebind','parent_id':m['id'],
                  'parent_manifest_sha256':info['manifest_sha256'],'plugin_changes':lineage}
    m['id']=m['id'][:90]+'-candidate-'+digest(json.dumps(lineage,sort_keys=True).encode())[:12]
    out.mkdir(parents=True,mode=0o700);(out/'blobs').mkdir(mode=0o700)
    for b in all_blobs(m):
        src=safe_file(source.parent,b['path']);dst=out/'blobs'/b['sha256']
        if not dst.exists():shutil.copyfile(src,dst);dst.chmod(0o600)
        require(file_hash(dst)==b['sha256'],'blob changed during candidate copy');b['path']='blobs/'+b['sha256']
    path=out/'manifest.json';write(path,m);validate(path);return path

def expect(sources: list[Path], output: Path, scope: str) -> dict:
    require(scope.strip() and len(scope)<=4096,'state a nonempty coverage scope')
    require(not output.exists(),'expected corpus already exists')
    require(output.parent.is_dir(),'corpus parent must already exist')
    cases=[];seen=set()
    for path in sources:
        path=path.resolve();info=validate(path)
        require(info['id'] not in seen,'duplicate fixture id in expected corpus');seen.add(info['id'])
        require(path.is_relative_to(output.parent.resolve()),'place corpus JSON above/alongside its fixture directories')
        cases.append({'id':info['id'],'manifest':str(path.relative_to(output.parent.resolve())),
                      'sha256':info['manifest_sha256']})
    require(cases,'empty expected corpus')
    value={'schema':'radiance-replay-corpus/1','scope':scope,'cases':cases};write(output,value);return value

def coverage(expected: Path, reports: list[Path], out: Path) -> tuple[int,dict]:
    require(not out.exists(),'coverage output exists');out.mkdir(parents=True,mode=0o700)
    result={'schema':'radiance-replay-corpus-result/1','status':'INCOMPLETE','scope':None,
            'all_radiance_kernels_covered':False,'production_equivalence_proved':False,'cases':[], 'errors':[]}
    code=2
    try:
        m=read_json(expected);keys(m,{'schema','scope','cases'});require(m['schema']=='radiance-replay-corpus/1','wrong corpus schema')
        require(isinstance(m['scope'],str) and m['scope'].strip(),'empty scope')
        result['scope']=m['scope'];result['corpus_sha256']=file_hash(expected)
        require(isinstance(m['cases'],list) and m['cases'],'empty corpus')
        cases={}
        for c in m['cases']:
            keys(c,{'id','manifest','sha256'});p=safe_file(expected.parent,c['manifest']);i=validate(p)
            require(c['sha256']==i['manifest_sha256'] and c['id']==i['id'],'expected fixture changed')
            require(c['sha256'] not in cases and c['id'] not in [x['id'] for x in cases.values()],'duplicate expected fixture')
            cases[c['sha256']]=dict(c,info=i,backend=read_json(p)['target']['backend'])
        by_hash={}
        for p in reports:
            r=read_json(p);require(r.get('schema')=='radiance-replay-result/1','wrong runtime report schema')
            h=r.get('fixture',{}).get('manifest_sha256');require(h in cases,'unexpected report/fixture')
            require(h not in by_hash,'duplicate reports for one expected fixture; do not select only a green attempt')
            require(r['fixture']['id']==cases[h]['id'],'report fixture id mismatch')
            require(r.get('status') in ('RUNTIME_CHECKS_PASS','FAIL','INCOMPLETE'),'unknown report status')
            if r['status']=='RUNTIME_CHECKS_PASS':
                rt=r.get('runtime',{});opts=r.get('options',{})
                require(r.get('runtime_executed') is True and r.get('exit_code')==0,'pass without execution evidence')
                require(rt.get('status')=='RUNTIME_CHECKS_PASS' and rt.get('fixture_sha256')==h and rt.get('backend')==cases[h]['backend'],'runtime report binding mismatch')
                require(type(opts.get('repetitions')) is int and opts['repetitions']>0 and type(opts.get('warmups')) is int and opts['warmups']>=0,'missing iteration contract')
                count=0 if opts.get('diagnostic_only',False) else opts['warmups']+opts['repetitions']
                require(rt.get('checked_iterations')==count,'unchecked iteration')
                require(len(rt.get('samples_us',[]))==(0 if opts.get('diagnostic_only',False) else opts['repetitions']),'missing timings')
                if opts.get('diagnostics')=='contracts':
                    require(rt.get('diagnostics',{}).get('status')=='PASS','missing diagnostic evidence')
                require(r.get('isa',{}).get('status')==('STATIC_CHECKS_PASS' if cases[h]['backend']=='aql' else 'NOT_APPLICABLE_HOST'),'unapproved ISA scope')
            by_hash[h]=(r,p)
        statuses=[]
        for h,c in cases.items():
            r,p=by_hash.get(h,({},None));status=r.get('status','MISSING');statuses.append(status)
            result['cases'].append({'id':c['id'],'fixture_sha256':h,'backend':c['backend'],'status':status,
                                    'report_sha256':file_hash(p) if p else None,'kernel_rows':c['info']['kernel_rows'],
                                    'provenance':c['info']['provenance']})
        result['expected_cases']=len(cases);result['passed_cases']=statuses.count('RUNTIME_CHECKS_PASS')
        if 'FAIL' in statuses:result['status']='FAIL';code=1
        elif all(s=='RUNTIME_CHECKS_PASS' for s in statuses):result['status']='CORPUS_CHECKS_PASS';code=0
    except (ValueError,OSError,KeyError,TypeError) as e:result['errors'].append(str(e))
    result['exit_code']=code;write(out/'coverage.json',result)
    lines=['# Scoped replay corpus coverage','',f"**{result['status']}**",'',str(result['scope']),'',
           'This covers only the explicitly approved fixtures, never all Radiance kernels by inference.','',
           '| Fixture | Backend | Result |','|---|---|---|']
    lines += [f"| {c['id']} | {c['backend']} | {c['status']} |" for c in result['cases']]
    lines += ['',*result['errors']];(out/'coverage.md').write_text('\n'.join(lines)+'\n')
    return code,result
