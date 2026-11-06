"""Execute an explicitly approved corpus. Stop on first nonpass, never reset/retry a GPU."""
from pathlib import Path
from .format import require,read_json,validate,safe_file,file_hash
from .corpus import write,coverage


def execute(args,invoke=None):
    from .cli import main
    if invoke is None:invoke=main
    require(args.execute_trusted_code,'batch execution requires --execute-trusted-code')
    expected=Path(args.expect).resolve();m=read_json(expected)
    require(m.get('schema')=='radiance-replay-corpus/1' and isinstance(m.get('cases'),list) and m['cases'],'invalid/empty approved corpus')
    out=Path(args.out).resolve();require(not out.exists(),'batch output exists')
    paths=[];seen=set();ids=set()
    for c in m['cases']:
        p=safe_file(expected.parent,c['manifest']);info=validate(p)
        require(info['manifest_sha256']==c['sha256'] and info['id']==c['id'],'approved case changed')
        require(c['sha256'] not in seen and c['id'] not in ids,'duplicate approved fixture')
        seen.add(c['sha256']);ids.add(c['id']);paths.append((c,p))
    bindings={}
    for b in args.plugin:
        key,sep,v=b.partition('=');require(sep and key not in bindings,'duplicate/invalid plugin binding');bindings[key]=v
    # Check every mapping/hash before running the first case; do not fail late on a typo.
    for _,p in paths:
        for pl in read_json(p)['plugins']:
            require(pl['id'] in bindings and file_hash(Path(bindings[pl['id']]))==pl['sha256'],'batch plugin mapping/hash mismatch')
    out.mkdir(parents=True,mode=0o700)
    report_paths=[];attempts=[];failure=None
    for index,(c,path) in enumerate(paths):
        require(file_hash(path)==c['sha256'],'approved case changed before execution')
        runout=out/f'{index:04d}-{c["id"]}'
        cmd=['run',str(path),'--out',str(runout),'--worker',args.worker,'--execute-trusted-code','--diagnostics','contracts',
             '--reps',str(args.reps),'--warmups',str(args.warmups),'--timeout',str(args.timeout),'--device',str(args.device)]
        for pl in read_json(path)['plugins']:cmd+=['--plugin',pl['id']+'='+bindings[pl['id']]]
        for flag in ('isa_manifest','isa_report'):
            if getattr(args,flag,None):cmd+=['--'+flag.replace('_','-'),str(getattr(args,flag))]
        for flag in ('gate','diagnostic_only','trace_dispatches'):
            if getattr(args,flag,False):cmd+=['--'+flag.replace('_','-')]
        rc=invoke(cmd);p=runout/'report.json'
        if p.exists():report_paths.append(p)
        attempts.append({'id':c['id'],'exit_code':rc,'report':str(p.relative_to(out)) if p.exists() else None})
        if rc!=0:
            failure=c['id'];break
    code,cov=coverage(expected,report_paths,out/'coverage')
    if failure:code=1 if any(a['exit_code']==1 for a in attempts) else 2
    result={'schema':'radiance-replay-batch/1','status':cov['status'] if not failure else ('FAIL' if any(a['exit_code']==1 for a in attempts) else 'INCOMPLETE'),
            'corpus_sha256':file_hash(expected),'requested_cases':len(paths),'attempted_cases':len(attempts),'stopped_at':failure,'attempts':attempts,
            'unattempted_cases':[c['id'] for c,_ in paths[len(attempts):]],'no_automatic_gpu_reset':True,'no_automatic_retries':True,'exit_code':code}
    write(out/'batch.json',result);return code,result
