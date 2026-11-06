"""Supervisor for the Radiance worker. No executable path is taken from a fixture."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from . import __version__
from .format import ReplayError, blob, file_hash, read_json, require, safe_file, validate, all_blobs, verified_json


def write_json(p: Path, value):
    p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'); p.chmod(0o600)


def isa_binding(manifest_path: Path, report_path: Path, required_plugins: set[str], binary_paths: dict[str, Path] | None = None) -> dict:
    """Bind evidence to exact plugins, not a convenient old green report."""
    m=read_json(manifest_path); r=read_json(report_path)
    require(m.get('schema')=='radiance-rdna4-isa/2' and r.get('schema')==m['schema'],'ISA evidence schema mismatch')
    require(r.get('manifest_sha256')==file_hash(manifest_path),'ISA report does not match manifest')
    require(r.get('status')=='STATIC_CHECKS_PASS' and r.get('strict_review') is True,'ISA gate not strict/pass')
    require(m.get('collection_complete') is True and not m.get('errors'),'ISA collection incomplete')
    require(required_plugins <= {x['sha256'] for x in m['inputs']},'not every exact plugin hash is an ISA input')
    if binary_paths is not None:
        from radisa.binary import extract_code_objects, MAX_BYTES
        covered={o['sha256'] for o in m['objects']}
        for h in required_plugins:
            path=binary_paths[h]
            require(path.stat().st_size<=MAX_BYTES and file_hash(path)==h,'plugin changed before ISA binding')
            objects,events=extract_code_objects(path.read_bytes(),str(path))
            require(not any(e['status']=='unsupported' for e in events),'unsupported plugin code-object container')
            relevant={o['sha256'] for o in objects if o['target']=='gfx1201'}
            require(relevant and relevant<=covered,'plugin has no covered gfx1201 code objects; input-list membership alone is insufficient')
            require(not any(o['target']=='gfx12-generic' or o['target'].startswith('unknown-') for o in objects),'unresolved generic/unknown plugin target')
    identities=set()
    for o in m['objects']:
        p=safe_file(manifest_path.parent,o['path'])
        require(p.stat().st_size==o['bytes'] and file_hash(p)==o['sha256'],'ISA object hash mismatch')
        identities.update(o['sha256']+':'+k['symbol'] for k in o['kernels'])
    require(identities and identities=={k['identity'] for k in r['kernels']},'ISA kernel coverage mismatch')
    c=r['coverage']
    require(c['expected_kernels']==c['analyzed_kernels']==c['fully_decoded_kernels']==len(identities),'ISA decoder coverage incomplete')
    require(c['expected_bytes']==c['decoded_bytes'],'ISA byte coverage incomplete')
    require(not any(f['severity'] in ('error','incomplete','review') for f in r['global_findings']), 'ISA global findings block replay')
    require(not any(f['severity'] in ('error','incomplete','review') for k in r['kernels'] for f in k['findings']), 'ISA kernel findings block replay')
    return {'status':'STATIC_CHECKS_PASS','manifest_sha256':file_hash(manifest_path),
            'report_sha256':file_hash(report_path),'scope':'submitted plugin binaries; launcher-to-dispatch mapping not proved'}


def telemetry() -> list[dict]:
    """Raw, best-effort readings; never invent units/versions of binary gpu_metrics."""
    out=[]
    for p in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        row={'path':str(p)}
        for name in ('pp_dpm_sclk','pp_dpm_mclk','power_dpm_force_performance_level'):
            try: row[name]=(p/name).read_text()[:8192]
            except OSError: pass
        for q in sorted((p/'hwmon').glob('hwmon*/*_input')):
            try: row[str(q.relative_to(p))]=q.read_text().strip()
            except OSError: pass
        out.append(row)
    return out


def report_files(out: Path, report: dict) -> None:
    write_json(out/'report.json',report)
    cov={'schema':'radiance-replay-coverage/1','fixture':report.get('fixture',{}),
         'runtime_status':report['status'],'isa':report.get('isa',{}),
         'all_radiance_kernels_covered':False,'production_equivalence_proved':False}
    write_json(out/'coverage.json',cov)
    lines=['# Radiance kernel snapshot replay','',f"**Status: {report['status']}**",'',
           'This report covers this fixture only. It is not full ISA, memory-safety, or GPU-wedge certification.',
           '', '## Input and execution evidence', '',
           '```json',json.dumps(report.get('fixture',{}),indent=2),'```','',
           '## Runtime', '', '```json',json.dumps(report.get('runtime',{}),indent=2),'```','',
           '## Limits','',
           '- Ordered launcher-level replay; not arbitrary raw AQL packets or device-side chains.',
           '- Single device/rank only. Captured baseline is a regression oracle, not independent correctness.',
           '- Guard bytes alone do not detect reads. Host ASan checks only instrumented host paths.',
           '- Diagnostic checkpoints serialize operations; they do not prove race freedom or production timing.',
           '- Output variation excludes readable aliases, padded/strided views and custom encodings.',
           '- Same-operation writes restored before a checkpoint can still be missed.',
           '- Queue-window time is not per-dispatch latency and is never converted to tokens/s.']
    if report.get('error'): lines += ['', '## Failure', '', report['error']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    suite=ET.Element('testsuite',name='radiance-snapshot-replay',tests='1')
    case=ET.SubElement(suite,'testcase',name=report.get('fixture',{}).get('id','input'))
    ok=report['status']=='RUNTIME_CHECKS_PASS'
    suite.set('failures','1' if report['status']=='FAIL' else '0'); suite.set('errors','0' if ok or report['status']=='FAIL' else '1')
    if not ok: ET.SubElement(case,'failure' if report['status']=='FAIL' else 'error',message=report['status']).text=report.get('error',json.dumps(report.get('runtime',{})))
    ET.ElementTree(suite).write(out/'junit.xml',encoding='utf-8',xml_declaration=True)


def run(args) -> int:
    out=Path(args.out).resolve(); require(not out.exists(),'result directory exists')
    out.mkdir(parents=True,mode=0o700)
    report={'schema':'radiance-replay-result/1','version':__version__,'status':'INCOMPLETE',
            'runtime_executed':False,'production_equivalence_proved':False,
            'options':{'warmups':args.warmups,'repetitions':args.reps,'timeout_seconds':args.timeout,'gate':args.gate,'diagnostics':args.diagnostics,'diagnostic_only':args.diagnostic_only,'memory_check':args.memory_check,'trace_dispatches':args.trace_dispatches},
            'host':{'platform':platform.platform(),'python':sys.version.split()[0]},'telemetry_before':telemetry()}
    code=2
    try:
        require(math.isfinite(args.timeout) and args.timeout>0,'invalid timeout')
        require(not args.diagnostic_only or args.diagnostics=='contracts','diagnostic-only requires contract diagnostics')
        require(args.memory_check=='none' or (args.diagnostic_only and args.diagnostics=='contracts'),
                'host-asan requires --diagnostic-only and --diagnostics contracts')
        require(0<=args.warmups<=1000 and 1<=args.reps<=10000,'invalid repetition count')
        path=Path(args.manifest).resolve(); info=validate(path,budget=args.max_bytes)
        report['fixture']=info; m=verified_json(path,info['manifest_sha256'])
        require(not m.get('checkpoints') or args.diagnostics=='contracts','checkpoints require diagnostics')
        require(not args.trace_dispatches or (m['target']['backend']=='aql' and args.diagnostic_only and args.diagnostics=='contracts'),'dispatch tracing requires AQL contract diagnostics and --diagnostic-only')
        if args.memory_check=='host-asan':
            require(m['target']['backend']=='host','GPU ASan is not certified for this gfx1201/AQL path')
        require(args.execute_trusted_code,'execution requires --execute-trusted-code (plugins are native code)')
        plugins={}
        for binding in args.plugin:
            key,sep,value=binding.partition('='); require(sep and key not in plugins,'use unique --plugin ID=/path/plugin.so')
            plugins[key]=str(Path(value).resolve(strict=True))
        require(set(plugins)=={p['id'] for p in m['plugins']},'plugin mapping differs from exact fixture inventory')
        for p in m['plugins']:
            require(file_hash(Path(plugins[p['id']]))==p['sha256'],'plugin hash mismatch; create an explicit candidate fixture')
        worker=Path(args.worker).resolve(strict=True); require(os.access(worker,os.X_OK),'worker is not executable')
        report['worker_sha256']=file_hash(worker)
        if args.memory_check=='host-asan':
            from radisa.binary import ELF
            inspected=[]
            for codepath in [worker,*map(Path,plugins.values())]:
                names={sym.name for sym in ELF(codepath.read_bytes()).symbols}
                require(any(n.startswith('__asan_report_') or n.startswith('__asan_load') or n.startswith('__asan_store') for n in names),
                        'host ASan instrumentation symbols absent: '+str(codepath))
                inspected.append({'path':str(codepath),'sha256':file_hash(codepath)})
            report['host_asan']={'binary_evidence':inspected,
                'scope':'symbol presence is a necessary check, not proof that every load or inline-assembly path is instrumented'}

        if m['target']['backend']=='aql':
            if args.isa_manifest and args.isa_report:
                report['isa']=isa_binding(Path(args.isa_manifest).resolve(),Path(args.isa_report).resolve(),{p['sha256'] for p in m['plugins']},{p['sha256']:Path(plugins[p['id']]) for p in m['plugins']})
            else:
                require(args.exploratory_without_isa,'GPU execution requires --isa-manifest and --isa-report')
                report['isa']={'status':'NOT_CHECKED','exploratory':True}
        else: report['isa']={'status':'NOT_APPLICABLE_HOST'}
        # Copy validated content into private evidence: worker never consumes changed package paths.
        stage=out/'staged'; stage.mkdir(mode=0o700)
        copied={}
        for b in all_blobs(m):
            if b['sha256'] not in copied:
                src=blob(path.parent,b,args.max_bytes)
                dst=stage/b['sha256']; shutil.copyfile(src,dst); dst.chmod(0o600)
                require(file_hash(dst)==b['sha256'],'blob changed during staging')
                copied[b['sha256']]=str(dst)
        plan={'schema':'radiance-replay-worker/1','fixture':m,'fixture_sha256':info['manifest_sha256'],
              'plugin_paths':plugins,'blob_paths':copied,'warmups':args.warmups,'repetitions':args.reps,
              'device':args.device,'gate':args.gate,'max_bytes':args.max_bytes,'out':str(out),
              'diagnostics':args.diagnostics,'diagnostic_only':args.diagnostic_only,'memory_check':args.memory_check,'trace_dispatches':args.trace_dispatches}
        if args.trace_dispatches:
            from .dispatch import allowed_identities
            require(report['isa']['status']=='STATIC_CHECKS_PASS','dispatch tracing requires strict static evidence')
            plan['allowed_dispatches']=sorted(allowed_identities(read_json(Path(args.isa_manifest))))
        planpath=out/'worker-plan.json'; write_json(planpath,plan)
        cmd=[str(worker),str(planpath)]
        report['command']=cmd
        start=time.monotonic()
        with (out/'worker.stdout').open('wb') as stdout,(out/'worker.stderr').open('wb') as stderr:
            env=os.environ.copy()
            if args.memory_check=='host-asan':
                # No recovery after corruption; leaks are irrelevant to this process-lifetime worker.
                env['ASAN_OPTIONS']='allow_user_poisoning=1:halt_on_error=1:abort_on_error=0:exitcode=86:detect_leaks=0'
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True,env=env)
            report['runtime_executed']=True
            try: rc=proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: pass
                raise ReplayError('TIMEOUT: process group killed; stop GPU testing and inspect device health. No automatic retry/reset.')
        report['wall_seconds']=time.monotonic()-start
        resultpath=out/'worker-result.json'
        if args.memory_check=='host-asan' and rc!=0:
            stderr=(out/'worker.stderr').read_text(errors='replace')
            if 'ERROR: AddressSanitizer:' in stderr:
                report['status']='FAIL';code=1
                report['error']='Host AddressSanitizer detected an invalid memory access; see worker.stderr'
                report['runtime']={'status':'FAIL','backend':'host','memory_check':'HOST_ASAN_DETECTED_ERROR',
                                   'fixture_sha256':info['manifest_sha256']}
                return code
        require(resultpath.exists(),f'worker exited {rc} without complete result')
        r=read_json(resultpath); report['runtime']=r
        require(r.get('schema')=='radiance-replay-worker-result/1','invalid worker report')
        require(r.get('fixture_sha256')==info['manifest_sha256'],'worker tested different fixture')
        require(r.get('backend')==m['target']['backend'],'worker backend mismatch; no silent host fallback')
        for p in m['plugins']:
            require(file_hash(Path(plugins[p['id']]))==p['sha256'],'plugin changed during execution')
        if rc==1 and r.get('status')=='FAIL': report['status']='FAIL'; code=1
        elif rc==0 and r.get('status')=='RUNTIME_CHECKS_PASS':
            count=0 if args.diagnostic_only else args.warmups+args.reps
            require(r.get('checked_iterations')==count,'not every timed/warmup input was checked')
            require(len(r.get('samples_us',[]))==(0 if args.diagnostic_only else args.reps),'missing timing samples')
            if args.diagnostics=='contracts':
                d=r.get('diagnostics',{})
                require(d.get('status')=='PASS' and d.get('completed_passes')==3 and d.get('checked_nodes')==info['node_count'],
                        'missing or partial contract diagnostics; old workers cannot pass')
                require(d.get('per_operation_write_contracts') is True and d.get('output_poison_variants')==2
                        and d.get('scratch_contracts') is True,'missing diagnostic dimensions')
                if args.memory_check=='host-asan':
                    require(d.get('read_checks')=='HOST_ASAN_INSTRUMENTED_PATHS','missing host read checks')
            if args.diagnostic_only:
                require(r.get('measurement')=='NOT_MEASURED_DIAGNOSTIC_ONLY','diagnostic timing mislabeled')
            require(all(type(t) in (int,float) and math.isfinite(t) and t>=0 for t in r['samples_us']),'invalid time values')
            require(r.get('checks_per_iteration')==len(m['checks']),'output-check inventory mismatch')
            if args.trace_dispatches:
                from .dispatch import bind
                t=out/'dispatch-trace.json'
                require(t.exists() and r.get('dispatch_trace',{}).get('sha256')==file_hash(t),'missing/changed dispatch trace')
                report['dispatch_binding']=bind(t,m,info['manifest_sha256'],read_json(Path(args.isa_manifest)),args.device)
            if report['isa']['status']=='NOT_CHECKED': report['status']='INCOMPLETE'; code=2
            else: report['status']='RUNTIME_CHECKS_PASS'; code=0
        else:
            report['status']='INCOMPLETE'; report['error']=f"worker exit {rc}: {r.get('error','incomplete evidence')}"
    except (ReplayError,OSError,ValueError,KeyError,TypeError) as exc:
        report['error']=str(exc)
        if not math.isfinite(args.timeout): report['options']['timeout_seconds']=str(args.timeout)
    finally:
        report['telemetry_after']=telemetry(); report['exit_code']=code; report_files(out,report)
    print(f"{report['status']}: {out/'report.md'}")
    return code


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description='Portable Radiance snapshot validation and launcher-level replay')
    p.add_argument('--version',action='version',version=__version__)
    s=p.add_subparsers(dest='command',required=True)
    v=s.add_parser('validate');v.add_argument('manifest');v.add_argument('--max-bytes',type=int,default=2*1024**3)
    r=s.add_parser('run');r.add_argument('manifest');r.add_argument('--out',required=True);r.add_argument('--worker',required=True)
    r.add_argument('--plugin',action='append',default=[]);r.add_argument('--execute-trusted-code',action='store_true')
    r.add_argument('--max-bytes',type=int,default=2*1024**3);r.add_argument('--reps',type=int,default=10);r.add_argument('--warmups',type=int,default=2)
    r.add_argument('--timeout',type=float,default=120);r.add_argument('--device',type=int,default=0)
    r.add_argument('--gate',action='store_true',help='single-stream AQL only; never silently falls back')
    r.add_argument('--isa-manifest');r.add_argument('--isa-report')
    r.add_argument('--diagnostics',choices=['contracts','none'],default='contracts',
                   help='contracts: three serial diagnostic passes before uninstrumented measurement (default)')
    r.add_argument('--diagnostic-only',action='store_true',help='do not produce performance samples')
    r.add_argument('--memory-check',choices=['none','host-asan'],default='none',
                   help='host-asan requires an instrumented worker AND instrumented plugins; not GPU instrumentation')
    r.add_argument('--trace-dispatches',action='store_true',help='diagnostic-only exact AQL dispatch whitelist; optional backend patch required')
    r.add_argument('--exploratory-without-isa',action='store_true',help='explicitly incomplete GPU exploration, never a clean release pass')
    c=s.add_parser('candidate');c.add_argument('manifest');c.add_argument('--out',required=True);c.add_argument('--plugin',action='append',required=True)
    e=s.add_parser('expect');e.add_argument('manifests',nargs='+');e.add_argument('--out',required=True);e.add_argument('--scope',required=True)
    g=s.add_parser('coverage');g.add_argument('--expect',required=True);g.add_argument('--report',action='append',default=[]);g.add_argument('--out',required=True)
    q=s.add_parser('qualify');q.add_argument('--expect',required=True);q.add_argument('--report',action='append',default=[]);q.add_argument('--out',required=True)
    q.add_argument('--profile',choices=['host-contracts','single-gpu','production-tp2'],required=True);q.add_argument('--numeric-policy')
    q=s.add_parser('row-coverage');q.add_argument('--inventory',required=True);q.add_argument('--manifest',action='append',required=True);q.add_argument('--report',action='append',default=[]);q.add_argument('--out',required=True)
    q=s.add_parser('ab');q.add_argument('--baseline',required=True);q.add_argument('--candidate',required=True);q.add_argument('--out',required=True)
    for arm in ('baseline','candidate'):
        q.add_argument('--'+arm+'-worker',required=True);q.add_argument('--'+arm+'-plugin',action='append',required=True)
        q.add_argument('--'+arm+'-isa-manifest');q.add_argument('--'+arm+'-isa-report')
    q.add_argument('--blocks',type=int,default=5);q.add_argument('--warmups',type=int,default=2);q.add_argument('--reps',type=int,default=20)
    q.add_argument('--timeout',type=float,default=120);q.add_argument('--device',type=int,default=0);q.add_argument('--gate',action='store_true')
    q.add_argument('--execute-trusted-code',action='store_true')
    q=s.add_parser('run-corpus');q.add_argument('--expect',required=True);q.add_argument('--worker',required=True);q.add_argument('--plugin',action='append',required=True);q.add_argument('--out',required=True)
    q.add_argument('--execute-trusted-code',action='store_true');q.add_argument('--reps',type=int,default=10);q.add_argument('--warmups',type=int,default=2)
    q.add_argument('--timeout',type=float,default=120);q.add_argument('--device',type=int,default=0);q.add_argument('--isa-manifest');q.add_argument('--isa-report')
    q.add_argument('--gate',action='store_true');q.add_argument('--diagnostic-only',action='store_true');q.add_argument('--trace-dispatches',action='store_true')
    args=p.parse_args(argv)
    try:
        if args.command=='validate': print(json.dumps(validate(Path(args.manifest),budget=args.max_bytes),indent=2));return 0
        if args.command=='candidate':
            from .corpus import candidate
            print(candidate(Path(args.manifest).resolve(),Path(args.out).resolve(),args.plugin));return 0
        if args.command=='expect':
            from .corpus import expect
            print(json.dumps(expect([Path(x) for x in args.manifests],Path(args.out).resolve(),args.scope),indent=2));return 0
        if args.command=='coverage':
            from .corpus import coverage
            code,result=coverage(Path(args.expect).resolve(),[Path(x).resolve() for x in args.report],Path(args.out).resolve())
            print(result['status']);return code
        if args.command=='qualify':
            from .qualification import qualify
            code,r=qualify(Path(args.expect).resolve(),[Path(x).resolve() for x in args.report],Path(args.out).resolve(),profile=args.profile,policy=Path(args.numeric_policy).resolve() if args.numeric_policy else None)
            print(r['status']);return code
        if args.command=='row-coverage':
            from .qualification import row_coverage
            code,r=row_coverage(Path(args.inventory).resolve(),[Path(x).resolve() for x in args.manifest],[Path(x).resolve() for x in args.report],Path(args.out).resolve())
            print(r['status']);return code
        if args.command=='ab':
            from .ab import experiment
            code,r=experiment(args);print(r['status']);return code
        if args.command=='run-corpus':
            from .batch import execute
            code,r=execute(args);print(r['status']);return code
        return run(args)
    except (OSError,ValueError,KeyError,TypeError) as ex:
        print(f'INCOMPLETE: {ex}',file=sys.stderr);return 2
