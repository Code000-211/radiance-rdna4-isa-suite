"""Collect immutable evidence and audit every independently enumerated kernel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import sys
import xml.etree.ElementTree as ET
from . import __version__
from .binary import ELF, FormatError, MAX_BYTES, extract_code_objects, kernel_inventory, sha256
from .decode import decode_object, select_backend
from .rules import audit_kernel, finding

SCHEMA = 'radiance-rdna4-isa/2'


def no_duplicate_keys(pairs):
    result = {}
    for k,v in pairs:
        if k in result: raise FormatError(f'duplicate JSON key {k!r}')
        result[k]=v
    return result


def read_json(path: Path):
    if path.stat().st_size > 64*1024*1024: raise FormatError('JSON exceeds 64 MiB limit')
    return json.loads(path.read_text(), object_pairs_hook=no_duplicate_keys)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True)+'\n')


def safe_child(root: Path, name: str) -> Path:
    p = (root/name).resolve()
    if not p.is_relative_to(root.resolve()): raise FormatError('evidence path escapes its directory')
    return p


def digest_file(path: Path) -> tuple[str, bytes]:
    if not stat.S_ISREG(path.stat().st_mode): raise FormatError('not a regular input file')
    if path.stat().st_size > MAX_BYTES: raise FormatError(f'file exceeds {MAX_BYTES} byte limit')
    b=path.read_bytes()
    return sha256(b), b


def discover(paths: list[str], excluded: Path) -> list[Path]:
    found = set()
    for arg in paths:
        p=Path(arg).resolve()
        if not p.exists(): raise FormatError(f'input does not exist: {arg}')
        if p.is_file(): found.add(p); continue
        for d, dirs, names in os.walk(p, followlinks=False):
            links=[str(Path(d)/x) for x in dirs if (Path(d)/x).is_symlink()]
            if links:
                raise FormatError('symlink directories are not traversed; select final artifact files explicitly: '+', '.join(links))
            dirs[:] = sorted(x for x in dirs if x not in ('.git','__pycache__') and
                              not (Path(d)/x).is_symlink() and (Path(d)/x).resolve()!=excluded.resolve())
            for name in sorted(names):
                f=Path(d)/name
                if not stat.S_ISREG(f.stat().st_mode):
                    raise FormatError(f'non-regular input file: {f}')
                # Candidate selection by ELF/container magic, not filename suffix.
                with f.open('rb') as handle: magic=handle.read(32)
                if magic.startswith((b'\x7fELF',b'!<arch>',b'!<thin>',b'__CLANG_OFFLOAD_BUNDLE__',b'CCOB',b'BC\xc0\xde')):
                    found.add(f.resolve())
                elif name.endswith(('.hsaco','.co','.hipfb','.fatbin')):
                    found.add(f.resolve())  # malformed expected GPU evidence cannot vanish
    return sorted(found)


def collect(args) -> int:
    out=Path(args.out).resolve()
    if out.exists(): raise FormatError('output exists; use a fresh evidence directory (no stale-file merging)')
    paths=discover(args.inputs,out)
    out.mkdir(parents=True); (out/'objects').mkdir()
    manifest={'schema':SCHEMA,'suite_version':__version__,'created_utc':datetime.now(timezone.utc).isoformat(),
              'target':'gfx1201','scope':'all gfx1201 code objects discoverable in the explicit inputs',
              'input_arguments':args.inputs,'inputs':[],'objects':[],'events':[], 'errors':[]}
    objects={}
    for p in paths:
        try:
            h,b=digest_file(p)
            manifest['inputs'].append({'path':str(p),'sha256':h,'bytes':len(b)})
            extracted,events=extract_code_objects(b,str(p));manifest['events'].extend(events)
            # A generic bundle is a fallback: the loader prefers an exact target match, so it
            # is only reachable when this input ships no gfx1201 object at all. Say which of
            # the two it is instead of leaving the reader to look it up.
            exact=any(o['target']=='gfx1201' for o in extracted)
            for obj in extracted:
                if obj['target']!='gfx1201':
                    if obj['target']=='gfx12-generic' or obj['target'].startswith('unknown-'):
                        note=('fallback, this input also ships gfx1201 objects' if exact
                              else 'no gfx1201 object in this input, the fallback is reachable')
                        manifest['errors'].append(f"unsupported potentially relevant target {obj['target']} ({note}): {obj['origin']}")
                    manifest['events'].append({'origin':obj['origin'],'status':'other_target','target':obj['target']})
                    continue
                h=obj['sha256']
                if h in objects:
                    objects[h]['origins'].append(obj['origin']);continue
                entry={'sha256':h,'path':f'objects/{h}.hsaco','origins':[obj['origin']],
                       'bytes':len(obj['data']),'target':obj['target'],'kernels':[],'issues':[]}
                objects[h]=entry; (out/entry['path']).write_bytes(obj['data'])
                try:
                    ks,issues=kernel_inventory(ELF(obj['data']))
                    entry['issues']=issues
                    entry['kernels']=[{k:v for k,v in x.items() if k!='code_hex'} for x in ks]
                except (FormatError,ValueError,KeyError,TypeError) as ex:
                    entry['issues'].append(str(ex))
        except (OSError,FormatError,ValueError) as ex:
            manifest['errors'].append(f'{p}: {ex}')
    for e in manifest['events']:
        if e['status']=='unsupported': manifest['errors'].append(f"{e['origin']}: {e['reason']}")
    manifest['objects']=sorted(objects.values(),key=lambda o:o['sha256'])
    manifest['expected_kernels']=sum(len(o['kernels']) for o in manifest['objects'])
    if not paths: manifest['errors'].append('no binary input candidates discovered')
    if not manifest['expected_kernels']: manifest['errors'].append('zero enumerated target kernels')
    manifest['collection_complete']=not manifest['errors'] and all(not o['issues'] and all(not k['issues'] for k in o['kernels']) for o in manifest['objects'])
    write_json(out/'manifest.json',manifest)
    print(f"Collected {len(objects)} unique gfx1201 objects, {manifest['expected_kernels']} kernels; complete={manifest['collection_complete']}")
    return 0 if manifest['collection_complete'] else 2


def runtime_reconcile(path: Path, manifest: dict) -> dict:
    """Optional capture integration records RESOLVED kernels, not dispatches."""
    result={'mode':'resolved-kernels-only','source':str(path),'resolved_records':0,'errors':[],'matched':0,
            'process_completion_proved':False,'exhaustive_workload_coverage':False}
    table={o['sha256']: {k['name'] for k in o['kernels']} for o in manifest['objects']}
    raw=path.read_bytes()
    if not raw or not raw.endswith(b'\n'):
        result['errors'].append('empty/truncated capture log');return result
    if len(raw)>64*1024*1024: raise FormatError('capture log exceeds limit')
    for line in raw.splitlines():
        try:
            event=json.loads(line,object_pairs_hook=no_duplicate_keys)
            if event.get('event')!='resolve':
                result['errors'].append('capture error or unknown event: '+str(event));continue
            if type(event.get('bytes')) is not int or event['bytes']<1:
                raise FormatError('invalid captured object size')
            p=safe_child(path.parent,event['file'])
            h,b=digest_file(p)
            if len(b)!=event['bytes'] or h!=event.get('sha256'):
                raise FormatError('capture-time hash/size differs from the snapshot')
            result['resolved_records']+=1
            if h not in table or event['kernel'] not in table[h]:
                result['errors'].append(f"resolved kernel not audited: {h}:{event['kernel']}")
            else: result['matched']+=1
        except (OSError,ValueError,KeyError,TypeError) as ex:
            result['errors'].append(f'invalid capture record: {ex}')
    if result['resolved_records']==0: result['errors'].append('no resolved kernels in capture log')
    return result


def classify(findings: list[dict], strict_review: bool=False) -> tuple[str,int]:
    levels={x['severity'] for x in findings}
    if 'error' in levels: return 'FAIL',1
    if 'incomplete' in levels or (strict_review and 'review' in levels): return 'INCOMPLETE',2
    return 'STATIC_CHECKS_PASS',0


def write_reports(out: Path, report: dict):
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'report.json',report)
    coverage={k:report[k] for k in ('schema','status','coverage','runtime_coverage','full_isa_certification')}
    write_json(out/'coverage.json',coverage)
    # One test case per object+descriptor identity plus collection evidence.
    suite=ET.Element('testsuite',name='gfx1201-binary-isa',tests=str(len(report['kernels'])+1))
    rootcase=ET.SubElement(suite,'testcase',name='collection-and-coverage')
    globalstatus,_=classify(report['global_findings'],report['strict_review'])
    if globalstatus!='STATIC_CHECKS_PASS':
        child=ET.SubElement(rootcase,'error' if globalstatus=='INCOMPLETE' else 'failure',message=globalstatus)
        child.text=json.dumps(report['global_findings'],indent=2)
    failures=errors=0
    if globalstatus=='INCOMPLETE': errors+=1
    elif globalstatus=='FAIL': failures+=1
    for k in report['kernels']:
        case=ET.SubElement(suite,'testcase',classname=k['object_sha256'],name=k['symbol'])
        status,_=classify(k['findings'],report['strict_review'])
        if status!='STATIC_CHECKS_PASS':
            tag='failure' if status=='FAIL' else 'error'
            ET.SubElement(case,tag,message=status).text=json.dumps(k['findings'],indent=2)
            failures+=tag=='failure';errors+=tag=='error'
        ET.SubElement(case,'system-out').text=json.dumps({'scope':k['checks'],'resources':k['resources']})
    suite.set('failures',str(failures));suite.set('errors',str(errors))
    ET.ElementTree(suite).write(out/'junit.xml',encoding='utf-8',xml_declaration=True)
    lines=['# RDNA4 binary ISA report','',f"**Status: {report['status']}**",'',
           '**This is not full ISA, memory-safety, numerical, or GPU-wedge certification.**','',
           f"Expected kernels: {report['coverage']['expected_kernels']}; analyzed: {report['coverage']['analyzed_kernels']}; "
           f"fully decoded: {report['coverage']['fully_decoded_kernels']}",
           f"Expected bytes: {report['coverage']['expected_bytes']}; decoded: {report['coverage']['decoded_bytes']}",'',
           '## Global evidence findings','']
    lines += [f"- {x['severity'].upper()} `{x['rule']}`: {x['message']}" for x in report['global_findings']] or ['None.']
    for k in report['kernels']:
        name=k['symbol'].replace('`','\\`')
        lines+=['',f"## `{name}`",f"Object: `{k['object_sha256']}`",'',
                f"Decoded {k['decoded_bytes']}/{k['expected_bytes']} bytes; {k['instruction_count']} instructions."]
        for x in k['findings']:
            lines.append(f"- {x['severity'].upper()} `{x['rule']}` at {hex(x['pc']) if x['pc'] is not None else 'n/a'}: {x['message']}")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'disassembly').mkdir()
    for k in report['kernels']:
        if '_instructions' not in k: continue
        # A bounded hash avoids untrusted/overlong symbol names in filenames.
        name=k['object_sha256']+'-'+sha256(k['symbol'].encode())[:20]+'.s'
        (out/'disassembly'/name).write_text('\n'.join(f"{i['pc']:016x}: {i['text']} // {i['raw']}" for i in k['_instructions'])+'\n')
        del k['_instructions']
    # Keep raw instruction evidence in files instead of duplicating it in JSON.
    write_json(out/'report.json',report)


def audit(args) -> int:
    path=Path(args.manifest).resolve();manifest=read_json(path)
    if manifest.get('schema')!=SCHEMA or manifest.get('target')!='gfx1201':
        raise FormatError('unsupported evidence manifest schema/target')
    backend=select_backend(args.llvm_library,args.objdump)
    report={'schema':SCHEMA,'suite_version':__version__,'manifest_sha256':sha256(path.read_bytes()),
            'full_isa_certification':False,'decoder':backend,'strict_review':args.strict_review,
            'global_findings':[],'kernels':[], 'runtime_coverage':{'mode':'not-collected','exhaustive_workload_coverage':False}}
    gf=report['global_findings']
    for m in manifest.get('errors',[]): gf.append(finding('COLLECTION','incomplete',m))
    expected=0; expected_bytes=0; hashes=set(); identities=set()
    for obj in manifest['objects']:
        h=obj['sha256']
        if not re.fullmatch('[0-9a-f]{64}',h) or h in hashes:
            raise FormatError('duplicate/invalid object identity')
        hashes.add(h)
        expected+=len(obj['kernels']);expected_bytes+=sum(k.get('size',0) for k in obj['kernels'])
        for m in obj['issues']: gf.append(finding('OBJECT_EVIDENCE','incomplete',f'{h}: {m}'))
        try:
            p=safe_child(path.parent,obj['path']);actual,b=digest_file(p)
            if actual!=h or len(b)!=obj['bytes']: raise FormatError('code object hash/size mismatch')
            elf=ELF(b)
            if elf.target!='gfx1201': raise FormatError('actual ELF target differs from audit target')
            ks,errors=kernel_inventory(elf)
            for m in errors: gf.append(finding('OBJECT_EVIDENCE','incomplete',f'{h}: {m}'))
            clean=[{k:v for k,v in x.items() if k!='code_hex'} for x in ks]
            if clean!=obj['kernels']: raise FormatError('independent kernel inventory differs from collection manifest')
            usable=[k for k in ks if 'code_hex' in k]
            decoded=decode_object(p,usable,backend,args.timeout) if usable else {}
            if set(decoded)!={k['symbol'] for k in usable}: raise FormatError('decoder omitted or invented kernel records')
            for k in ks:
                ident=h+':'+k['symbol']
                if ident in identities: raise FormatError('duplicate kernel identity')
                identities.add(ident)
                d=decoded.get(k['symbol'],{'instructions':[],'issues':[],'decoded_bytes':0})
                result=audit_kernel(k,d);result['object_sha256']=h;result['identity']=ident
                result['_instructions']=d['instructions'];report['kernels'].append(result)
        except (OSError,FormatError,ValueError,KeyError,TypeError) as ex:
            gf.append(finding('AUDIT_OBJECT','incomplete',f'{h}: {ex}'))
    stored={x.name for x in (path.parent/'objects').glob('*.hsaco')}
    listed={h+'.hsaco' for h in hashes}
    if stored!=listed:
        gf.append(finding('OBJECT_SET_MISMATCH','incomplete','evidence directory and manifest object sets differ',
                          unlisted=sorted(stored-listed),missing=sorted(listed-stored)))
    if not expected: gf.append(finding('EMPTY_COVERAGE','incomplete','zero expected target kernels'))
    if expected!=manifest.get('expected_kernels'):
        gf.append(finding('MANIFEST_COUNT','incomplete','manifest kernel total is inconsistent'))
    if len(report['kernels'])!=expected:
        gf.append(finding('MISSING_KERNELS','incomplete','not every enumerated kernel reached the checker'))
    if args.expect:
        baseline=read_json(Path(args.expect))
        expected_ids=set(baseline['identities'])
        if len(expected_ids)!=len(baseline['identities']): raise FormatError('duplicate baseline identity')
        actual_ids={o['sha256']+':'+k['symbol'] for o in manifest['objects'] for k in o['kernels']}
        if expected_ids!=actual_ids:
            gf.append(finding('BASELINE_MISMATCH','incomplete','audited inventory differs from approved baseline',
                              missing=sorted(expected_ids-actual_ids),added=sorted(actual_ids-expected_ids)))
    if args.capture_log:
        report['runtime_coverage']=runtime_reconcile(Path(args.capture_log).resolve(),manifest)
        for e in report['runtime_coverage']['errors']: gf.append(finding('RUNTIME_COVERAGE','incomplete',e))
    else:
        gf.append(finding('RUNTIME_NOT_CAPTURED','info','runtime-resolved kernels not reconciled; scope is supplied build artifacts only'))
    full=sum(k['decoded_bytes']==k['expected_bytes'] and k['instruction_count']>0 for k in report['kernels'])
    report['coverage']={'objects':len(hashes),'expected_kernels':expected,'analyzed_kernels':len(report['kernels']),
                        'fully_decoded_kernels':full,'expected_bytes':expected_bytes,
                        'decoded_bytes':sum(k['decoded_bytes'] for k in report['kernels']),
                        'all_source_instantiations_proved':False,'helper_call_graph_proved':False}
    allfindings=gf+[f for k in report['kernels'] for f in k['findings']]
    report['status'],code=classify(allfindings,args.strict_review);report['exit_code']=code
    report['finding_counts']={s:sum(f['severity']==s for f in allfindings) for s in ('error','incomplete','review','info')}
    write_reports(Path(args.out),report)
    print(f"{report['status']}: {full}/{expected} kernels fully decoded; findings={report['finding_counts']}")
    return code


def baseline(args) -> int:
    m=read_json(Path(args.manifest))
    if m.get('schema')!=SCHEMA or not m.get('collection_complete'): raise FormatError('cannot freeze incomplete collection')
    out=Path(args.out)
    if out.exists(): raise FormatError('baseline exists; replace only after deliberate review')
    write_json(out,{'schema':SCHEMA,'identities':sorted(o['sha256']+':'+k['symbol'] for o in m['objects'] for k in o['kernels']),
                    'scope':m['scope'],'note':'Approval is external; creating this file does not establish independent completeness.'})
    return 0


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description='RDNA4 binary static-check gate (not full conformance certification)')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect');c.add_argument('inputs',nargs='+');c.add_argument('--out',required=True);c.set_defaults(fn=collect)
    a=sub.add_parser('audit');a.add_argument('manifest');a.add_argument('--out',required=True)
    a.add_argument('--llvm-library');a.add_argument('--objdump');a.add_argument('--timeout',type=float,default=120)
    a.add_argument('--strict-review',action='store_true',default=True,help='block unresolved review findings (default)')
    a.add_argument('--allow-review',action='store_false',dest='strict_review',help='exploratory mode only; does not override incomplete evidence')
    a.add_argument('--expect');a.add_argument('--capture-log');a.set_defaults(fn=audit)
    b=sub.add_parser('baseline');b.add_argument('manifest');b.add_argument('--out',required=True);b.set_defaults(fn=baseline)
    args=p.parse_args(argv)
    try:
        if getattr(args,'timeout',1)<=0: raise FormatError('timeout must be positive')
        return args.fn(args)
    except (OSError,ValueError,KeyError,TypeError) as ex:
        print(f'INCOMPLETE: {type(ex).__name__}: {ex}',file=sys.stderr);return 2


if __name__=='__main__': raise SystemExit(main())
