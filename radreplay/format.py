"""Validate a portable replay package *before* loading any executable code.

This is a declared memory/dataflow contract, not pointer discovery, a sandbox,
or a proof that a plugin obeys its access annotations.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA = 'radiance-kernel-replay/1'

def radiance_abi() -> int | None:
    """RAD_ABI_VERSION of the checkout this suite lives in, or None when out-of-tree.

    The ABI moves with the Radiance tree, so it is read from its header instead of
    being frozen here: a stale constant rejects every current plugin at the worker.
    """
    roots = [Path(__file__).resolve().parents[3]]
    if os.environ.get('RADIANCE_SOURCE'):
        roots.insert(0, Path(os.environ['RADIANCE_SOURCE']))
    for root in roots:
        try:
            found = re.search(r'#define\s+RAD_ABI_VERSION\s+(\d+)u?', (root/'abi'/'rad_abi.h').read_text())
        except OSError:
            continue
        if found:
            return int(found.group(1))
    return None
MAX_JSON = 16 * 1024 * 1024
MAX_NODES = 2048
MAX_I64 = (1 << 63) - 1
ID = re.compile(r'[A-Za-z0-9_.-]{1,128}\Z')
HASH = re.compile(r'[a-f0-9]{64}\Z')
# Stable RAD_ABI_VERSION 30 vocabulary. Plugin-private dtypes require explicit spans.
BITS = {1:32, 2:16, 3:16, 4:8, 5:8, 8:8, 9:8, 10:16, 11:32,
        12:64, 13:32, 14:8, 16:4, 17:2, 18:4, 19:4, 20:2, 21:8,
        22:1, 23:3, 24:3, 25:5, 26:5, 27:6, 28:6, 29:7, 30:7,
        31:9, 32:10, 33:11, 34:12, 35:6, 36:6, 37:16}

class ReplayError(ValueError):
    """Incomplete or unsupported fixture; must never become a runtime pass."""

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReplayError(message)

def keys(obj: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    require(isinstance(obj, dict), 'expected an object')
    require(required <= obj.keys(), f'missing keys: {sorted(required - obj.keys())}')
    require(obj.keys() <= required | optional, f'unknown keys: {sorted(obj.keys() - required - optional)}')

def integer(x: Any, lo: int = 0, hi: int = MAX_I64) -> int:
    require(type(x) is int and lo <= x <= hi, f'integer outside [{lo}, {hi}]: {x!r}')
    return x

def identifier(x: Any) -> str:
    require(isinstance(x, str) and ID.fullmatch(x) is not None and x not in ('.','..'), 'invalid identifier')
    return x

def text(x: Any) -> str:
    require(isinstance(x, str) and '\x00' not in x and len(x) <= 16384, 'invalid text')
    return x

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def json_pairs(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, f'duplicate JSON key: {k}')
        result[k] = v
    return result

def parse_json(raw: bytes) -> dict:
    require(len(raw) <= MAX_JSON, 'JSON exceeds 16 MiB')
    def nonfinite(value):
        raise ReplayError(f'non-finite JSON number: {value}')
    return json.loads(raw, object_pairs_hook=json_pairs, parse_constant=nonfinite)

def read_json(path: Path) -> dict:
    require(path.stat().st_size <= MAX_JSON, 'JSON exceeds 16 MiB')
    return parse_json(path.read_bytes())

def verified_json(path: Path, expected_hash: str) -> dict:
    require(path.stat().st_size <= MAX_JSON, 'JSON exceeds 16 MiB')
    raw=path.read_bytes()
    require(digest(raw)==expected_hash, 'JSON changed after validation')
    return parse_json(raw)

def safe_file(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative and '\x00' not in relative, 'invalid file path')
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'blob path is not confined')
    q = root
    for part in p.parts:
        q = q / part
        require(not q.is_symlink(), 'symlink inside replay package')
    require(q.resolve().is_relative_to(root.resolve()), 'blob escapes replay package')
    require(stat.S_ISREG(q.stat().st_mode), 'blob is not a regular file')
    return q

def blob(root: Path, item: dict, budget: int, verify: bool = True) -> Path:
    keys(item, {'path','sha256','bytes'})
    n = integer(item['bytes'], 1, budget)
    require(isinstance(item['sha256'], str) and HASH.fullmatch(item['sha256']) is not None, 'invalid blob hash')
    p = safe_file(root, item['path'])
    require(p.stat().st_size == n, 'blob size mismatch')
    if verify:
        require(file_hash(p) == item['sha256'], 'blob hash mismatch')
    return p

def unique(items: Any, what: str) -> dict:
    require(isinstance(items, list) and len(items) <= 16384, f'invalid {what} list')
    out = {}
    for x in items:
        require(isinstance(x, dict), f'invalid {what} item')
        i = identifier(x.get('id'))
        require(i not in out, f'duplicate {what} id: {i}')
        out[i] = x
    return out

def checked_span(view: dict) -> int:
    shape, stride = view['shape'], view['stride']
    require(isinstance(shape,list) and isinstance(stride,list) and 1 <= len(shape) <= 6 and len(shape)==len(stride), 'invalid tensor rank')
    for n in shape: integer(n,1)
    for n in stride: integer(n,0)
    elements = 1 + sum((n-1)*s for n,s in zip(shape,stride))
    require(elements <= MAX_I64 and math.prod(shape) <= MAX_I64, 'tensor index overflow')
    dtype = integer(view['dtype'],1,(1<<32)-1)
    if dtype not in BITS:
        require(dtype >= 256, 'unknown standard dtype; do not guess its byte size')
        return view['bytes']
    b = BITS[dtype]
    if b < 8 or b % 8:
        # Radiance sub-byte storage is row-byte-padded, not one continuous bitstream.
        require(stride[-1] == 1, 'sub-byte last stride must be one')
        row_bytes = (shape[-1]*b+7)//8
        require(len(shape)==1 or all(s % shape[-1] == 0 for s in stride[:-1]), 'unsupported sub-byte row stride')
        rows = sum((n-1)*(s//shape[-1]) for n,s in zip(shape[:-1],stride[:-1]))
        return (rows+1)*row_bytes
    return elements*(b//8)

def contiguous(v: dict) -> bool:
    step = 1
    for n,s in zip(reversed(v['shape']),reversed(v['stride'])):
        if n != 1 and s != step: return False
        step *= n
    return True

def validate(path: Path, *, budget: int = 2*1024**3, verify_blobs: bool = True) -> dict:
    """Return inventory/coverage without executing plugins or touching a GPU."""
    integer(budget,1)
    path = path.resolve()
    require(path.stat().st_size <= MAX_JSON, "JSON exceeds 16 MiB")
    raw = path.read_bytes(); manifest_hash = digest(raw)
    m = parse_json(raw); root = path.parent
    keys(m, {'schema','id','provenance','target','plugins','allocations','views','streams','nodes','checks','contract'}, {'lineage', 'checkpoints'})
    require(m['schema']==SCHEMA, 'unsupported replay schema')
    identifier(m['id'])
    if 'lineage' in m:
        l=m['lineage'];keys(l,{'purpose','parent_id','parent_manifest_sha256','plugin_changes'})
        require(l['purpose']=='explicit-candidate-rebind','unknown lineage purpose');identifier(l['parent_id'])
        require(isinstance(l['parent_manifest_sha256'],str) and HASH.fullmatch(l['parent_manifest_sha256']),'invalid parent hash')
        require(isinstance(l['plugin_changes'],list) and l['plugin_changes'],'empty candidate changes')
        ids=set()
        for c in l['plugin_changes']:
            keys(c,{'id','before','after'});identifier(c['id']);require(c['id'] not in ids,'duplicate candidate change');ids.add(c['id'])
            require(all(isinstance(c[x],str) and HASH.fullmatch(c[x]) for x in ('before','after')) and c['before']!=c['after'],'invalid candidate hash change')
            require(any(p['id']==c['id'] and p['sha256']==c['after'] for p in m['plugins']),'lineage differs from candidate plugin')
    keys(m['provenance'], {'kind','reference','note'})
    require(m['provenance']['kind'] in ('synthetic','kbench-seeded','kbench-model-weights','production-snapshot'), 'unknown input provenance')
    require(m['provenance']['reference'] in ('captured-baseline','independent'), 'unknown reference provenance')
    text(m['provenance']['note'])
    keys(m['target'], {'backend','arch','world_size'})
    require(m['target']['world_size']==1 and type(m['target']['world_size']) is int, 'TP/P2P is not supported by replay v0.4')
    require((m['target']['backend'],m['target']['arch']) in (('host','host'),('aql','gfx1201')), 'unsupported backend/architecture')
    gpu = m['target']['backend']=='aql'
    keys(m['contract'], {'pointer_fields_complete','hidden_state','execution'})
    require(m['contract']['pointer_fields_complete'] is True, 'pointer inventory must be explicitly complete')
    require(m['contract']['hidden_state']=='none', 'hidden instance/global state is not replayable in v0.4')
    require(m['contract']['execution']=='ordered-launchers', 'raw AQL/chain/tape replay is not supported in v0.3')
    plugins = unique(m['plugins'],'plugin')
    require(plugins, 'no plugins')
    for p in plugins.values():
        keys(p, {'id','name','sha256','abi'})
        text(p['name']); require(p['name'], 'empty plugin name')
        require(isinstance(p['sha256'], str) and HASH.fullmatch(p['sha256']) is not None, 'invalid plugin hash')
        expected_abi = radiance_abi()
        require(type(p['abi']) is int and p['abi']>=26, 'invalid plugin ABI version')
        require(expected_abi is None or p['abi']==expected_abi, f'expected Radiance ABI {expected_abi}')
    allocs = unique(m['allocations'],'allocation'); require(allocs,'no allocations')
    total = 0; total_blobs = 0
    for a in allocs.values():
        keys(a, {'id','kind','bytes','alignment','initial','mutable','relocations'})
        total += integer(a['bytes'],1,budget)
        require(total <= budget, 'allocation budget exceeded')
        align = integer(a['alignment'],8,65536)
        require(align & (align-1)==0, 'alignment is not a power of two')
        require(a['kind'] in (('device','host_pinned') if gpu else ('host',)), 'unsupported allocation kind for backend')
        require(type(a['mutable']) is bool, 'mutable must be boolean')
        blob(root,a['initial'],budget,verify_blobs)
        require(a['initial']['bytes']==a['bytes'], 'initial snapshot must cover entire allocation')
        total_blobs += a['bytes']
        require(isinstance(a['relocations'],list), 'invalid relocation list')
        slots=set()
        for r in a['relocations']:
            keys(r, {'offset','target','addend'})
            off=integer(r['offset'],0,a['bytes']-8)
            require(off%8==0 and off not in slots, 'overlapping/unaligned pointer slot')
            slots.add(off)
            require(r['target'] in allocs, 'unresolved relocation target')
            integer(r['addend'],0,allocs[r['target']]['bytes']-1)
            require(not a['mutable'], 'mutable pointer tables are unsupported')
        if slots:
            with safe_file(root,a['initial']['path']).open('rb') as f:
                for off in slots:
                    f.seek(off); require(f.read(8)==b'\0'*8, 'portable pointer slots must be zero, not captured addresses')
    views=unique(m['views'],'view'); require(views,'no views')
    for v in views.values():
        keys(v, {'id','allocation','offset','bytes','dtype','shape','stride'})
        require(v['allocation'] in allocs, 'unresolved view allocation')
        integer(v['offset']); integer(v['bytes'],1,budget)
        require(v['offset']+v['bytes']<=allocs[v['allocation']]['bytes'], 'view exceeds allocation')
        require(checked_span(v)<=v['bytes'], 'tensor exceeds declared view span')
    streams=unique(m['streams'],'stream'); require(streams,'no streams')
    require(len(streams)<=32, 'too many streams')
    for s in streams.values():
        keys(s, {'id','priority'}); integer(s['priority'],0,1)
    require(gpu or len(streams)==1, 'host replay supports one stream')
    nodes=unique(m['nodes'],'node'); require(0<len(nodes)<=MAX_NODES,'invalid node count')
    history={}; events={}; ancestors={}; accesses={}; measured=False; kernels=[]; used_plugins=set()
    for n in nodes.values():
        base={'id','kind','stream','phase'}
        require(n.get('stream') in streams, 'unresolved stream')
        require(n.get('phase') in ('prefix','measure'), 'invalid phase')
        if n['phase']=='measure': measured=True
        else: require(not measured, 'prefix nodes must precede measurement')
        kind=n['kind']; ni=n['id']; deps=set()
        if n['stream'] in history: deps.add(history[n['stream']])
        if kind=='kernel':
            keys(n, base|{'plugin','row','op','params','tensors','accesses','scratch_bytes'}, {'scratch_policy'})
            require(n['plugin'] in plugins, 'unresolved plugin')
            used_plugins.add(n['plugin'])
            text(n['row']); text(n['op']); require(n['row'] and n['op'], 'empty row/op')
            require(n['op'] not in ('all_reduce','all_gather') and 'allreduce' not in n['row'].lower(), 'collective requires a multi-rank runner')
            require(isinstance(n['params'],list) and len(n['params'])<=256, 'invalid params')
            pk=set()
            for p in n['params']:
                keys(p, {'key','kind','value'}); k=text(p['key']); require(k and k not in pk,'duplicate/empty parameter'); pk.add(k)
                if p['kind']=='int': integer(p['value'],-MAX_I64-1,MAX_I64)
                elif p['kind']=='str': text(p['value'])
                elif p['kind']=='f64': require(type(p['value']) in (float,int) and math.isfinite(p['value']), 'nonfinite parameter')
                else: raise ReplayError('parameter must be resolved int/str/f64')
                require(k!='chain' or p['value']==0, 'device-side chaining is not certified by v0.3')
            integer(n['scratch_bytes'],0,budget)
            if 'scratch_policy' in n:
                require(n['scratch_policy'] in ('zeroed','kernel-initialized','unspecified'), 'invalid scratch policy')
            total += n['scratch_bytes']; require(total<=budget,'scratch exceeds total memory budget')
            require(isinstance(n['tensors'],list) and 0<len(n['tensors'])<=128, 'invalid tensor list')
            require(isinstance(n['accesses'],list) and n['accesses'], 'missing access contract')
            for v in n['tensors']: require(v is None or v in views, 'unresolved tensor view')
            kernels.append(ni)
        elif kind=='copy': keys(n,base|{'src','dst','bytes'})
        elif kind=='fill': keys(n,base|{'dst','value','bytes'}); integer(n['value'],0,255)
        elif kind=='record':
            keys(n,base|{'event'}); identifier(n['event']); require(n['event'] not in events,'event may be recorded only once per iteration'); events[n['event']]=ni
        elif kind=='wait':
            keys(n,base|{'event'}); require(n['event'] in events,'wait has no earlier producer inside window'); deps.add(events[n['event']])
        else: raise ReplayError(f'unsupported node kind: {kind}')
        ancestor=set(deps)
        for d in deps: ancestor.update(ancestors[d])
        ancestors[ni]=ancestor; history[n['stream']]=ni
        acc=[]
        if kind=='kernel':
            seen=set()
            for x in n['accesses']:
                keys(x, {'view','mode'}); require(x['view'] in views,'unknown access view'); require(x['mode'] in ('r','w','rw'),'invalid access mode')
                require(x['view'] not in seen,'duplicate access view'); seen.add(x['view']); acc.append((x['view'],x['mode']))
            require(all(v is None or v in seen for v in n['tensors']), 'tensor omitted from access contract')
            # Every allocation reachable through pointer tables must have a declared access.
            reached={views[v]['allocation'] for v in n['tensors'] if v is not None}; todo=list(reached)
            while todo:
                a=todo.pop()
                for r in allocs[a]['relocations']:
                    if r['target'] not in reached: reached.add(r['target']); todo.append(r['target'])
            described={views[v]['allocation'] for v in seen}
            require(reached <= described,'nested pointer target missing from access contract')
        elif kind in ('copy','fill'):
            require(n['dst'] in views,'unknown destination'); integer(n['bytes'],1,views[n['dst']]['bytes'])
            acc.append((n['dst'],'w'))
            if kind=='copy':
                require(n['src'] in views,'unknown copy source'); require(n['bytes']<=views[n['src']]['bytes'],'copy source too small'); acc.append((n['src'],'r'))
                a,b=views[n['src']],views[n['dst']]
                require(a['allocation']!=b['allocation'] or max(a['offset'],b['offset'])>=min(a['offset'],b['offset'])+n['bytes'],'overlapping memcpy is unsupported')
        for v,mode in acc:
            if 'w' in mode: require(allocs[views[v]['allocation']]['mutable'],'node writes immutable allocation')
        accesses[ni]=acc
        for old,aa in accesses.items():
            if old==ni or old in ancestor: continue
            for va,ma in aa:
                for vb,mb in acc:
                    a,b=views[va],views[vb]
                    overlap=a['allocation']==b['allocation'] and max(a['offset'],b['offset'])<min(a['offset']+a['bytes'],b['offset']+b['bytes'])
                    require(not overlap or ('w' not in ma and 'w' not in mb),f'unordered cross-stream conflict: {old} -> {ni}')
    require(measured and kernels, 'no measured work or no kernel')
    require(any(nodes[k]['phase']=='measure' for k in kernels), 'no kernel in measured window')
    require(used_plugins==plugins.keys(),'unused or missing plugin inventory')
    checks=m['checks']; require(isinstance(checks,list) and checks,'no output checks')
    checkpoints=m.get('checkpoints',[])
    require(isinstance(checkpoints,list) and len(checkpoints)<=MAX_NODES,'invalid checkpoints')
    checkpoint_ids=set()
    for cp in checkpoints:
        keys(cp,{'after','checks'})
        require(cp['after'] in nodes and cp['after'] not in checkpoint_ids,'invalid/duplicate checkpoint node')
        checkpoint_ids.add(cp['after'])
        require(isinstance(cp['checks'],list) and cp['checks'],'empty checkpoint')
    extra_checks=[c for cp in checkpoints for c in cp['checks']]
    for cp in checkpoints:
        require(len({c.get('view') for c in cp['checks']})==len(cp['checks']),'duplicate checkpoint view')
    covered=set(); checkviews=set()
    for index,c in enumerate(checks+extra_checks):
        keys(c, {'view','expected','mode','atol','rtol','rel_l2','nonfinite'})
        require(c['view'] in views,'unknown checked view')
        if index<len(checks):
            require(c['view'] not in checkviews,'unknown/duplicate checked view');checkviews.add(c['view'])
        v=views[c['view']]
        if index<len(checks):covered.add(v['allocation'])
        blob(root,c['expected'],budget,verify_blobs); total_blobs+=c['expected']['bytes']
        require(c['expected']['bytes']==v['bytes'],'expected bytes differ from view')
        require(c['mode'] in ('bytes','float'),'unsupported comparison mode')
        for key in ('atol','rtol','rel_l2'):
            x=c[key]; require(type(x) in (float,int) and math.isfinite(x) and 0<=x<=1e6, 'invalid tolerance')
        require(c['nonfinite'] in ('reject','match'),'invalid nonfinite policy')
        if c['mode']=='bytes': require(c['atol']==c['rtol']==c['rel_l2']==0,'byte comparison cannot use tolerance')
        else: require(v['dtype'] in (1,2,3) and contiguous(v) and checked_span(v)==v['bytes'],'float check needs a packed f32/f16/bf16 view')
        a=allocs[v['allocation']]
        require(not any(v['offset'] < r['offset']+8 and r['offset']<v['offset']+v['bytes'] for r in a['relocations']),'cannot compare relocated pointer bytes')
    written={views[v]['allocation'] for ac in accesses.values() for v,mode in ac if 'w' in mode}
    require(written <= covered,'written allocation has no output check')
    for ac in accesses.values():
        for vid,mode in ac:
            if 'w' not in mode: continue
            v=views[vid]; lo=v['offset']; hi=lo+v['bytes']
            intervals=sorted((views[c['view']]['offset'],views[c['view']]['offset']+views[c['view']]['bytes'])
                             for c in checks if views[c['view']]['allocation']==v['allocation'])
            for start,end in intervals:
                if start<=lo: lo=max(lo,end)
            require(lo>=hi,'written view not completely covered by expected-output checks')
    require(total_blobs<=3*budget,'snapshot blob budget exceeded')
    return {'schema':SCHEMA,'status':'FIXTURE_VALID','manifest_sha256':manifest_hash,
            'id':m['id'],'allocation_bytes_including_scratch':total,'blob_bytes':total_blobs,
            'kernel_nodes':kernels,'node_count':len(nodes),'stream_count':len(streams),
            'kernel_rows':sorted({plugins[nodes[k]['plugin']]['name']+':'+nodes[k]['row'] for k in kernels}),
            'provenance':m['provenance'],'runtime_executed':False,'all_radiance_kernels_covered':False}


def all_blobs(m: dict) -> list[dict]:
    """All byte evidence, including explicit intermediate reference checkpoints."""
    return ([a['initial'] for a in m['allocations']] + [c['expected'] for c in m['checks']]
            + [c['expected'] for cp in m.get('checkpoints',[]) for c in cp['checks']])
