"""Explicit snapshot construction. Never discover pointers by inspecting integers."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .format import SCHEMA, digest, file_hash, identifier, radiance_abi, require, validate

class Package:
    def __init__(self, root: Path, name: str, *, backend: str = 'host', provenance: str = 'synthetic',
                 reference: str = 'independent', note: str = ''):
        self.root=Path(root)
        require(not self.root.exists(),'package output already exists')
        self.root.mkdir(parents=True, mode=0o700)
        (self.root/'blobs').mkdir(mode=0o700)
        self.m={'schema':SCHEMA,'id':identifier(name),
                'provenance':{'kind':provenance,'reference':reference,'note':note},
                'target':{'backend':backend,'arch':'host' if backend=='host' else 'gfx1201','world_size':1},
                'plugins':[],'allocations':[],'views':[],'streams':[],'nodes':[],'checks':[],
                'contract':{'pointer_fields_complete':True,'hidden_state':'none','execution':'ordered-launchers'}}

    def blob(self, data: bytes) -> dict:
        require(len(data)>0, 'empty blob')
        h=digest(data); p=self.root/'blobs'/h
        if not p.exists():
            fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f: f.write(data)
        return {'path':'blobs/'+h,'sha256':h,'bytes':len(data)}

    def plugin(self, name: str, path: Path, *, id: str = 'p0') -> str:
        abi=radiance_abi()
        require(abi is not None, 'cannot read RAD_ABI_VERSION: abi/rad_abi.h not found for this suite')
        self.m['plugins'].append({'id':id,'name':name,'sha256':file_hash(Path(path)),'abi':abi})
        return id

    def allocation(self, id: str, data: bytes, *, mutable: bool = False, kind: str | None = None,
                   alignment: int = 256, relocations: list | None = None) -> str:
        raw=bytearray(data)
        for r in relocations or []:
            off=r['offset']; require(0<=off<=len(raw)-8,'pointer slot outside allocation')
            raw[off:off+8]=b'\0'*8
        self.m['allocations'].append({'id':id,'kind':kind or ('host' if self.m['target']['backend']=='host' else 'device'),
            'bytes':len(raw),'alignment':alignment,'initial':self.blob(bytes(raw)),
            'mutable':mutable,'relocations':relocations or []})
        return id

    def view(self, id: str, allocation: str, *, nbytes: int, dtype: int, shape: list[int],
             offset: int = 0, stride: list[int] | None = None) -> str:
        if stride is None:
            stride=[]; acc=1
            for x in reversed(shape): stride.insert(0,acc); acc*=x
        self.m['views'].append({'id':id,'allocation':allocation,'offset':offset,'bytes':nbytes,
                               'dtype':dtype,'shape':shape,'stride':stride})
        return id

    def stream(self, id: str = 'q0', priority: int = 0) -> str:
        self.m['streams'].append({'id':id,'priority':priority}); return id

    def kernel(self, id: str, plugin: str, row: str, op: str, tensors: list[str | None],
               accesses: list[tuple[str,str]], *, params: dict | None = None, stream: str = 'q0',
               phase: str = 'measure', scratch_bytes: int = 0, scratch_policy: str | None = None) -> None:
        ps=[]
        for key,value in (params or {}).items():
            require(type(value) in (int,str,float),'parameters are int/str/float, not bool/objects')
            ps.append({'key':key,'kind':{int:'int',str:'str',float:'f64'}[type(value)],'value':value})
        self.m['nodes'].append({'id':id,'kind':'kernel','stream':stream,'phase':phase,
            'plugin':plugin,'row':row,'op':op,'tensors':tensors,
            'accesses':[{'view':v,'mode':mode} for v,mode in accesses], 'params':ps,'scratch_bytes':scratch_bytes})
        if scratch_policy is not None:self.m['nodes'][-1]['scratch_policy']=scratch_policy

    def node(self, id: str, kind: str, *, stream: str = 'q0', phase: str = 'measure', **args) -> None:
        self.m['nodes'].append({'id':id,'kind':kind,'stream':stream,'phase':phase,**args})

    def check(self, view: str, expected: bytes, *, mode: str = 'bytes', atol: float = 0,
              rtol: float = 0, rel_l2: float = 0, nonfinite: str = 'reject') -> None:
        self.m['checks'].append({'view':view,'expected':self.blob(expected),'mode':mode,
            'atol':atol,'rtol':rtol,'rel_l2':rel_l2,'nonfinite':nonfinite})

    def checkpoint(self, after: str, view: str, expected: bytes, **kwargs) -> None:
        """Independent or captured intermediate oracle; executed only in diagnostics."""
        self.check(view, expected, **kwargs)
        c=self.m['checks'].pop()
        groups=self.m.setdefault('checkpoints',[])
        group=next((g for g in groups if g['after']==after),None)
        if group is None:
            group={'after':after,'checks':[]};groups.append(group)
        group['checks'].append(c)

    def finish(self, *, budget: int = 2*1024**3) -> Path:
        path=self.root/'manifest.json'
        require(not path.exists(),'package already finalized')
        path.write_text(json.dumps(self.m,indent=2,allow_nan=False)+'\n'); path.chmod(0o600)
        try: validate(path,budget=budget)
        except Exception:
            path.rename(self.root/'INVALID-manifest.json'); raise
        return path
