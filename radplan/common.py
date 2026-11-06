"""Strict, small JSON inputs and explicit evidence identities."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

class PlanError(ValueError):
    pass

def need(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)

def number(value: Any, name: str, low: float = 0, *, positive: bool = False) -> float:
    need(type(value) in (float, int) and math.isfinite(value), f'{name}: finite number required')
    need(value > low if positive else value >= low, f'{name}: out of range')
    return float(value)

def integer(value: Any, name: str, low: int = 1, high: int = 2**31-1) -> int:
    need(type(value) is int and low <= value <= high, f'{name}: integer in [{low}, {high}] required')
    return value

def fields(d: Any, required: set[str], optional: set[str], name: str) -> None:
    need(isinstance(d, dict), f'{name}: object required')
    need(required <= set(d), f'{name}: missing {sorted(required-set(d))}')
    need(set(d) <= required | optional, f'{name}: unknown {sorted(set(d)-(required|optional))}')

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def hash_string(s: Any, name: str) -> str:
    need(isinstance(s, str) and len(s)==64 and all(c in '0123456789abcdef' for c in s), f'{name}: SHA256 required')
    return s

def read(path: str | Path) -> Any:
    p=Path(path)
    need(p.is_file() and p.stat().st_size<=64*1024*1024, 'JSON missing or over 64 MiB')
    def pairs(kvs):
        d={}
        for k,v in kvs:
            need(k not in d, f'duplicate JSON key: {k}'); d[k]=v
        return d
    def constant(s): raise PlanError('nonfinite JSON constant: '+s)
    try: return json.loads(p.read_text(), object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeError, RecursionError) as e: raise PlanError(str(e)) from e

def write(path: str | Path, value: Any) -> None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')

def new_dir(path: str | Path) -> Path:
    p=Path(path).resolve(); need(not p.exists(), 'output exists; use a fresh directory'); p.mkdir(parents=True)
    return p

def ceildiv(a: int, b: int) -> int:
    return (a+b-1)//b
