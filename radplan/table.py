"""Measured-table predictor for the gfx1201 attention rows.

Inside the measured grid the time comes from trilinear interpolation in (q, log2 ctx, b);
outside it the call is refused instead of extrapolated. This is a lookup by design: it is
the mode that can promise the accuracy the analytical model cannot.
"""
from __future__ import annotations
import json
import math
from pathlib import Path


def load(path):
    d = json.loads(Path(path).read_text())
    if d.get('schema') != 'radiance-attention-table/1':
        raise ValueError('unsupported table schema')
    grid = {}
    for r in d['records']:
        grid.setdefault(r['family'], {})[(r['q'], r['ctx'], r['b'])] = r['min_us']
    return d, grid


def _brk(vals, v):
    lo = max([x for x in vals if x <= v], default=vals[0])
    hi = min([x for x in vals if x >= v], default=vals[-1])
    return lo, hi, (0.0 if hi == lo else (math.log2(v) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)))


def predict(grid, family, q, ctx, b):
    g = grid[family]
    qs = sorted({k[0] for k in g}); cs = sorted({k[1] for k in g}); bs = sorted({k[2] for k in g})
    if not (qs[0] <= q <= qs[-1] and cs[0] <= ctx <= cs[-1] and bs[0] <= b <= bs[-1]):
        raise ValueError(f'outside the measured envelope: q={q} ctx={ctx} b={b}')
    q0, q1, tq = _brk(qs, q); c0, c1, tc = _brk(cs, ctx); b0, b1, tb = _brk(bs, b)
    num = den = 0.0
    for A, wa in ((q0, 1 - tq), (q1, tq)):
        for C, wc in ((c0, 1 - tc), (c1, tc)):
            for B, wb in ((b0, 1 - tb), (b1, tb)):
                w = wa * wc * wb
                if w == 0:
                    continue
                num += w * g[(A, C, B)]; den += w
    if den == 0:
        raise ValueError('no table node covers this geometry')
    return num / den
