"""The measured table must keep the accuracy it was built for.

Each interior axis level is held out and interpolated from its neighbours — that is the
error a caller sees for a geometry between two measured ones. Bounds are the measured
values with a small margin; the analytical model in dual-r9700-measured.json is the
fallback outside the grid, not this table.
"""
import json
import math
import statistics
import unittest
from pathlib import Path

from radplan.table import load, predict

EX = Path(__file__).resolve().parents[1] / 'planner' / 'examples' / 'v06'


class AttentionTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc, cls.grid = load(EX / 'attention-table-r9700.json')

    def axis_errors(self, family, axis):
        g = self.grid[family]
        levels = sorted({k[axis] for k in g if axis != 2 or k[2] <= 8})
        errs = []
        for i in range(1, len(levels) - 1):
            v, lo, hi = levels[i], levels[i - 1], levels[i + 1]
            for k, t in g.items():
                if k[axis] != v or (axis == 2 and k[2] > 8):
                    continue
                kl, kh = list(k), list(k)
                kl[axis], kh[axis] = lo, hi
                if tuple(kl) not in g or tuple(kh) not in g:
                    continue
                f = (math.log2(v) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
                errs.append(abs(g[tuple(kl)] * (1 - f) + g[tuple(kh)] * f - t) / t)
        return errs

    def test_table_has_the_measured_grid(self):
        self.assertGreaterEqual(len(self.doc['records']), 200)
        self.assertEqual(self.doc['clock_control']['sclk'], '1634 MHz')

    def test_interpolation_holds_across_context(self):
        for fam in ('attn_decode', 'attn_prefill'):
            e = self.axis_errors(fam, 1)
            self.assertLessEqual(statistics.median(e), 0.07, f'{fam} ctx median {statistics.median(e):.3f}')
            self.assertLessEqual(max(e), 0.15, f'{fam} ctx max {max(e):.3f}')

    def test_dense_envelope_is_declared(self):
        env = self.doc['validated_envelope']
        self.assertEqual(env['b'], [1, 8])
        self.assertIn(32, self.doc['axes']['b'])   # measured, but outside the validated envelope

    def test_interpolation_holds_across_batch(self):
        for fam in ('attn_decode', 'attn_prefill'):
            e = self.axis_errors(fam, 2)
            self.assertLessEqual(statistics.median(e), 0.12, f'{fam} b median {statistics.median(e):.3f}')
            self.assertLessEqual(max(e), 0.26, f'{fam} b max {max(e):.3f}')

    def test_outside_the_envelope_is_refused(self):
        with self.assertRaises(ValueError):
            predict(self.grid, 'attn_decode', 1, 512, 1)
        with self.assertRaises(ValueError):
            predict(self.grid, "attn_decode", 1, 16384, 40)

    def test_grid_point_is_exact(self):
        g = self.grid['attn_decode']
        k = sorted(g)[0]
        self.assertAlmostEqual(predict(self.grid, 'attn_decode', *k), g[k], places=9)


if __name__ == '__main__':
    unittest.main()
