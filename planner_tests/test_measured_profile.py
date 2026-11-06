"""The measured profile must actually reproduce the kernel timings it was fitted to.

This is a regression gate on prediction quality, not a hardware test: the records
are real gfx1201 medians taken once and archived in
`planner/examples/v06/measured-attention-r9700.json`, so the check runs without a
GPU. If the attention model or the profile moves, the error bound fails.

Bounds are deliberately loose (the model is a declared hypothesis, not a simulator):
median relative error 30 %, worst case 100 %. The shipped assumed profile fails this
by design, which is why it must not be mistaken for a calibrated one.
"""
import json
import statistics
import unittest
from pathlib import Path

from radplan.operators import op_plan

EX = Path(__file__).resolve().parents[1] / 'planner' / 'examples' / 'v06'


def load(name):
    return json.loads((EX / name).read_text())


def workloads():
    m = load('measured-attention-r9700.json')
    g = m['geometry']
    return {'schema': 'radiance-operator-workloads/1', 'workloads': [
        {'id': r['id'], 'op': 'attention', 'contract': 'bf16-q-fp8-kv-selected-causal',
         'device': 'gpu0', 'query_lengths': r['query_lengths'], 'sequence_ids': r['sequence_ids'],
         'selected_blocks': None, 'q_heads': g['n_head'], 'kv_heads': g['kv_heads'],
         'head_dim': g['head_dim'], 'block_size': g['block_size'], 'q_bits': 16, 'kv_bits': 8,
         'out_bits': 16, 'kv_scale_bytes_per_block': 0, 'family': r['family']}
        for r in m['records'][::3]]}  # subset keeps the test fast; the geometry laws are the same


def errors(profile_name, family=None):
    m = load('measured-attention-r9700.json')
    plan = op_plan(load('attention-r9700-catalog.json'), workloads(), load(profile_name))
    by_id = {w['workload_id']: w for w in plan['workloads']}
    out = []
    for r in m['records'][::3]:
        if family and r['family'] != family:
            continue
        ranked = by_id[r['id']]['ranked']
        assert ranked, f"no candidate survived for {r['id']}"
        predicted = min(x['scenario_us'] for x in ranked)
        out.append(abs(predicted - r['min_us']) / r['min_us'])
    return out


class MeasuredProfileTests(unittest.TestCase):
    def test_records_are_plausible(self):
        m = load('measured-attention-r9700.json')
        self.assertGreaterEqual(len(m['records']), 20)
        for r in m['records']:
            self.assertGreater(r['median_us'], 0)
            nseq = r.get('nseq', 1)
            self.assertEqual(len(r['query_lengths']), r['q_len'] * nseq)
            self.assertEqual(len(r['sequence_ids']), len(r['query_lengths']))
            self.assertEqual(r['query_lengths'][-1], r['seqused'])
            self.assertEqual(len(set(r['sequence_ids'])), nseq)

    def test_measured_profile_predicts_the_measured_times(self):
        e = errors('dual-r9700-measured.json')
        self.assertLessEqual(statistics.median(e), 0.10, f'median error {statistics.median(e):.2f}')
        self.assertLessEqual(max(e), 0.30, f'worst error {max(e):.2f}')

    def test_per_family_rates_predict_their_own_family(self):
        for family, ceiling in (('attn_decode', 0.08), ('attn_prefill', 0.05)):
            e = errors('dual-r9700-measured.json', family)
            self.assertLessEqual(statistics.median(e), ceiling,
                                 f'{family}: median error {statistics.median(e):.2f}')
            self.assertLessEqual(max(e), 0.30, f'{family}: worst error {max(e):.2f}')

    def test_unknown_family_is_refused(self):
        m = load('measured-attention-r9700.json')
        wl = workloads()
        wl['workloads'][0] = dict(wl['workloads'][0], family='attn_unmeasured')
        plan = op_plan(load('attention-r9700-catalog.json'), wl, load('dual-r9700-measured.json'))
        row = next(w for w in plan['workloads'] if w['workload_id'] == m['records'][0]['id'])
        self.assertEqual(row['ranked'], [])
        self.assertIn('no measured rates for operator family', row['rejected'][0]['reason'])

    def test_assumed_profile_is_not_a_predictor(self):
        # Guards against someone "calibrating" prose instead of measurements: the
        # assumed profile must keep failing badly, otherwise the measured one is noise.
        e = errors('dual-r9700-unmeasured.json')
        self.assertGreaterEqual(statistics.median(e), 0.50, f'median error {statistics.median(e):.2f}')

    def test_profiles_differ_in_provenance_and_rates(self):
        a = load('dual-r9700-unmeasured.json')['devices'][0]
        m = load('dual-r9700-measured.json')['devices'][0]
        self.assertEqual(a['provenance'], 'assumed')
        self.assertEqual(m['provenance'], 'external-measured-profile')
        self.assertNotEqual(a['launch_us'], m['launch_us'])
        self.assertNotEqual(a['exp_ops_per_us'], m['exp_ops_per_us'])


if __name__ == '__main__':
    unittest.main()
