import unittest,copy
from radplan.common import PlanError
from radplan.operators import build,op_plan
from support_v06 import *

class OperatorTests(unittest.TestCase):
    def test_attention_tail_count(self):
        w=wop('attention');w.update(query_lengths=[9],sequence_ids=[0],selected_blocks=[[0,2]])
        r=build(w,cop('attention'),system());self.assertEqual(r['counts']['selected_tokens_per_query'],[5])
        self.assertEqual(r['counts']['useful_matrix_ops'],4*256*12*5)
    def test_attention_no_quadratic_allocation(self):
        r=build(wop('attention'),cop('attention'),system());self.assertFalse(r['counts']['quadratic_score_matrix_materialized'])
    def test_attention_splits_clamped_to_nonempty_tiles(self):
        w=wop('attention');w.update(query_lengths=[2],sequence_ids=[0],selected_blocks=None)
        r=build(w,cop('attention',split_kv=16),system());self.assertEqual(r['counts']['effective_splits_per_query'],[1])
    def test_attention_partial_dtype_accounted(self):
        a=build(wop('attention'),cop('attention',partial_bits=16),system());b=build(wop('attention'),cop('attention',partial_bits=32),system())
        self.assertGreater(b['counts']['split_partial_bytes'],a['counts']['split_partial_bytes'])
    def test_attention_gqa_reuse_saves_traffic(self):
        a=build(wop('attention'),cop('attention'),system());b=build(wop('attention'),cop('attention',gqa_reuse=False),system())
        self.assertEqual(b['counts']['kv_requested_bytes'],12*a['counts']['kv_requested_bytes'])
    def test_attention_invalid_future_block(self):
        w=wop('attention');w.update(query_lengths=[4],sequence_ids=[0],selected_blocks=[[1]])
        with self.assertRaises(PlanError):build(w,cop('attention'),system())
    def test_attention_duplicate_block(self):
        w=wop('attention');w['selected_blocks'][0]=[0,0]
        with self.assertRaises(PlanError):build(w,cop('attention'),system())
    def test_attention_nonmonotone_sequence(self):
        w=wop('attention');w['query_lengths'][1]=w['query_lengths'][0]
        with self.assertRaises(PlanError):build(w,cop('attention'),system())
    def test_attention_precision_mismatch(self):
        w=wop('attention');w['kv_bits']=16
        with self.assertRaises(PlanError):build(w,cop('attention'),system())
    def test_attention_contract_mismatch(self):
        w=wop('attention');w['contract']='other'
        with self.assertRaises(PlanError):build(w,cop('attention'),system())
    def test_attention_dense_prefill_reuses_causal_union(self):
        w=wop('attention');w.update(query_lengths=[64,65,66,67,68],selected_blocks=None)
        a=build(w,cop('attention',query_tile=32,kv_tile=48,force_merge=False),system())
        b=build(w,cop('attention',query_tile=1,kv_tile=48,force_merge=False),system())
        self.assertLess(a['counts']['kv_requested_bytes'],b['counts']['kv_requested_bytes'])
        self.assertEqual(a['counts']['useful_matrix_ops'],b['counts']['useful_matrix_ops'])
    def test_attention_sparse_lists_not_reused_by_batch(self):
        w=wop('attention');w.update(query_lengths=[16,17],sequence_ids=[0,0],selected_blocks=[[0],[1]])
        r=build(w,cop('attention',query_tile=32),system());self.assertEqual(r['counts']['kv_requested_bytes'],2*256*8)
    def test_attention_rejects_unsupported_query_width(self):
        w=wop('attention');w['query_lengths']=list(range(10,16));w['sequence_ids']=[0]*6;w['selected_blocks']=None
        with self.assertRaises(PlanError):build(w,cop('attention'),system())
    def test_qsa_scores_full_history(self):
        w=wop('qsa_indexer');r=build(w,cop('qsa_indexer'),system())
        self.assertEqual(r['counts']['score_matrix_elements'],sum(n//4 for n in w['query_lengths']))
        self.assertGreater(r['counts']['score_matrix_elements'],512*len(w['query_lengths']))
    def test_qsa_dense_skip_with_update(self):
        w=wop('qsa_indexer');w.update(query_lengths=[8],sequence_ids=[0],skip_score_when_dense=True,updated_blocks=1)
        r=build(w,cop('qsa_indexer'),system());self.assertEqual(r['counts']['scored_queries'],0);self.assertGreater(r['scenario_us'],0)
    def test_qsa_noop(self):
        w=wop('qsa_indexer');w.update(query_lengths=[8],sequence_ids=[0],skip_score_when_dense=True,updated_blocks=0)
        self.assertEqual(build(w,cop('qsa_indexer'),system())['scenario_us'],0)
    def test_qsa_tail_does_not_add_complete_score(self):
        w=wop('qsa_indexer');w.update(query_lengths=[9],sequence_ids=[0],skip_score_when_dense=False,updated_blocks=0)
        r=build(w,cop('qsa_indexer'),system());self.assertEqual(r['counts']['history_blocks_per_query'],[2]);self.assertEqual(r['counts']['partial_block_present'],[True])
    def test_moe_histogram_not_uniform(self):
        w=wop('moe');r=build(w,cop('moe'),system());self.assertEqual(r['counts']['active_experts'],50);self.assertEqual(r['counts']['empty_experts'],462)
    def test_moe_weighted_width_exact(self):
        w=wop('moe');r=build(w,cop('moe'),system());self.assertEqual(r['counts']['useful_matrix_ops'],6*w['hidden']*sum(n*f for n,f in zip(w['expert_tokens'],w['expert_widths'])))
    def test_moe_padding_not_doublecounted(self):
        r=build(wop('moe'),cop('moe'),system());self.assertGreater(r['counts']['executed_matrix_ops_scenario'],r['counts']['useful_matrix_ops'])
    def test_moe_router_provided_omits_kernels(self):
        w=wop('moe');a=build(w,cop('moe'),system());w['router']='provided';b=build(w,cop('moe'),system());self.assertLess(len(b['phases']),len(a['phases']))
    def test_moe_bad_total(self):
        w=wop('moe');w['expert_tokens'][0]+=1
        with self.assertRaises(PlanError):build(w,cop('moe'),system())
    def test_moe_expert_cannot_get_token_twice(self):
        w=wop('moe');w['expert_tokens'][0]=w['tokens']+1
        with self.assertRaises(PlanError):build(w,cop('moe'),system())
    def test_moe_scale_group_split(self):
        w=wop('moe');w['expert_widths'][0]=320
        with self.assertRaises(PlanError):build(w,cop('moe'),system())
    def test_moe_warm_weight_assumption_explicit(self):
        w=wop('moe');w['cold_weight_fraction']=0;r=build(w,cop('moe'),system());self.assertEqual(r['counts']['cold_weight_traffic_scenario_bytes'],0);self.assertEqual(r['status'],'NEEDS_MEASUREMENT')
    def test_moe_invalid_cold_fraction(self):
        w=wop('moe');w['cold_weight_fraction']=1.01
        with self.assertRaises(PlanError):build(w,cop('moe'),system())
    def test_collective_bytes_both_directions(self):
        w=wop('collective');r=build(w,cop('collective'),system());self.assertEqual(r['counts']['wire_payload_bytes_total'],2*w['bytes'])
    def test_collective_ring_reduce_work(self):
        w=wop('collective');c=cop('collective',algorithm='ring2',chunk_bytes=65536)
        r=build(w,c,system());self.assertEqual(r['counts']['reduction_ops_total'],w['bytes']//w['element_bytes'])
    def test_collective_no_cpu_profile(self):
        s=system();s['devices'][0]['arch']='host'
        with self.assertRaises(PlanError):build(wop('collective'),cop('collective'),s)
    def test_collective_missing_direction(self):
        s=system();s['links']=[l for l in s['links'] if l['source']=='gpu0']
        with self.assertRaises(PlanError):build(wop('collective'),cop('collective'),s)
    def test_collective_pull_slower_under_declared_profile(self):
        a=build(wop('collective'),cop('collective',access='push'),system());b=build(wop('collective'),cop('collective',access='pull'),system());self.assertGreater(b['scenario_us'],a['scenario_us'])
    def test_collective_ring_rejects_half_misalignment(self):
        w=wop('collective');w['bytes']=48
        with self.assertRaises(PlanError):build(w,cop('collective',algorithm='ring2',chunk_bytes=65536),system())
    def test_collective_chunk_overflow_rejected(self):
        w=wop('collective');w['bytes']=16*4097
        with self.assertRaises(PlanError):build(w,cop('collective',chunk_bytes=16),system())
    def test_collective_exactly_two_starts(self):
        r=build(wop('collective'),cop('collective',chunk_bytes=4096),system());self.assertEqual(len([p for p in r['phases'] if p['id'].startswith('start:')]),2)
    def test_plan_all_new_models_request_measurement(self):
        r=op_plan(catalog(),workloads(),system())
        self.assertFalse(r['gpu_executed']);self.assertFalse(r['deployment_approved'])
        for w in r['workloads']:
            self.assertTrue(w['ranked'])
            for c in w['ranked']:self.assertEqual(c['status'],'NEEDS_MEASUREMENT');self.assertIsNone(c['interval_us'])
    def test_plan_rejected_variants_remain_visible(self):
        r=op_plan(catalog(),workloads(),system());self.assertTrue(next(w for w in r['workloads'] if w['workload']['op']=='attention')['rejected'])
