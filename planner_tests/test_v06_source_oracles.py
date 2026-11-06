import unittest,copy,math,tempfile
from pathlib import Path
from unittest.mock import patch
from radplan.common import PlanError
from radplan.source_ops import block_slices,shared_slices,model_workloads,verify_source,PINNED
from radplan.reference_ops import attention_reference,gated_expert,sharded_expert
from support_v06 import *

class SourceTests(unittest.TestCase):
    def test_tp2_384_256(self):self.assertEqual(block_slices(640,2,0),[(0,384),(384,640)])
    def test_tp2_expert_parity_rotates(self):self.assertEqual(block_slices(640,2,1),[(384,640),(0,384)])
    def test_tp4_exact_coverage(self):
        for e in range(8):
            sl=block_slices(640,4,e);self.assertEqual(sorted(x for lo,hi in sl for x in range(lo,hi)),list(range(640)));self.assertEqual(sorted(hi-lo for lo,hi in sl),[128,128,128,256])
    def test_equal_split(self):self.assertEqual(block_slices(1024,2,0),[(0,512),(512,1024)])
    def test_no_fractional_block_split(self):
        with self.assertRaises(PlanError):block_slices(641,2,0)
    def test_unsupported_world(self):
        with self.assertRaises(PlanError):block_slices(768,3,0)
    def test_shared_rotation(self):self.assertEqual(shared_slices(640,640,2),[(0,384),(384,640)])
    def test_shared_zero(self):self.assertEqual(shared_slices(0,640,2),[(0,0),(0,0)])
    def test_missing_source_not_trusted(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(PlanError):verify_source(Path(d))
    def expand(self,cfg=None,run=None):
        # Parser unit tests isolate hash validation. Real source pins are checked
        # separately by the end-to-end release command, not asserted here.
        with patch('radplan.source_ops.verify_source',return_value=PINNED):
            return model_workloads(cfg or load('qwen4exp-example-config.json'),run or load('qwen4exp-example-run.json'),Path('.'))
    def test_partial_model_counts(self):
        r=self.expand();self.assertEqual(len(r['workloads']),144);self.assertFalse(r['scope']['complete_model']);self.assertEqual(len(r['scope']['unmodeled_recurrent_layers']),36)
    def test_dense_requests_not_fake_dispatch(self):
        r=self.expand();self.assertEqual(len(r['scope']['dense_shape_requests']),120)
        self.assertTrue(all(x['status']=='NEEDS_RECIPE_FORMAT_AND_MATCHING_GEMM_CATALOG' for x in r['scope']['dense_shape_requests']))
    def test_model_histogram_mandatory(self):
        run=load('qwen4exp-example-run.json');run['routing_by_layer'].pop('1')
        with self.assertRaises(PlanError):self.expand(run=run)
    def test_missing_dimensions_not_inferred(self):
        cfg=load('qwen4exp-example-config.json');cfg.pop('head_dim')
        with self.assertRaises(PlanError):self.expand(cfg=cfg)
    def test_architecture_not_guessed(self):
        cfg=load('qwen4exp-example-config.json');cfg['model_type']='qwen_other'
        with self.assertRaises(PlanError):self.expand(cfg=cfg)
    def test_sparse_tail_required(self):
        run=load('qwen4exp-example-run.json');ids=run['selection_by_layer']['3'][1];ids.pop()
        with self.assertRaises(PlanError):self.expand(run=run)
    def test_sparse_dense_fallback_rejected_at_long_context(self):
        run=load('qwen4exp-example-run.json');run['selection_by_layer']['3']=None
        with self.assertRaises(PlanError):self.expand(run=run)
    def test_no_synthetic_routing_inference(self):
        run=load('qwen4exp-example-run.json');run['routing_by_layer']['0']=[0]*512
        with self.assertRaises(PlanError):self.expand(run=run)
    def test_partition_uses_expert_identity(self):
        r=self.expand();p=r['scope']['partition'];self.assertEqual(p[0]['expert_widths'][:2],[384,256]);self.assertEqual(p[1]['expert_widths'][:2],[256,384])
    def test_input_type_validation(self):
        run=load('qwen4exp-example-run.json');run['selection_by_layer']['3'][0][0]='bad'
        with self.assertRaises(PlanError):self.expand(run=run)

class OracleTests(unittest.TestCase):
    def test_uniform_attention_mean(self):
        r=attention_reference([[0.,0.]],[[1.,2.],[3.,4.]],[[2.,4.],[4.,8.]],[[0,1]])
        self.assertEqual(r,[[3.,6.]])
    def test_attention_single_key(self):self.assertEqual(attention_reference([[2.]],[[3.]],[[5.,7.]],[[0]]),[[5.,7.]])
    def test_online_softmax_matches_reference(self):
        q=[[i*.1,-i*.07,.3] for i in range(4)];k=[[i*.2,1-i*.1,-.1] for i in range(17)];v=[[i*.5,-i] for i in range(17)];vis=[list(range(5+i*4)) for i in range(4)]
        expected=attention_reference(q,k,v,vis)
        for block in (1,2,4,8,32):
            got=attention_reference(q,k,v,vis,chunk_size=block)
            for a,b in zip(expected,got):
                for x,y in zip(a,b):self.assertAlmostEqual(x,y,12)
    def test_attention_sparse_permutation(self):
        a=attention_reference([[1.]],[[1.],[2.],[3.]],[[2.],[4.],[8.]],[[0,2]])
        b=attention_reference([[1.]],[[1.],[2.],[3.]],[[2.],[4.],[8.]],[[2,0]])
        self.assertEqual(a,b)
    def test_stable_large_logits(self):
        self.assertTrue(math.isfinite(attention_reference([[1000.]],[[1000.],[999.]],[[2.],[3.]],[[0,1]],chunk_size=1)[0][0]))
    def test_attention_duplicate_index_rejected(self):
        with self.assertRaises(PlanError):attention_reference([[1.]],[[1.]],[[1.]],[[0,0]])
    def test_attention_nan_rejected(self):
        with self.assertRaises(PlanError):attention_reference([[float('nan')]],[[1.]],[[1.]],[[0]])
    def test_expert_partition_math(self):
        x=[[.25,-.5],[1.,0.]];g=[[i*.001,.1] for i in range(640)];u=[[.2,-i*.002] for i in range(640)];d=[[.001*(i%7) for i in range(640)],[-.002*(i%5) for i in range(640)]]
        ref=gated_expert(x,g,u,d)
        for world in (1,2,4):
            for e in (0,1,2,3):
                got=sharded_expert(x,g,u,d,block_slices(640,world,e))
                for a,b in zip(ref,got):
                    for xx,yy in zip(a,b):self.assertAlmostEqual(xx,yy,12)
    def test_expert_duplicate_partition_rejected(self):
        with self.assertRaises(PlanError):sharded_expert([[1.]],[[1.],[2.]],[[1.],[2.]],[[1.,2.]],[(0,2),(0,2)])
    def test_expert_zero_input(self):self.assertEqual(gated_expert([[0.]],[[1.]],[[2.]],[[3.]]),[[0.]])
