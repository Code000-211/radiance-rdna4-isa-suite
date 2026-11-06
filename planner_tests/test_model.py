import copy
import math
import unittest
from radplan.common import PlanError
from radplan.model import features,workload,candidate,catalog_check,workloads_check,profile,shape_key
from support_planner import *

class ModelTests(unittest.TestCase):
    def setUp(self):self.p=gpu_profile();self.c=gpu_catalog()['candidates'][0];self.w=gpu_work()
    def test_default_shape_counts(self):
        f=features(self.w,self.c,self.p)
        self.assertEqual(f['groups'],10);self.assertEqual(f['useful_ops'],2*128*2560*640);self.assertEqual(f['tile_efficiency'],1)
    def test_partial_m_work(self):
        c=gpu_catalog()['candidates'][-4];w=gpu_work(5)
        self.assertFalse(c['known_issues']);f=features(w,c,self.p);self.assertAlmostEqual(f['tile_efficiency'],5/128)
    def test_source_tail_excluded(self):
        with self.assertRaisesRegex(PlanError,'Ascale'):features(gpu_work(129),self.c,self.p)
    def test_registry_constraint_not_just_bn(self):
        c=next(c for c in gpu_catalog()['candidates'] if c['id'].endswith('tile10'))
        with self.assertRaisesRegex(PlanError,'multiple'):features(gpu_work(128,128),c,self.p)
    def test_k_group_rejected(self):
        with self.assertRaises(PlanError):features(gpu_work(k=641),self.c,self.p)
    def test_no_cpu_gpu_mixing(self):
        with self.assertRaises(PlanError):features(self.w,self.c,cpu_profile())
    def test_contract_mismatch(self):
        self.w['contract']='different'
        with self.assertRaises(PlanError):features(self.w,self.c,self.p)
    def test_storage_not_compute(self):
        f=features(self.w,self.c,self.p);c=copy.deepcopy(self.c);c['compute_path']='wmma_i4'
        f4=features(self.w,c,self.p)
        self.assertGreater(f['scenario_terms_us']['matrix_and_unpack_us'],f4['scenario_terms_us']['matrix_and_unpack_us'])
    def test_format_mismatch(self):
        self.w['a_bits']=4
        with self.assertRaisesRegex(PlanError,'format'):features(self.w,self.c,self.p)
    def test_scale_bytes_counted(self):
        f=features(self.w,self.c,self.p)
        expected=128*640+2560*640//2+128*2560*2+2560*5*2+128*4
        self.assertEqual(f['unique_operand_bytes'],expected)
    def test_no_padding_double_penalty(self):
        c=gpu_catalog()['candidates'][-4];f=features(gpu_work(5),c,self.p);full=features(gpu_work(128),c,self.p)
        self.assertEqual(f['executed_ops'],full['executed_ops'])
    def test_compiler_register_granule(self):
        self.c['resources']={'vgprs':128,'descriptor_vgprs':128,'scratch_bytes':0,'origin':'compiler_metadata'}
        f=features(self.w,self.c,self.p);self.assertEqual(f['physical_vgpr_allocation'],144);self.assertEqual(f['register_limited_waves'],10)
    def test_descriptor_rounding_dominates(self):
        self.c['resources']={'vgprs':120,'descriptor_vgprs':128,'scratch_bytes':0,'origin':'compiler_metadata'}
        self.assertEqual(features(self.w,self.c,self.p)['physical_vgpr_allocation'],144)
    def test_wave64_registers(self):
        self.c['wave_size']=64;self.c['resources']={'vgprs':128,'descriptor_vgprs':128,'scratch_bytes':0,'origin':'compiler_metadata'}
        f=features(self.w,self.c,self.p);self.assertEqual(f['physical_vgpr_allocation'],132);self.assertEqual(f['register_limited_waves'],5)
    def test_missing_regs_not_zero(self):
        f=features(self.w,self.c,self.p);self.assertIsNone(f['register_limited_waves']);self.assertTrue(f['warnings'])
    def test_lds_resource_limit(self):
        self.c['lds_bytes']=65537
        with self.assertRaisesRegex(PlanError,'LDS'):features(self.w,self.c,self.p)
    def test_thread_limit(self):
        self.c['waves_per_group']=32;self.c['wave_size']=64
        with self.assertRaisesRegex(PlanError,'thread'):features(self.w,self.c,self.p)
    def test_scratch_is_not_assumed_traffic(self):
        self.c['resources']={'vgprs':96,'descriptor_vgprs':96,'scratch_bytes':20,'origin':'compiler_metadata'}
        f=features(self.w,self.c,self.p);self.assertFalse(f['scratch_traffic_modeled']);self.assertTrue(any('scratch' in x for x in f['warnings']))
    def test_warm_has_no_dram_lower_bound(self):
        self.w['cache_state']='warm';f=features(self.w,self.c,self.p);self.assertIsNone(f['cold_resource_floor_us'])
    def test_cold_floor_exists(self):self.assertGreater(features(self.w,self.c,self.p)['cold_resource_floor_us'],0)
    def test_scenario_not_confidence_interval(self):self.assertIsNone(features(self.w,self.c,self.p)['confidence_interval'])
    def test_reuse_changes_traffic(self):
        self.w=gpu_work(256);f=features(self.w,self.c,self.p);self.p['assumptions']['reuse_fraction']=0
        self.assertGreaterEqual(features(self.w,self.c,self.p)['scenario_memory_bytes'],f['scenario_memory_bytes'])
    def test_split_reduction_not_double_counted(self):
        p=cpu_profile();c=cpu_candidate();w=cpu_work();c['split_k']=2
        f=features(w,c,p);self.assertEqual(f['split_reduction_bytes'],4*8*16*2+4*8*16)
    def test_split_k_nondivisible(self):
        c=cpu_candidate();c['split_k']=3
        with self.assertRaises(PlanError):features(cpu_work(),c,cpu_profile())
    def test_unsupported_op_refused(self):
        self.w['op']='qsa'
        with self.assertRaises(PlanError):workload(self.w)
    def test_boolean_shape_rejected(self):
        self.w['M']=True
        with self.assertRaises(PlanError):workload(self.w)
    def test_zero_shape_rejected(self):
        self.w['M']=0
        with self.assertRaises(PlanError):workload(self.w)
    def test_nan_metadata_rejected(self):
        self.w['weight_scale_bytes']=math.nan
        with self.assertRaises(PlanError):workload(self.w)
    def test_negative_efficiency_rejected(self):
        self.p['assumptions']['memory_efficiency']=-1
        with self.assertRaises(PlanError):profile(self.p)
    def test_efficiency_above_one_rejected(self):
        self.p['assumptions']['compute_efficiency']=1.1
        with self.assertRaises(PlanError):profile(self.p)
    def test_unknown_workload_fields(self):
        self.w['typo']=1
        with self.assertRaises(PlanError):workload(self.w)
    def test_unknown_candidate_fields(self):
        self.c['typo']=1
        with self.assertRaises(PlanError):candidate(self.c)
    def test_unknown_throughput_refused(self):
        self.c['compute_path']='pretend-fast'
        with self.assertRaises(PlanError):features(self.w,self.c,self.p)
    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(PlanError):catalog_check({'schema':'radiance-planner-catalog/1','candidates':[self.c,self.c]})
    def test_empty_workloads_rejected(self):
        with self.assertRaises(PlanError):workloads_check({'schema':'radiance-planner-workloads/1','workloads':[]})
    def test_duplicate_workloads_rejected(self):
        with self.assertRaises(PlanError):workloads_check({'schema':'radiance-planner-workloads/1','workloads':[self.w,self.w]})
    def test_shape_identity_ignores_names(self):
        b=copy.deepcopy(self.w);b['id']='other';b['note']='different';self.assertEqual(shape_key(self.w),shape_key(b))
    def test_shape_identity_includes_cache(self):
        b=copy.deepcopy(self.w);b['cache_state']='warm';self.assertNotEqual(shape_key(self.w),shape_key(b))
    def test_no_nan_outputs(self):
        import json
        json.dumps(features(self.w,self.c,self.p),allow_nan=False)
