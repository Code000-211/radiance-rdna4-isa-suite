import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from radplan.common import PlanError,read,write,new_dir,digest
from radplan.planner import plan,serial_path
from radplan.cli import main
from radplan.radiance import int_expr,extract,bind_resources,KNOWN_SOURCE
from support_planner import *

class PlannerTests(unittest.TestCase):
    def setUp(self):self.p=gpu_profile();self.cat=gpu_catalog();self.ws={'schema':'radiance-planner-workloads/1','workloads':[gpu_work()]}
    def test_uncalibrated_is_not_prediction(self):
        r=plan(self.cat,self.ws,self.p);w=r['workloads'][0]
        self.assertEqual(w['status'],'NEEDS_MEASUREMENT');self.assertIsNone(w['ranked'][0]['prediction']['prediction_us'])
    def test_no_execution_side_effect_claim(self):
        r=plan(self.cat,self.ws,self.p);self.assertFalse(r['gpu_executed']);self.assertFalse(r['deployment_approved'])
    def test_candidates_excluded_not_silently_changed(self):
        self.ws['workloads'][0]=gpu_work(129);r=plan(self.cat,self.ws,self.p)['workloads'][0]
        self.assertEqual(len(r['ranked']),4);self.assertEqual(len(r['rejected']),10)
    def test_no_candidates_explicit(self):
        self.ws['workloads'][0]=gpu_work(n=33);r=plan(self.cat,self.ws,self.p)['workloads'][0]
        self.assertEqual(r['status'],'NO_SUPPORTED_CANDIDATE');self.assertIsNone(r['suggested_candidate'])
    def test_plan_deterministic(self):self.assertEqual(plan(self.cat,self.ws,self.p),plan(self.cat,self.ws,self.p))
    def test_profile_fingerprint_changes(self):
        old=plan(self.cat,self.ws,self.p)['profile_sha256'];self.p['environment']['clock_power']='different'
        self.assertNotEqual(old,plan(self.cat,self.ws,self.p)['profile_sha256'])
    def test_no_static_test_override(self):
        r=plan(self.cat,self.ws,self.p)
        self.assertTrue(r['workloads'][0]['correctness_validation_still_required'])
    def test_unknown_tp2_field_refused(self):
        self.ws['workloads'][0]['world_size']=2
        with self.assertRaises(PlanError):plan(self.cat,self.ws,self.p)

class PathTests(unittest.TestCase):
    def setUp(self):
        cs=[cpu_candidate('a'),cpu_candidate('b')];cs[0].update(input_layout='x',output_layout='x');cs[1].update(input_layout='y',output_layout='y')
        self.r=plan({'schema':'radiance-planner-catalog/1','candidates':cs},{'schema':'radiance-planner-workloads/1','workloads':[cpu_work()]},cpu_profile())
        self.wid=self.r['workloads'][0]['workload_id']
        self.spec={'schema':'radiance-planner-path/1','execution':'serial_no_overlap',
            'stages':[{'workload_id':self.wid,'allowed_candidates':['a']},{'workload_id':self.wid,'allowed_candidates':['b']}],
            'transitions':[{'from':'x','to':'y','cost_us':10,'provenance':'assumed'}]}
    def test_explicit_transition_charged(self):
        r=serial_path(self.r,self.spec);costs=sum(x['cost']['scenario_us'] for x in self.r['workloads'][0]['ranked'])
        self.assertAlmostEqual(r['best_path']['cost_us'],costs+10);self.assertFalse(r['production_latency_proved'])
    def test_missing_conversion_refused(self):
        self.spec['transitions']=[]
        with self.assertRaises(PlanError):serial_path(self.r,self.spec)
    def test_same_layout_zero_cost(self):
        self.spec['stages'][1]['allowed_candidates']=['a'];self.spec['transitions']=[]
        self.assertEqual(serial_path(self.r,self.spec)['best_path']['candidates'],['a','a'])
    def test_concurrency_not_assumed(self):
        self.spec['execution']='TP2_overlap'
        with self.assertRaises(PlanError):serial_path(self.r,self.spec)
    def test_negative_transition_refused(self):
        self.spec['transitions'][0]['cost_us']=-10
        with self.assertRaises(PlanError):serial_path(self.r,self.spec)
    def test_unknown_candidate_refused(self):
        self.spec['stages'][0]['allowed_candidates']=['invented']
        with self.assertRaises(PlanError):serial_path(self.r,self.spec)
    def test_empty_stage_refused(self):
        self.spec['stages'][0]['allowed_candidates']=[]
        with self.assertRaises(PlanError):serial_path(self.r,self.spec)

class SourceTests(unittest.TestCase):
    def test_exact_fourteen_arms(self):self.assertEqual(len(gpu_catalog()['candidates']),14)
    def test_int8_not_int4_peak_path(self):self.assertTrue(all(c['compute_path']=='wmma_i8' for c in gpu_catalog()['candidates']))
    def test_declared_shared_allocation(self):
        for c in gpu_catalog()['candidates']:
            bm,bn,_=c['tile'];self.assertEqual(c['lds_bytes'],2*(bm*64+bn*32))
    def test_parse_bitwise_options(self):self.assertEqual(int_expr('8 | 128 | 256 | 512'),904)
    def test_no_eval(self):
        for value in ['__import__("os")','2**100000','1+2','True','-1']:
            with self.subTest(value=value),self.assertRaises(PlanError):int_expr(value)
    def test_modified_source_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for name in KNOWN_SOURCE:(p/name).parent.mkdir(parents=True,exist_ok=True);(p/name).write_text('changed')
            with self.assertRaises(PlanError):extract(p)
    def test_no_source_implies_no_catalog(self):
        with tempfile.TemporaryDirectory() as t,self.assertRaises(OSError):extract(Path(t))

class IOAndCLITests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_duplicate_json_rejected(self):
        p=self.root/'x';p.write_text('{"a":1,"a":2}')
        with self.assertRaises(PlanError):read(p)
    def test_nonfinite_json_rejected(self):
        p=self.root/'x';p.write_text('{"a":NaN}')
        with self.assertRaises(PlanError):read(p)
    def test_output_not_overwritten(self):
        p=self.root/'x';write(p,{'a':1})
        with self.assertRaises(FileExistsError):write(p,{'a':2})
    def test_output_dir_not_reused(self):
        p=new_dir(self.root/'o')
        with self.assertRaises(PlanError):new_dir(p)
    def test_plan_cli(self):
        rc=main(['plan','--catalog',str(ROOT/'planner/examples/radiance-w4a8-catalog.json'),'--profile',str(ROOT/'planner/profiles/r9700-unmeasured.json'),
                 '--workloads',str(ROOT/'planner/examples/w4a8-workloads.json'),'--out',str(self.root/'out')])
        self.assertEqual(rc,0);self.assertTrue((self.root/'out/plan.md').is_file())
    def test_require_calibrated_exit2(self):
        rc=main(['plan','--catalog',str(ROOT/'planner/examples/radiance-w4a8-catalog.json'),'--profile',str(ROOT/'planner/profiles/r9700-unmeasured.json'),
                 '--workloads',str(ROOT/'planner/examples/w4a8-workloads.json'),'--out',str(self.root/'out'),'--require-calibrated'])
        self.assertEqual(rc,2)
    def test_malformed_cli_input_not_pass(self):
        p=self.root/'bad';p.write_text('{}')
        with contextlib.redirect_stderr(io.StringIO()):rc=main(['plan','--catalog',str(p),'--profile',str(p),'--workloads',str(p),'--out',str(self.root/'out')])
        self.assertEqual(rc,2)
    def test_native_execution_requires_consent_flag(self):
        with contextlib.redirect_stderr(io.StringIO()):rc=main(['cpu-demo','--worker','/none','--plugin','/none','--libref','/none','--out',str(self.root/'out')])
        self.assertEqual(rc,2);self.assertFalse((self.root/'out').exists())
