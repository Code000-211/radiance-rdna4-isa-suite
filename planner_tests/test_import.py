"""Synthetic evidence validation tests, not measured benchmark claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from radplan.common import PlanError,file_hash,digest
from radplan.importer import import_replay,profile_from_replay
from radreplay.builder import Package
from support_planner import cpu_work,cpu_candidate,cpu_profile

class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.binary=self.root/'fake-plugin';self.binary.write_bytes(b'SYNTHETIC UNIT TEST ONLY')
        self.w=cpu_work(2,3,4);self.c=cpu_candidate('cpu');self.c['identity']['sha256']=file_hash(self.binary)
        p=Package(self.root/'fixture','unit-input');p.plugin('unit',self.binary);p.stream()
        for key,shape in [('a',[2,4]),('b',[3,4]),('y',[2,3])]:
            data=bytes(shape[0]*shape[1]*4);p.allocation(key,data,mutable=key=='y');p.view(key,key,nbytes=len(data),dtype=1,shape=shape)
        p.kernel('k','p0','cpu','gemm_nt',['a','b','y'],[('a','r'),('b','r'),('y','w')],params={'M':2,'N':3,'K':4,'dtype':'f32'})
        p.check('y',bytes(24));self.fixture=p.finish()
        self.r={'status':'RUNTIME_CHECKS_PASS','runtime_executed':True,'exit_code':0,'worker_sha256':'1'*64,'host':{'platform':'unit-host'},
            'options':{'warmups':1,'repetitions':3,'diagnostic_only':False},'runtime':{
            'status':'RUNTIME_CHECKS_PASS','fixture_sha256':file_hash(self.fixture),'worker_build':{'host_asan':False},
            'checked_iterations':4,'diagnostics':{'status':'PASS'},'backend':'host','device':{'arch':'host','name':'unit'},
            'measurement':'unit-mode','samples_us':[1,1,1]}}
        self.p=cpu_profile();self.p['environment']={'measurement_mode':'unit-mode','worker_sha256':'1'*64,'device_name':'unit'}
        self.path=self.root/'report.json';self.save()
    def tearDown(self):self.temp.cleanup()
    def save(self):self.path.write_text(json.dumps(self.r))
    def run_import(self):self.save();return import_replay(self.path,self.fixture,self.w,self.c,self.p)
    def test_import_binds_actual_file_hashes(self):
        r=self.run_import();self.assertEqual(r['evidence']['report_sha256'],file_hash(self.path));self.assertEqual(r['evidence']['fixture_sha256'],file_hash(self.fixture))
    def test_not_executed_rejected(self):
        self.r['runtime_executed']=False
        with self.assertRaises(PlanError):self.run_import()
    def test_failed_rejected(self):
        self.r['status']='FAIL'
        with self.assertRaises(PlanError):self.run_import()
    def test_asan_timings_rejected(self):
        self.r['runtime']['worker_build']['host_asan']=True
        with self.assertRaisesRegex(PlanError,'sanitized'):self.run_import()
    def test_diagnostic_only_rejected(self):
        self.r['options']['diagnostic_only']=True
        with self.assertRaises(PlanError):self.run_import()
    def test_unchecked_iterations_rejected(self):
        self.r['runtime']['checked_iterations']=1
        with self.assertRaises(PlanError):self.run_import()
    def test_missing_samples_rejected(self):
        self.r['runtime']['samples_us']=[1]
        with self.assertRaises(PlanError):self.run_import()
    def test_no_diagnostics_rejected(self):
        self.r['runtime']['diagnostics']={}
        with self.assertRaises(PlanError):self.run_import()
    def test_cpu_report_not_gpu(self):
        self.r['runtime']['backend']='aql'
        with self.assertRaises(PlanError):self.run_import()
    def test_wrong_shape_rejected(self):
        self.w['M']=3
        with self.assertRaises(PlanError):self.run_import()
    def test_wrong_row_rejected(self):
        self.c['launch']['row']='other'
        with self.assertRaises(PlanError):self.run_import()
    def test_wrong_binary_rejected(self):
        self.c['identity']['sha256']='c'*64
        with self.assertRaises(PlanError):self.run_import()
    def test_stale_fixture_rejected(self):
        self.r['runtime']['fixture_sha256']='b'*64
        with self.assertRaises(PlanError):self.run_import()
    def test_wrong_worker_rejected(self):
        self.p['environment']['worker_sha256']='b'*64
        with self.assertRaises(PlanError):self.run_import()
    def test_window_metric_mismatch(self):
        self.r['runtime']['measurement']='different-clock'
        with self.assertRaises(PlanError):self.run_import()
    def test_device_name_mismatch(self):
        self.r['runtime']['device']['name']='other CPU'
        with self.assertRaises(PlanError):self.run_import()
    def test_profile_from_replay_not_calibration(self):
        p=profile_from_replay(self.p,self.path,'fixed-test-settings','test-compiler')
        self.assertEqual(p['environment']['device_identity'],self.r['runtime']['device']);self.assertFalse(p['environment']['profile_declarations_verified'])
    def test_unknown_power_profile_refused(self):
        with self.assertRaises(PlanError):profile_from_replay(self.p,self.path,'UNMEASURED','test')
