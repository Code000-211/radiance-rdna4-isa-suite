"""Real C++/Radiance host execution. Missing prerequisites are errors, never skips."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from radreplay.builder import Package
from radreplay.cli import main
from radreplay.format import validate
from examples.libref_chain import build

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bindir=Path(os.environ['RAD_REPLAY_TEST_BUILD']).resolve()
        cls.libref=Path(os.environ['RAD_REPLAY_TEST_LIBREF']).resolve()
        cls.worker=cls.bindir/'rad-kbench-replay';cls.plugin=cls.bindir/'replay_test.so';cls.capture=cls.bindir/'replay-capture-harness'
        for p in (cls.worker,cls.plugin,cls.capture,cls.libref):
            if not p.is_file():raise RuntimeError('required compiled artifact missing: '+str(p))
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def package(self,row='copy',*,alias=False,scratch=0,nonfinite='reject',numeric=False,expected=None):
        p=Package(self.root/'case','native-'+row);p.plugin('replay_test',self.plugin);p.stream()
        raw=bytes(range(4))
        p.allocation('x',raw,mutable=alias)
        if not alias:p.allocation('y',bytes(4),mutable=True)
        p.view('x','x',nbytes=4,dtype=9,shape=[4]);p.view('y','x' if alias else 'y',nbytes=4,dtype=1 if numeric else 9,shape=[1] if numeric else [4])
        p.kernel('k','p0',row,'replay_probe',['x','y'],[('x','r'),('y','w')],scratch_bytes=scratch,scratch_policy="zeroed" if scratch else None)
        p.check('y',raw if expected is None else expected,mode='float' if numeric else 'bytes',nonfinite=nonfinite)
        return p.finish()
    def run_case(self,path,plugin=None,**kw):
        args=['run',str(path),'--worker',str(self.worker),'--plugin','p0='+str(plugin or self.plugin),'--out',str(self.root/'run'),'--execute-trusted-code','--warmups','1','--reps','3']
        for k,v in kw.items():args.extend(['--'+k,str(v)])
        with contextlib.redirect_stdout(io.StringIO()):rc=main(args)
        return rc,json.loads((self.root/'run'/'report.json').read_text())
    def test_real_registry_and_host_copy(self):
        rc,r=self.run_case(self.package());self.assertEqual(rc,0,r);self.assertEqual(r['runtime']['checked_iterations'],4)
    def test_alias_preserved(self):
        rc,r=self.run_case(self.package(alias=True));self.assertEqual(rc,0,r)
    def test_wrong_output_fails(self):
        rc,r=self.run_case(self.package('bad'));self.assertEqual(rc,1,r);self.assertIn('comparison failed',r['runtime']['error'])
    def test_allocation_redzone(self):
        rc,r=self.run_case(self.package('guard'));self.assertEqual(rc,1,r);self.assertIn('red-zone',r['runtime']['error'])
    def test_immutable_input(self):
        rc,r=self.run_case(self.package('mutate'));self.assertEqual(rc,1,r);self.assertIn('immutable',r['runtime']['error'])
    def test_scratch_restored_each_iteration(self):
        rc,r=self.run_case(self.package('scratch',scratch=16));self.assertEqual(rc,0,r)
    def test_scratch_guard(self):
        rc,r=self.run_case(self.package('scratchbad',scratch=16));self.assertEqual(rc,1,r);self.assertIn('scratch',r['runtime']['error'])
    def test_scratch_underallocation_refused(self):
        rc,r=self.run_case(self.package('scratch',scratch=8));self.assertEqual(rc,2,r);self.assertIn('scratch',r['error'])
    def test_hidden_state_row_refused(self):
        rc,r=self.run_case(self.package('stateful'));self.assertEqual(rc,2,r);self.assertIn('stateless',r['error'])
    def test_unknown_kernel_refused(self):
        rc,r=self.run_case(self.package('unknown'));self.assertEqual(rc,2,r)
    def test_hung_worker_stopped(self):
        rc,r=self.run_case(self.package('hang'),timeout='.5');self.assertEqual(rc,2,r);self.assertIn('TIMEOUT',r['error'])
    def test_nan_rejected(self):
        rc,r=self.run_case(self.package('nan',numeric=True,expected=struct.pack('<f',float('nan'))));self.assertEqual(rc,1,r)
    def test_matching_nan_explicit(self):
        rc,r=self.run_case(self.package('nan',numeric=True,nonfinite='match',expected=struct.pack('<f',float('nan'))));self.assertEqual(rc,0,r)
    def test_inf_is_not_nan(self):
        rc,r=self.run_case(self.package('inf',numeric=True,nonfinite='match',expected=struct.pack('<f',float('nan'))));self.assertEqual(rc,1,r)
    def test_indirect_pointer_relocation(self):
        p=Package(self.root/'case','indirect');p.plugin('replay_test',self.plugin);p.stream()
        p.allocation('x',b'xxxxabcd');p.allocation('y',bytes(4),mutable=True)
        p.allocation('ptr',bytes(8),relocations=[dict(offset=0,target='x',addend=4)])
        p.view('x','x',nbytes=8,dtype=9,shape=[8]);p.view('y','y',nbytes=4,dtype=9,shape=[4]);p.view('ptr','ptr',nbytes=8,dtype=12,shape=[1])
        p.kernel('k','p0','indirect','replay_indirect',['ptr','y'],[('ptr','r'),('x','r'),('y','w')]);p.check('y',b'abcd')
        rc,r=self.run_case(p.finish());self.assertEqual(rc,0,r)
    def test_libref_chain_and_restore(self):
        path=build(self.root/'chain',self.libref);rc,r=self.run_case(path,plugin=self.libref);self.assertEqual(rc,0,r)
    def test_capture_real_libref_then_replay(self):
        q=subprocess.run([str(self.capture),str(self.libref),str(self.root/'capture')],capture_output=True,text=True,timeout=15)
        self.assertEqual(q.returncode,0,q.stderr)
        cases=list((self.root/'capture').glob('case-*/manifest.json'));self.assertEqual(len(cases),1)
        info=validate(cases[0]);self.assertEqual(info['provenance']['kind'],'kbench-seeded');self.assertEqual(info['provenance']['reference'],'captured-baseline')
        rc,r=self.run_case(cases[0],plugin=self.libref);self.assertEqual(rc,0,r)
    def test_frozen_tunable_outside_schema(self):
        path=self.package('tuned');m=json.loads(path.read_text());m['nodes'][0]['params']=[{'key':'tile','kind':'int','value':2}];path.write_text(json.dumps(m))
        rc,r=self.run_case(path);self.assertEqual(rc,0,r)
    def test_missing_tunable_refused(self):
        rc,r=self.run_case(self.package('tuned'));self.assertEqual(rc,2,r)
    def test_illegal_tunable_refused(self):
        path=self.package('tuned');m=json.loads(path.read_text());m['nodes'][0]['params']=[{'key':'tile','kind':'int','value':3}];path.write_text(json.dumps(m))
        rc,r=self.run_case(path);self.assertEqual(rc,2,r)
    def test_unknown_parameter_refused(self):
        path=self.package();m=json.loads(path.read_text());m['nodes'][0]['params']=[{'key':'typo','kind':'int','value':1}];path.write_text(json.dumps(m))
        rc,r=self.run_case(path);self.assertEqual(rc,2,r)
    def test_capture_fresh_directory(self):
        d=self.root/'capture';d.mkdir()
        q=subprocess.run([str(self.capture),str(self.libref),str(d)],capture_output=True,text=True,timeout=15)
        self.assertEqual(q.returncode,2);self.assertIn('already exists',q.stderr)
