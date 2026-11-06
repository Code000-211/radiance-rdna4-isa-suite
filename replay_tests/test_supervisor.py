import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from radreplay.builder import Package
from radreplay.cli import main

class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.plugin=self.root/'dummy.so';self.plugin.write_bytes(b'fake-worker tests only')
        p=Package(self.root/'case','supervisor');p.plugin('test',self.plugin);p.stream()
        for n in ('x','y'):p.allocation(n,b'abcd',mutable=n=='y');p.view(n,n,nbytes=4,dtype=9,shape=[4])
        p.kernel('k','p0','copy','copy',['x','y'],[('x','r'),('y','w')]);p.check('y',b'abcd');self.manifest=p.finish()
        self.worker=self.root/'fake-worker'
    def tearDown(self):self.tmp.cleanup()
    def fake(self,extra=''):
        self.worker.write_text('#!/usr/bin/env python3\nimport json,sys,time,pathlib\np=json.load(open(sys.argv[1]))\nr={"schema":"radiance-replay-worker-result/1","status":"RUNTIME_CHECKS_PASS","fixture_sha256":p["fixture_sha256"],"backend":"host","checked_iterations":p["warmups"]+p["repetitions"],"checks_per_iteration":1,"samples_us":[1.0]*p["repetitions"]}\n'+'''r['diagnostics']={'status':'PASS','completed_passes':3,'checked_nodes':len(p['fixture']['nodes']),'per_operation_write_contracts':True,'output_poison_variants':2,'scratch_contracts':True}
'''+extra+'\npathlib.Path(p["out"],"worker-result.json").write_text(json.dumps(r))\n')
        self.worker.chmod(0o700)
    def run_case(self,extra_args=(),out='run'):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            rc=main(['run',str(self.manifest),'--worker',str(self.worker),'--plugin','p0='+str(self.plugin),'--out',str(self.root/out),'--execute-trusted-code','--warmups','1','--reps','2',*extra_args])
        return rc,json.loads((self.root/out/'report.json').read_text())
    def test_complete_protocol(self):
        self.fake();rc,r=self.run_case();self.assertEqual(rc,0);self.assertEqual(r['status'],'RUNTIME_CHECKS_PASS')
    def test_missing_diagnostics_v03_worker_rejected(self):
        self.fake('del r["diagnostics"]');self.assertEqual(self.run_case()[0],2)
    def test_incomplete_diagnostic_passes_rejected(self):
        self.fake('r["diagnostics"]["completed_passes"]=2');self.assertEqual(self.run_case()[0],2)
    def test_missing_write_contract_rejected(self):
        self.fake('r["diagnostics"]["per_operation_write_contracts"]=False');self.assertEqual(self.run_case()[0],2)
    def test_missing_node_diagnostic_rejected(self):
        self.fake('r["diagnostics"]["checked_nodes"]=0');self.assertEqual(self.run_case()[0],2)
    def test_missing_checks(self):
        self.fake('r["checked_iterations"]=1');self.assertEqual(self.run_case()[0],2)
    def test_wrong_fixture(self):
        self.fake('r["fixture_sha256"]="wrong"');self.assertEqual(self.run_case()[0],2)
    def test_wrong_backend(self):
        self.fake('r["backend"]="aql"');self.assertEqual(self.run_case()[0],2)
    def test_missing_timing(self):
        self.fake('r["samples_us"]=[]');self.assertEqual(self.run_case()[0],2)
    def test_bad_timing(self):
        self.fake('r["samples_us"]=[-1,2]');self.assertEqual(self.run_case()[0],2)
    def test_exit_without_result(self):
        self.fake('sys.exit(0)');self.assertEqual(self.run_case()[0],2)
    def test_timeout(self):
        self.fake('time.sleep(5)');rc,r=self.run_case(['--timeout','.1']);self.assertEqual(rc,2);self.assertIn('TIMEOUT',r['error'])
    def test_plugin_modified(self):
        self.fake();self.plugin.write_bytes(b'modified');rc,r=self.run_case();self.assertEqual(rc,2);self.assertFalse(r['runtime_executed'])
    def test_explicit_execution_consent(self):
        self.fake()
        with contextlib.redirect_stdout(io.StringIO()):
            rc=main(['run',str(self.manifest),'--worker',str(self.worker),'--plugin','p0='+str(self.plugin),'--out',str(self.root/'no-consent')])
        self.assertEqual(rc,2)
    def test_gpu_needs_static_gate(self):
        self.fake();m=json.loads(self.manifest.read_text());m['target'].update(backend='aql',arch='gfx1201')
        for a in m['allocations']:a['kind']='device'
        self.manifest.write_text(json.dumps(m));rc,r=self.run_case();self.assertEqual(rc,2);self.assertFalse(r['runtime_executed'])
    def test_invalid_timeout(self):
        self.fake();self.assertEqual(self.run_case(['--timeout','nan'])[0],2)
    def test_fresh_output(self):
        self.fake();self.run_case()
        with contextlib.redirect_stderr(io.StringIO()):
            rc=main(['run',str(self.manifest),'--worker',str(self.worker),'--out',str(self.root/'run')])
        self.assertEqual(rc,2)
