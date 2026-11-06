import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from radreplay.builder import Package
from radreplay.format import read_json,file_hash,ReplayError,verified_json
from radreplay.corpus import expect
from radreplay.batch import execute

class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.plugin=self.root/'p.so';self.plugin.write_bytes(b'protocol mock only')
        self.cases=[]
        for i in range(3):
            p=Package(self.root/f'case{i}',f'case{i}');p.plugin('test',self.plugin);p.stream()
            for a in ('x','y'):p.allocation(a,b'abcd',mutable=a=='y');p.view(a,a,nbytes=4,dtype=9,shape=[4])
            p.kernel('k','p0','copy','copy',['x','y'],[('x','r'),('y','w')]);p.check('y',b'abcd');self.cases.append(p.finish())
        self.expected=self.root/'expected.json';expect(self.cases,self.expected,'three protocol mock cases')
        self.args=SimpleNamespace(expect=str(self.expected),out=str(self.root/'out'),execute_trusted_code=True,plugin=['p0='+str(self.plugin)],worker='mockworker',
                                  reps=3,warmups=1,timeout=5,device=0,gate=False,diagnostic_only=False,trace_dispatches=False,isa_manifest=None,isa_report=None)
        self.invocations=0
    def tearDown(self):self.tmp.cleanup()
    def invoke(self,cmd):
        self.invocations+=1;p=Path(cmd[1]);m=read_json(p);h=file_hash(p);out=Path(cmd[cmd.index('--out')+1]);out.mkdir()
        r={'schema':'radiance-replay-result/1','status':'RUNTIME_CHECKS_PASS','runtime_executed':True,'exit_code':0,
           'fixture':{'id':m['id'],'manifest_sha256':h},'isa':{'status':'NOT_APPLICABLE_HOST'},'options':{'warmups':1,'repetitions':3,'diagnostics':'contracts'},
           'runtime':{'status':'RUNTIME_CHECKS_PASS','fixture_sha256':h,'backend':'host','samples_us':[1,1,1],'checked_iterations':4,'diagnostics':{'status':'PASS'}}}
        (out/'report.json').write_text(json.dumps(r));return 0
    def test_complete_corpus(self):
        code,r=execute(self.args,self.invoke);self.assertEqual(code,0);self.assertEqual(self.invocations,3)
    def test_stop_failure(self):
        def fail(cmd):
            self.invoke(cmd);p=Path(cmd[cmd.index('--out')+1])/'report.json';r=read_json(p);r.update(status='FAIL',exit_code=1);p.write_text(json.dumps(r));return 1
        code,r=execute(self.args,fail);self.assertEqual(code,1);self.assertEqual(self.invocations,1);self.assertEqual(len(r['unattempted_cases']),2)
    def test_stop_timeout_or_incomplete(self):
        def incomplete(cmd):
            self.invoke(cmd);p=Path(cmd[cmd.index('--out')+1])/'report.json';r=read_json(p);r.update(status='INCOMPLETE',exit_code=2);p.write_text(json.dumps(r));return 2
        code,r=execute(self.args,incomplete);self.assertEqual(code,2);self.assertEqual(self.invocations,1)
    def test_later_stale_case_checked_before_any_execution(self):
        self.cases[2].write_text(self.cases[2].read_text()+'\n')
        with self.assertRaises(ReplayError):execute(self.args,self.invoke)
        self.assertEqual(self.invocations,0)
    def test_plugin_changed_before_any_execution(self):
        self.plugin.write_bytes(b'changed')
        with self.assertRaises(ReplayError):execute(self.args,self.invoke)
        self.assertEqual(self.invocations,0)
    def test_duplicate_case_blocks(self):
        m=read_json(self.expected);m['cases'].append(m['cases'][0]);self.expected.write_text(json.dumps(m))
        with self.assertRaises(ReplayError):execute(self.args,self.invoke)
    def test_consent_required(self):
        self.args.execute_trusted_code=False
        with self.assertRaises(ReplayError):execute(self.args,self.invoke)
    def test_verified_json_exact_bytes(self):
        h=file_hash(self.cases[0]);self.assertEqual(verified_json(self.cases[0],h)['id'],'case0')
        self.cases[0].write_text(self.cases[0].read_text()+' ')
        with self.assertRaises(ReplayError):verified_json(self.cases[0],h)
