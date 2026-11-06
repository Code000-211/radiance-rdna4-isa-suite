import copy
import json
from pathlib import Path
import tempfile
import unittest
from radreplay.builder import Package
from radreplay.corpus import candidate, expect, coverage
from radreplay.format import ReplayError, file_hash, read_json, validate

class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.plugin=self.root/'a.so';self.plugin.write_bytes(b'baseline plugin')
        self.new=self.root/'b.so';self.new.write_bytes(b'candidate plugin')
        p=Package(self.root/'case','case0');p.plugin('test',self.plugin);p.stream()
        for a in ('x','y'):p.allocation(a,b'abcd',mutable=a=='y');p.view(a,a,nbytes=4,dtype=9,shape=[4])
        p.kernel('k','p0','row','op',['x','y'],[('x','r'),('y','w')]);p.check('y',b'abcd');self.path=p.finish()
        self.expected=self.root/'expected.json';expect([self.path],self.expected,'one approved host-only fixture, not all kernels')
        h=file_hash(self.path)
        self.r={'schema':'radiance-replay-result/1','status':'RUNTIME_CHECKS_PASS','runtime_executed':True,'exit_code':0,
                'fixture':{'id':'case0','manifest_sha256':h},'isa':{'status':'NOT_APPLICABLE_HOST'},'options':{'warmups':1,'repetitions':1},
                'runtime':{'status':'RUNTIME_CHECKS_PASS','fixture_sha256':h,'backend':'host','samples_us':[1.0],'checked_iterations':2}}
        self.report=self.root/'report.json';self.save()
    def save(self):self.report.write_text(json.dumps(self.r))
    def tearDown(self):self.tmp.cleanup()
    def test_candidate_explicit_lineage(self):
        c=candidate(self.path,self.root/'candidate',['p0='+str(self.new)]);m=read_json(c);orig=read_json(self.path)
        self.assertEqual(m['checks'],orig['checks']);self.assertEqual(m['allocations'],orig['allocations'])
        self.assertEqual(m['lineage']['parent_manifest_sha256'],file_hash(self.path));self.assertEqual(m['plugins'][0]['sha256'],file_hash(self.new))
    def test_identical_candidate_rejected(self):
        with self.assertRaises(ReplayError):candidate(self.path,self.root/'candidate',['p0='+str(self.plugin)])
    def test_unknown_candidate_rejected(self):
        with self.assertRaises(ReplayError):candidate(self.path,self.root/'candidate',['absent='+str(self.new)])
    def test_pass_scoped_only(self):
        rc,r=coverage(self.expected,[self.report],self.root/'cov');self.assertEqual(rc,0);self.assertFalse(r['all_radiance_kernels_covered'])
    def test_missing_report(self):
        rc,r=coverage(self.expected,[],self.root/'cov');self.assertEqual(rc,2);self.assertEqual(r['cases'][0]['status'],'MISSING')
    def test_duplicate_report(self):self.assertEqual(coverage(self.expected,[self.report,self.report],self.root/'cov')[0],2)
    def test_stale_fixture(self):
        self.path.write_text(self.path.read_text()+'\n');self.assertEqual(coverage(self.expected,[self.report],self.root/'cov')[0],2)
    def test_failed_case(self):
        self.r.update(status='FAIL',exit_code=1);self.save();self.assertEqual(coverage(self.expected,[self.report],self.root/'cov')[0],1)
    def test_no_execution_not_pass(self):
        self.r['runtime_executed']=False;self.save();self.assertEqual(coverage(self.expected,[self.report],self.root/'cov')[0],2)
    def test_unchecked_iteration(self):
        self.r['runtime']['checked_iterations']=1;self.save();self.assertEqual(coverage(self.expected,[self.report],self.root/'cov')[0],2)
    def test_unknown_report(self):
        self.r['fixture']['manifest_sha256']='0'*64;self.save();self.assertEqual(coverage(self.expected,[self.report],self.root/'cov')[0],2)
    def test_duplicate_corpus(self):
        with self.assertRaises(ReplayError):expect([self.path,self.path],self.root/'duplicate.json','scope')
    def test_lineage_tamper(self):
        c=candidate(self.path,self.root/'candidate',['p0='+str(self.new)]);m=read_json(c);m['lineage']['plugin_changes'][0]['after']='0'*64;c.write_text(json.dumps(m))
        with self.assertRaises(ReplayError):validate(c)
