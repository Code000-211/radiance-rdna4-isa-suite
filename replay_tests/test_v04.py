import copy,json,math,struct,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from radreplay.builder import Package
from radreplay.format import read_json,file_hash,validate,ReplayError
from radreplay.corpus import candidate,expect
from radreplay.qualification import qualify,row_coverage,numeric_policy
from radreplay.ab import workload_identity,summarize,experiment
from radreplay.oracles import gemm_nt,boundaries

class V04Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.plugin=self.root/'p.so';self.plugin.write_bytes(b'trusted test protocol, no native execution')
        self.worker=self.root/'worker';self.worker.write_bytes(b'test fake protocol only')
        p=Package(self.root/'case','v04');p.plugin('test',self.plugin);p.stream()
        for a in ('x','y'):p.allocation(a,b'abcd',mutable=a=='y');p.view(a,a,nbytes=4,dtype=9,shape=[4])
        p.kernel('k','p0','copy','copy',['x','y'],[('x','r'),('y','w')]);p.check('y',b'abcd')
        self.manifest=p.finish();self.m=read_json(self.manifest);self.info=validate(self.manifest)
        self.expected=self.root/'expected.json';expect([self.manifest],self.expected,'one explicit test fixture')
        self.d={'status':'PASS','completed_passes':3,'checked_nodes':1,'per_operation_write_contracts':True,'output_poison_variants':2,'scratch_contracts':True}
        h=file_hash(self.manifest)
        self.r={'schema':'radiance-replay-result/1','status':'RUNTIME_CHECKS_PASS','runtime_executed':True,'exit_code':0,'worker_sha256':file_hash(self.worker),
                'fixture':{'id':'v04','manifest_sha256':h},'isa':{'status':'NOT_APPLICABLE_HOST'},'options':{'diagnostics':'contracts','warmups':1,'repetitions':3},
                'runtime':{'status':'RUNTIME_CHECKS_PASS','fixture_sha256':h,'backend':'host','samples_us':[1.,1.,1.],'checked_iterations':4,'diagnostics':self.d}}
        self.report=self.root/'report.json';self.save()
        self.inv={'schema':'radiance-declared-rows/1','plugin_sha256':file_hash(self.plugin),'declared_row_count':2,'rows':[{'name':'copy','tunables':[]},{'name':'other','tunables':[]}]}
        self.inventory=self.root/'inventory.json';self.inventory.write_text(json.dumps(self.inv))
    def tearDown(self):self.tmp.cleanup()
    def save(self):self.report.write_text(json.dumps(self.r))
    def q(self,profile='host-contracts',reports=None):return qualify(self.expected,[self.report] if reports is None else reports,self.root/'q',profile=profile)
    def test_host_scoped_pass(self):
        code,r=self.q();self.assertEqual(code,0);self.assertFalse(r['full_memory_safety_proved'])
    def test_missing_diagnostics_blocks(self):
        del self.r['runtime']['diagnostics'];self.save();self.assertEqual(self.q()[0],2)
    def test_partial_passes_blocks(self):
        self.d['completed_passes']=2;self.save();self.assertEqual(self.q()[0],2)
    def test_no_execution_blocks(self):
        self.r['runtime_executed']=False;self.save();self.assertEqual(self.q()[0],2)
    def test_failure_is_failure(self):
        self.r['status']='FAIL';self.r['exit_code']=1;self.save();self.assertEqual(self.q()[0],1)
    def test_missing_report_blocks(self):self.assertEqual(self.q(reports=[])[0],2)
    def test_duplicate_report_blocks(self):self.assertEqual(self.q(reports=[self.report,self.report])[0],2)
    def test_host_is_not_gpu(self):self.assertEqual(self.q('single-gpu')[0],2)
    def test_tp2_unimplemented_cannot_be_claimed(self):
        self.r['dispatch_binding']={'status':'DISPATCH_BINDING_PASS'};self.r['gpu_asan']=True;self.save()
        code,r=self.q('production-tp2');self.assertEqual(code,2)
        missing={x['name'] for x in r['requirements'] if x['status']=='MISSING'};self.assertIn('tp2-protocol-tests',missing)
    def test_captured_not_independent(self):
        self.m['provenance']['reference']='captured-baseline';self.manifest.write_text(json.dumps(self.m))
        self.expected.unlink();expect([self.manifest],self.expected,'updated fixture')
        h=file_hash(self.manifest);self.r['fixture']['manifest_sha256']=h;self.r['runtime']['fixture_sha256']=h;self.save()
        self.assertEqual(self.q()[0],2)
    def test_traversal_approved_corpus_not_followed(self):
        e=read_json(self.expected);e['cases'][0]['manifest']='../escape.json';self.expected.write_text(json.dumps(e))
        self.assertEqual(self.q()[0],2)
    def test_all_declared_rows_denominator(self):
        code,r=row_coverage(self.inventory,[self.manifest],[self.report],self.root/'rows')
        self.assertEqual((code,r['tested_rows'],r['declared_rows']),(2,1,2));self.assertFalse(r['all_templates_tested'])
    def test_row_coverage_rejects_fake_execution(self):
        self.r['runtime_executed']=False;self.save()
        code,r=row_coverage(self.inventory,[self.manifest],[self.report],self.root/'rows');self.assertEqual(r['tested_rows'],0)
    def test_row_coverage_missing(self):
        code,r=row_coverage(self.inventory,[self.manifest],[],self.root/'rows');self.assertEqual(code,2);self.assertTrue(r['errors'])
    def test_row_coverage_whole_declared_set(self):
        self.inv['rows'].pop();self.inv['declared_row_count']=1;self.inventory.write_text(json.dumps(self.inv))
        code,r=row_coverage(self.inventory,[self.manifest],[self.report],self.root/'rows');self.assertEqual(code,0);self.assertFalse(r['rows'][0]['all_tuning_combinations_tested'])
    def test_row_coverage_wrong_plugin_no_credit(self):
        self.inv['plugin_sha256']='f'*64;self.inventory.write_text(json.dumps(self.inv))
        self.assertEqual(row_coverage(self.inventory,[self.manifest],[self.report],self.root/'rows')[1]['tested_rows'],0)
    def test_checkpoint_blob_copied_candidate(self):
        self.m['checkpoints']=[{'after':'k','checks':copy.deepcopy(self.m['checks'])}];self.manifest.write_text(json.dumps(self.m));validate(self.manifest)
        p=self.root/'other.so';p.write_bytes(b'new binary')
        c=candidate(self.manifest,self.root/'candidate',['p0='+str(p)]);validate(c)
        self.assertEqual(read_json(c)['checkpoints'],self.m['checkpoints'])
    def test_checkpoint_invalid_node(self):
        self.m['checkpoints']=[{'after':'absent','checks':self.m['checks']}];self.manifest.write_text(json.dumps(self.m))
        with self.assertRaises(ReplayError):validate(self.manifest)
    def test_checkpoint_duplicate_after(self):
        c={'after':'k','checks':self.m['checks']};self.m['checkpoints']=[c,c];self.manifest.write_text(json.dumps(self.m))
        with self.assertRaises(ReplayError):validate(self.manifest)
    def test_bad_scratch_policy(self):
        self.m['nodes'][0]['scratch_policy']='guess';self.manifest.write_text(json.dumps(self.m))
        with self.assertRaises(ReplayError):validate(self.manifest)
    def numeric(self):
        m=copy.deepcopy(self.m);m['views'][1]['dtype']=1
        m['checks'][0].update(mode='float',atol=1e-5,rtol=1e-5,rel_l2=1e-5)
        p={'schema':'radiance-numeric-policy/1','version':'v1','rules':[{'op':'copy','dtype':1,'atol':1e-5,'rtol':1e-5,'rel_l2':1e-5,'nonfinite':'reject'}]}
        return m,p
    def test_numeric_policy_exact(self):m,p=self.numeric();numeric_policy(m,p)
    def test_numeric_policy_unknown_dtype(self):
        m,p=self.numeric();p['rules'][0]['dtype']=2
        with self.assertRaises(ReplayError):numeric_policy(m,p)
    def test_numeric_policy_relaxed_blocked(self):
        m,p=self.numeric();m['checks'][0]['rtol']=1e-4
        with self.assertRaises(ReplayError):numeric_policy(m,p)
    def test_numeric_policy_nonfinite_blocked(self):
        m,p=self.numeric();m['checks'][0]['nonfinite']='match'
        with self.assertRaises(ReplayError):numeric_policy(m,p)
    def test_numeric_policy_duplicate_blocked(self):
        m,p=self.numeric();p['rules']*=2
        with self.assertRaises(ReplayError):numeric_policy(m,p)
    def test_numeric_policy_missing_version(self):
        m,p=self.numeric();del p['version']
        with self.assertRaises(ReplayError):numeric_policy(m,p)
    def test_workload_hash_ignores_binary_only(self):
        p=self.root/'new.so';p.write_bytes(b'new');c=candidate(self.manifest,self.root/'candidate',['p0='+str(p)])
        self.assertEqual(workload_identity(c),workload_identity(self.manifest))
    def test_workload_hash_tolerance_detected(self):
        self.m['views'][1].update(dtype=1,shape=[1],stride=[1]);self.m['checks'][0]['mode']='float';self.manifest.write_text(json.dumps(self.m))
        a=workload_identity(self.manifest);self.m['checks'][0]['atol']=1e-5;self.manifest.write_text(json.dumps(self.m))
        self.assertNotEqual(a,workload_identity(self.manifest))
    def test_ab_summary_known_ratio(self):
        r=summarize([{'a_us':[10,10],'b_us':[5,5]}]*4);self.assertEqual(r['ratio'],.5);self.assertEqual(r['bootstrap_interval_95'],[.5,.5])
    def test_ab_summary_small_rejected(self):
        with self.assertRaises(ReplayError):summarize([{'a_us':[10,10],'b_us':[5,5]}]*2)
    def test_ab_summary_negative_rejected(self):
        with self.assertRaises(ReplayError):summarize([{'a_us':[0,10],'b_us':[5,5]}]*3)
    def abargs(self):
        return SimpleNamespace(out=str(self.root/'ab'),baseline=str(self.manifest),candidate=str(self.manifest),
            baseline_worker=str(self.worker),candidate_worker=str(self.worker),baseline_plugin=['p0='+str(self.plugin)],candidate_plugin=['p0='+str(self.plugin)],
            execute_trusted_code=True,blocks=3,warmups=1,reps=3,timeout=10,device=0,gate=False)
    def invoke(self,cmd):
        out=Path(cmd[cmd.index('--out')+1]);out.mkdir();r=copy.deepcopy(self.r)
        r['runtime']['device']={'arch':'host','name':'unit-test'};r['runtime']['measurement']='host window'
        (out/'report.json').write_text(json.dumps(r));return 0
    def test_ab_order_and_attempts(self):
        code,r=experiment(self.abargs(),self.invoke);self.assertEqual(code,0);self.assertEqual(''.join(x['arm'] for x in r['runs']),'abba'*3)
    def test_ab_stops_first_failure(self):
        def invoke(cmd):
            self.invoke(cmd);p=Path(cmd[cmd.index('--out')+1])/'report.json';r=read_json(p);r['status']='FAIL';p.write_text(json.dumps(r));return 1
        code,r=experiment(self.abargs(),invoke);self.assertEqual(code,1);self.assertEqual(len(r['runs']),1)
    def test_ab_instrumented_blocked(self):
        self.r['runtime']['worker_build']={'host_asan':True}
        code,r=experiment(self.abargs(),self.invoke);self.assertEqual(code,2)
    def test_oracle_basic(self):
        self.assertEqual(gemm_nt([1.,2.,3.,4.],[1.,0.,0.,1.],2,2,2),[1.,2.,3.,4.])
    def test_boundary_values(self):self.assertEqual(boundaries(16),[1,15,16,17,31,32,33])
