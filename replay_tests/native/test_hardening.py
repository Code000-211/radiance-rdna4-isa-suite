"""v0.3 blind spots as mandatory host mutation tests. No GPU execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from radreplay.builder import Package
from radreplay.cli import main

class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.build=Path(os.environ['RAD_REPLAY_TEST_BUILD']);self.worker=self.build/'rad-kbench-replay';self.plugin=self.build/'replay_test.so'
    def tearDown(self):self.tmp.cleanup()
    def package(self,row='copy',*,initial=b'abcd',scratch=0,policy=None,masked=False,alias=False):
        p=Package(self.root/'case','hardening-'+row,note='HOST-only mutation test; copy contract oracle.')
        p.plugin('replay_test',self.plugin);p.stream()
        p.allocation('x',b'abcd',mutable=masked or alias)
        if not alias:p.allocation('y',initial,mutable=True)
        p.view('x','x',nbytes=4,dtype=9,shape=[4]);p.view('y','x' if alias else 'y',nbytes=4,dtype=9,shape=[4])
        p.kernel('tested','p0',row,'replay_probe',['x','y'],[('x','r'),('y','w')],scratch_bytes=scratch,scratch_policy=policy)
        if masked:
            p.allocation('z',b'abcd');p.view('z','z',nbytes=4,dtype=9,shape=[4])
            p.kernel('later','p0','copy','replay_probe',['z','x'],[('z','r'),('x','w')]);p.check('x',b'abcd')
        p.check('y',b'abcd');return p
    def run_case(self,p,flags=()):
        path=p.finish()
        with contextlib.redirect_stdout(io.StringIO()):
            rc=main(['run',str(path),'--worker',str(self.worker),'--plugin','p0='+str(self.plugin),
                     '--out',str(self.root/'result'),'--execute-trusted-code','--warmups','1','--reps','3',*flags])
        r=json.loads((self.root/'result/report.json').read_text());return rc,r
    def test_no_store_caught_despite_matching_initial_output(self):
        rc,r=self.run_case(self.package('audit_no_store'));self.assertEqual(rc,1,r)
        self.assertIn('invariance',r['runtime']['error'])
    def test_partial_store_caught_despite_matching_initial_tail(self):
        rc,r=self.run_case(self.package('partial_store'));self.assertEqual(rc,1,r)
    def test_scratch_read_before_initialization_caught(self):
        rc,r=self.run_case(self.package('audit_scratch_read',scratch=16,policy='kernel-initialized'))
        self.assertEqual(rc,1,r);self.assertIn('invariance',r['runtime']['error'])
    def test_zeroed_scratch_is_valid_only_with_explicit_contract(self):
        rc,r=self.run_case(self.package('audit_scratch_read',scratch=16,policy='zeroed'));self.assertEqual(rc,0,r)
    def test_unknown_scratch_contract_never_green(self):
        rc,r=self.run_case(self.package('audit_scratch_read',scratch=16));self.assertEqual(rc,2,r)
    def test_kernel_initialized_scratch_good_control(self):
        rc,r=self.run_case(self.package('scratch_overwrite',scratch=16,policy='kernel-initialized'));self.assertEqual(rc,0,r)
    def test_later_write_does_not_hide_earlier_readonly_write(self):
        rc,r=self.run_case(self.package('mutate',masked=True));self.assertEqual(rc,1,r)
        self.assertIn('operation write contract',r['runtime']['error'])
    def test_legitimate_later_write_is_not_rejected(self):
        rc,r=self.run_case(self.package('copy',masked=True));self.assertEqual(rc,0,r)
    def test_readable_alias_not_poisoned(self):
        rc,r=self.run_case(self.package(alias=True));self.assertEqual(rc,0,r)
        self.assertEqual(r['runtime']['diagnostics']['poisoned_output_bytes_total'],0)
    def test_diagnostic_only_has_no_timing(self):
        rc,r=self.run_case(self.package(),['--diagnostic-only']);self.assertEqual(rc,0,r)
        self.assertEqual(r['runtime']['samples_us'],[]);self.assertEqual(r['runtime']['checked_iterations'],0)
    def test_three_diagnostics_and_four_ordinary_checks(self):
        rc,r=self.run_case(self.package());self.assertEqual(rc,0,r)
        self.assertEqual(r['runtime']['checked_iterations'],4);self.assertEqual(r['runtime']['diagnostics']['completed_passes'],3)
    def test_disabled_diagnostics_are_visible(self):
        rc,r=self.run_case(self.package('audit_no_store'),['--diagnostics','none']);self.assertEqual(rc,0,r)
        self.assertEqual(r['runtime']['diagnostics']['status'],'NOT_RUN')
    def test_explicit_checkpoint_good(self):
        p=self.package();p.checkpoint('tested','y',b'abcd');rc,r=self.run_case(p);self.assertEqual(rc,0,r)
        self.assertEqual(r['runtime']['diagnostics']['explicit_checkpoints'],1)
    def test_explicit_checkpoint_wrong(self):
        p=self.package();p.checkpoint('tested','y',b'wxyz');rc,r=self.run_case(p);self.assertEqual(rc,1,r)
    def test_checkpoint_cannot_be_silently_ignored(self):
        p=self.package();p.checkpoint('tested','y',b'abcd');rc,r=self.run_case(p,['--diagnostics','none']);self.assertEqual(rc,2,r)
    def test_noninstrumented_worker_refuses_read_check(self):
        rc,r=self.run_case(self.package(),['--memory-check','host-asan','--diagnostic-only']);self.assertEqual(rc,2,r)
    def test_guards_do_not_claim_read_detection(self):
        rc,r=self.run_case(self.package('audit_oob_read'));self.assertEqual(rc,0,r)
        self.assertEqual(r['runtime']['diagnostics']['read_checks'],'NOT_CHECKED')
    def test_long_explicit_inplace_state_sequence(self):
        lib=Path(os.environ['RAD_REPLAY_TEST_LIBREF']);p=Package(self.root/'state','state-sequence',note='32 explicit in-place adds, independent integer oracle.')
        p.plugin('libref',lib);p.stream()
        for key,vals,mut in [('state',[1,2,3,4],True),('inc',[1,1,1,1],False)]:
            p.allocation(key,struct.pack('<4f',*vals),mutable=mut);p.view(key,key,nbytes=16,dtype=1,shape=[1,4])
        for i in range(32):
            node=f'step-{i}';p.kernel(node,'p0','ref_add','add',['state','inc','state'],[('state','rw'),('inc','r')],params={'M':1,'n':4,'dtype':'f32'})
            p.checkpoint(node,'state',struct.pack('<4f',*[j+i+1 for j in [1,2,3,4]]))
        p.check('state',struct.pack('<4f',33,34,35,36));self.plugin=lib
        rc,r=self.run_case(p);self.assertEqual(rc,0,r);self.assertEqual(r['runtime']['diagnostics']['explicit_checkpoints'],32)
