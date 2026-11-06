"""Instrumented host mutation tests; no automatic skip and no GPU code."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from radreplay.builder import Package
from radreplay.cli import main

class ASanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.build=Path(os.environ['RAD_REPLAY_ASAN_BUILD']);self.worker=self.build/'rad-kbench-replay';self.plugin=self.build/'replay_test.so'
    def tearDown(self):self.tmp.cleanup()
    def run_probe(self,row,*,allocation_bytes=4,offset=0,span=4,extra_flags=()):
        p=Package(self.root/'case','asan-'+row);p.plugin('replay_test',self.plugin);p.stream()
        raw=b'abcdefgh';p.allocation('x',raw[:allocation_bytes]);p.allocation('y',bytes(span),mutable=True)
        p.view('x','x',offset=offset,nbytes=span,dtype=9,shape=[span]);p.view('y','y',nbytes=span,dtype=9,shape=[span])
        p.kernel('tested','p0',row,'replay_probe',['x','y'],[('x','r'),('y','w')]);p.check('y',raw[offset:offset+span])
        with contextlib.redirect_stdout(io.StringIO()):
            rc=main(['run',str(p.finish()),'--worker',str(self.worker),'--plugin','p0='+str(self.plugin),'--out',str(self.root/'result'),
                     '--execute-trusted-code','--memory-check','host-asan','--diagnostic-only',*extra_flags])
        r=json.loads((self.root/'result/report.json').read_text());return rc,r
    def test_valid_copy(self):
        rc,r=self.run_probe('copy');self.assertEqual(rc,0,r)
        self.assertEqual(r['runtime']['diagnostics']['read_checks'],'HOST_ASAN_INSTRUMENTED_PATHS')
        self.assertEqual(r['runtime']['samples_us'],[])
    def test_one_byte_overread_is_detected(self):
        rc,r=self.run_probe('audit_oob_read');self.assertEqual(rc,1,r)
        self.assertEqual(r['runtime']['memory_check'],'HOST_ASAN_DETECTED_ERROR')
        self.assertIn('READ of size 1',(self.root/'result/worker.stderr').read_text())
    def test_read_past_view_inside_larger_allocation_is_detected(self):
        rc,r=self.run_probe('audit_oob_read',allocation_bytes=8);self.assertEqual(rc,1,r)
    def test_prefix_overread_is_detected(self):
        rc,r=self.run_probe('prefix_read');self.assertEqual(rc,1,r)
    def test_redzone_store_is_detected(self):
        rc,r=self.run_probe('guard');self.assertEqual(rc,1,r)
        self.assertIn('WRITE of size 1',(self.root/'result/worker.stderr').read_text())
    def test_missing_stores_still_detected(self):
        rc,r=self.run_probe('audit_no_store');self.assertEqual(rc,1,r)
    def test_unrepresentable_shadow_boundary_is_incomplete(self):
        rc,r=self.run_probe('copy',allocation_bytes=8,offset=1);self.assertEqual(rc,2,r)
        self.assertIn('8-byte aligned',r['error'])
    def test_asan_cannot_produce_benchmark(self):
        rc,r=self.run_probe('copy',extra_flags=['--diagnostics','none']);self.assertEqual(rc,2,r)
