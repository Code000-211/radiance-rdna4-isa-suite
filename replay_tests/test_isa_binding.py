import copy
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from radisa.cli import collect, audit
from radreplay.cli import isa_binding
from radreplay.format import ReplayError, file_hash, read_json
import json
import contextlib
import io

class ISABindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.smoke=Path(__file__).resolve().parents[1]/'fixtures'/'smoke.hsaco'
        with contextlib.redirect_stdout(io.StringIO()):
            rc=collect(Namespace(inputs=[str(cls.smoke)],out=str(cls.root/'evidence')))
            if rc:raise RuntimeError('real gfx1201 fixture collection failed')
            # Invoke public CLI so all audit option defaults match the production tool.
            from radisa.cli import main
            rc=main(['audit',str(cls.root/'evidence'/'manifest.json'),'--out',str(cls.root/'analysis')])
            if rc:raise RuntimeError('real gfx1201 static audit failed')
        cls.m=cls.root/'evidence'/'manifest.json';cls.r=cls.root/'analysis'/'report.json'
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_real_evidence_binds(self):
        h=file_hash(self.smoke);self.assertEqual(isa_binding(self.m,self.r,{h},{h:self.smoke})['status'],'STATIC_CHECKS_PASS')
    def test_unlisted_plugin(self):
        with self.assertRaises(ReplayError):isa_binding(self.m,self.r,{'0'*64})
    def test_stale_report(self):
        with tempfile.TemporaryDirectory() as d:
            m=read_json(self.m);m['extra']='change';p=Path(d)/'manifest.json';p.write_text(json.dumps(m))
            with self.assertRaises(ReplayError):isa_binding(p,self.r,{file_hash(self.smoke)})
    def test_input_membership_without_code_is_not_enough(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'host.so';p.write_bytes(b'not a GPU container');h=file_hash(p)
            m=read_json(self.m);m['inputs'].append({'path':str(p),'sha256':h,'bytes':p.stat().st_size})
            # Keep object paths valid while producing an alternate manifest in the evidence directory.
            alt=self.m.parent/'binding-negative.json';alt.write_text(json.dumps(m))
            r=read_json(self.r);r['manifest_sha256']=file_hash(alt);rp=Path(d)/'report.json';rp.write_text(json.dumps(r))
            with self.assertRaises(ReplayError):isa_binding(alt,rp,{h},{h:p})
            alt.unlink()
