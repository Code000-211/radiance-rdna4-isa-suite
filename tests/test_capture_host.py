import json
import hashlib
import random
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from radisa.binary import extract_code_objects
from support import ROOT,FIXTURE,bundle

class HostCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=shutil.which('g++')or shutil.which('clang++')
        if not cls.compiler:
            if os.getenv('RAD_ISA_REQUIRE_CXX')=='1':raise RuntimeError('C++ compiler required')
            raise unittest.SkipTest('no C++ compiler for optional integration tests')
        cls.t=tempfile.TemporaryDirectory();cls.root=Path(cls.t.name);cls.exe=cls.root/'capture'
        subprocess.run([cls.compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pthread',
                        '-I',str(ROOT/'integration'),str(ROOT/'tests/capture_harness.cpp'),'-o',str(cls.exe)],check=True,capture_output=True,text=True)
    @classmethod
    def tearDownClass(cls):cls.t.cleanup()
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name);self.cap=self.p/'capture';self.cap.mkdir(mode=0o700)
    def tearDown(self):self.t.cleanup()
    def run_capture(self,directory=None,mode=None):
        env=dict(os.environ);env.pop('RAD_ISA_CAPTURE_DIR',None)
        if directory is not None:env['RAD_ISA_CAPTURE_DIR']=str(directory)
        cmd=[str(self.exe),str(ROOT/'fixtures/smoke.hsaco')]+([mode]if mode else[])
        return subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=10)
    def test_sha256_matches_hashlib_at_padding_boundaries(self):
        rng=random.Random(1201)
        for size in (0,1,3,55,56,63,64,65,119,120,127,128,1000,65536):
            b=rng.randbytes(size);p=self.p/'input.bin';p.write_bytes(b)
            run=subprocess.run([str(self.exe),str(p),'hash'],capture_output=True,text=True,check=True,timeout=10)
            self.assertEqual(run.stdout.strip(),hashlib.sha256(b).hexdigest())
    def test_disabled_has_no_files(self):
        r=self.run_capture();self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(list(self.cap.iterdir()),[])
    def test_capture_exact_bytes_and_name(self):
        r=self.run_capture(self.cap);self.assertEqual(r.returncode,0,r.stderr)
        records=json.loads(next(self.cap.glob('resolved-*.jsonl')).read_text())
        self.assertEqual(records['kernel'],'smoke');self.assertEqual((self.cap/records['file']).read_bytes(),FIXTURE)
        self.assertEqual((self.cap/records['file']).stat().st_mode&0o777,0o600)
    def test_threaded_records_not_interleaved(self):
        r=self.run_capture(self.cap,'threads');self.assertEqual(r.returncode,0,r.stderr)
        lines=next(self.cap.glob('resolved-*.jsonl')).read_text().splitlines()
        self.assertEqual(len(lines),8)
        for l in lines:self.assertEqual(json.loads(l)['event'],'resolve')
    def test_missing_directory_fails(self):self.assertEqual(self.run_capture(self.p/'missing').returncode,2)
    def test_world_writable_directory_fails(self):
        self.cap.chmod(0o777);self.assertEqual(self.run_capture(self.cap).returncode,2)
    def test_symlink_directory_fails(self):
        link=self.p/'link';link.symlink_to(self.cap,target_is_directory=True)
        self.assertEqual(self.run_capture(link).returncode,2)
    def test_fork_rejected_without_locking(self):self.assertEqual(self.run_capture(self.cap,'fork').returncode,2)
    def test_real_host_hip_fatbin_extraction(self):
        b=bundle([('hip-amdgcn-amd-amdhsa--gfx1201',FIXTURE)])
        src=self.p/'embed.cpp';obj=self.p/'embed.o';so=self.p/'embed.so'
        src.write_text('__attribute__((section(".hip_fatbin"),used)) const unsigned char blob[]={'+','.join(str(x)for x in b)+'};')
        subprocess.run([self.compiler,'-fPIC','-c',str(src),'-o',str(obj)],check=True,capture_output=True)
        subprocess.run([self.compiler,'-shared',str(obj),'-o',str(so)],check=True,capture_output=True)
        for p in (obj,so):
            a,e=extract_code_objects(p.read_bytes(),str(p));self.assertEqual(len(a),1);self.assertEqual(a[0]['data'],FIXTURE)
