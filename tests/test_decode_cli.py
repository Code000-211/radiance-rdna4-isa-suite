import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from radisa.binary import ELF,FormatError,kernel_inventory,sha256
from radisa.decode import parse_objdump,decode_object,select_backend
from radisa.cli import main,read_json,write_json,runtime_reconcile,classify,safe_child
from support import ROOT,FIXTURE,bundle,archive,vopd_bytes,replace_function

class DisassemblyTests(unittest.TestCase):
    def test_amd_trailing_address(self):
        d=parse_objdump(' s_endpgm // 000000001904: BFB00000\n');self.assertEqual(d[0x1904][0],bytes.fromhex('0000b0bf'))
    def test_regular_address_bytes(self):
        d=parse_objdump('1904: 00 00 b0 bf s_endpgm\n');self.assertIn(0x1904,d)
    def test_template_symbol_not_boundary_parser(self):
        d=parse_objdump('1900 <kernel<A<B>,C>>:\n s_endpgm // 1900: BFB00000');self.assertEqual(len(d),1)
    def test_duplicate_addresses_conflict(self):
        with self.assertRaises(FormatError):parse_objdump('s_endpgm // 0: BFB00000\nv_nop // 0: 7E000280')
    def test_branch_target_annotation(self):
        # llvm-objdump appends the branch target after the words; such a line was dropped, so every
        # kernel with a branch came out as `instruction absent from objdump output` with --objdump.
        d=parse_objdump('\ts_cbranch_scc0 7    // 000000000000: BFA10007 <.text+0x20>\n'
                        '\ts_branch 1 // 000000000004: BFA00001 <kernel<A<B>,C>+0x10>\n')
        self.assertEqual((d[0],d[4]),((bytes.fromhex('0700a1bf'),'s_cbranch_scc0 7'),(bytes.fromhex('0100a0bf'),'s_branch 1')))
    def test_unrecognized_line_not_invented(self):self.assertEqual(parse_objdump('this is not disassembly'),{})
    def test_exclusive_backend_options(self):
        with self.assertRaises(FormatError):select_backend('a','b')
    def test_missing_objdump(self):
        with self.assertRaises(FormatError):select_backend(objdump='/does/not/exist')

class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name)
        (self.p/'input.hsaco').write_bytes(FIXTURE)
    def tearDown(self):self.temp.cleanup()
    def run_cli(self,*args):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return main([str(a)for a in args])
    def collect(self,path=None,name='ev'):
        c=self.run_cli('collect',path or self.p/'input.hsaco','--out',self.p/name)
        self.assertEqual(c,0);return self.p/name/'manifest.json'
    def backend(self):
        try:return select_backend()
        except FormatError:
            if os.environ.get('RAD_ISA_REQUIRE_LLVM')=='1':raise
            self.skipTest('AMDGPU LLVM decoder unavailable; set RAD_ISA_REQUIRE_LLVM=1 to require it')
    def audit(self,m,name='report',extra=()):
        b=self.backend();flag='--llvm-library' if b['kind']=='library' else '--objdump'
        return self.run_cli('audit',m,'--out',self.p/name,flag,b['path'],*extra)
    def test_collect_real_linked(self):
        m=read_json(self.collect());self.assertEqual(m['expected_kernels'],1);self.assertTrue(m['collection_complete'])
    def test_real_cpu_decode_pass(self):
        self.assertEqual(self.audit(self.collect()),0)
        r=read_json(self.p/'report/report.json');self.assertFalse(r['full_isa_certification'])
        self.assertEqual(r['coverage']['decoded_bytes'],r['coverage']['expected_bytes'])
        self.assertEqual(r['coverage']['fully_decoded_kernels'],1)
        self.assertTrue((self.p/'report/junit.xml').exists())
    def test_real_cpu_decode_bad_vopd_fails(self):
        e=ELF(FIXTURE);c=e.symbol_bytes(e.named_symbol('smoke'))
        # Replace the MOV/MOV with an intentionally invalid ADD/ADD SRC0-bank pair.
        # Never execute this fixture on a GPU.
        bad=replace_function(FIXTURE,c[:8]+vopd_bytes(sx=260,sy=256)+c[16:])
        (self.p/'input.hsaco').write_bytes(bad)
        self.assertEqual(self.audit(self.collect()),1)
        r=read_json(self.p/'report/report.json')
        self.assertIn('VOPD_BANK',{f['rule']for k in r['kernels']for f in k['findings']})
    def test_empty_directory_is_incomplete(self):
        (self.p/'empty').mkdir();self.assertEqual(self.run_cli('collect',self.p/'empty','--out',self.p/'ev'),2)
    def test_missing_input_is_incomplete(self):self.assertEqual(self.run_cli('collect',self.p/'missing','--out',self.p/'ev'),2)
    def test_stale_output_not_reused(self):
        self.collect();self.assertEqual(self.run_cli('collect',self.p/'input.hsaco','--out',self.p/'ev'),2)
    def test_corrupt_code_object_not_silent(self):
        (self.p/'bad.hsaco').write_bytes(b'\x7fELFbroken')
        self.assertEqual(self.run_cli('collect',self.p/'bad.hsaco','--out',self.p/'ev'),2)
    def test_compressed_bundle_not_silent(self):
        (self.p/'compressed').write_bytes(b'CCOBjunk')
        self.assertEqual(self.run_cli('collect',self.p/'compressed','--out',self.p/'ev'),2)
    def test_archive_and_bundle(self):
        b=bundle([('hip-amdgcn-amd-amdhsa--gfx1201',FIXTURE)])
        (self.p/'library.a').write_bytes(archive([('gpu.o',b)]))
        self.assertEqual(read_json(self.collect(self.p/'library.a'))['expected_kernels'],1)
    def test_external_file_symlink_not_lost(self):
        inp=self.p/'links';inp.mkdir();(inp/'gpu.so').symlink_to(self.p/'input.hsaco')
        self.assertEqual(read_json(self.collect(inp))['expected_kernels'],1)
    def test_symlink_directory_not_silently_ignored(self):
        inp=self.p/'links';inp.mkdir();(inp/'outside').symlink_to(self.p,target_is_directory=True)
        self.assertEqual(self.run_cli('collect',inp,'--out',self.p/'ev'),2)
    def test_content_dedup_preserves_origins(self):
        (self.p/'duplicate.co').write_bytes(FIXTURE)
        m=read_json(self.collect(self.p));self.assertEqual(m['expected_kernels'],1)
        self.assertEqual(len(m['objects'][0]['origins']),2)
    def test_same_name_different_objects_not_overwritten(self):
        e=ELF(FIXTURE);c=bytearray(e.symbol_bytes(e.named_symbol('smoke')))
        c[12]=171 # v_dual_mov source inline 43 instead of 42
        (self.p/'other.hsaco').write_bytes(replace_function(FIXTURE,bytes(c)))
        m=self.collect(self.p);self.assertEqual(read_json(m)['expected_kernels'],2)
        self.assertEqual(self.audit(m),0)
        self.assertEqual(len(read_json(self.p/'report/report.json')['kernels']),2)
    def test_hash_tampering_fails_closed(self):
        m=self.collect();j=read_json(m);p=m.parent/j['objects'][0]['path'];p.write_bytes(p.read_bytes()+b'x')
        self.assertEqual(self.audit(m),2)
    def test_missing_object_fails_closed(self):
        m=self.collect();j=read_json(m);(m.parent/j['objects'][0]['path']).unlink()
        self.assertEqual(self.audit(m),2)
    def test_removed_kernel_manifest_detected(self):
        m=self.collect();j=read_json(m);j['objects'][0]['kernels']=[];j['expected_kernels']=0;write_json(m,j)
        self.assertEqual(self.audit(m),2)
    def test_bad_decoder_tool_fails_closed(self):
        m=self.collect();exe=self.p/'false-objdump';exe.write_text('#!/bin/sh\necho error >&2\nexit 1\n');exe.chmod(0o700)
        self.assertEqual(self.run_cli('audit',m,'--out',self.p/'report','--objdump',exe),2)
    def test_empty_decoder_output_fails_closed(self):
        m=self.collect();exe=self.p/'empty-objdump';exe.write_text('#!/bin/sh\nexit 0\n');exe.chmod(0o700)
        self.assertEqual(self.run_cli('audit',m,'--out',self.p/'report','--objdump',exe),2)
    def test_bad_library_fails_closed(self):
        m=self.collect();self.assertEqual(self.run_cli('audit',m,'--out',self.p/'report','--llvm-library','/missing.so'),2)
    def test_worker_timeout(self):
        m=self.collect();exe=self.p/'sleep-objdump';exe.write_text('#!/bin/sh\nexec sleep 5\n');exe.chmod(0o700)
        self.assertEqual(self.run_cli('audit',m,'--out',self.p/'report','--objdump',exe,'--timeout','.01'),2)
    def test_expected_baseline(self):
        m=self.collect();self.assertEqual(self.run_cli('baseline',m,'--out',self.p/'expected.json'),0)
        self.assertEqual(self.audit(m,extra=('--expect',self.p/'expected.json')),0)
    def test_baseline_missing_identity(self):
        m=self.collect();write_json(self.p/'expected.json',{'identities':['0'*64+':missing.kd']})
        self.assertEqual(self.audit(m,extra=('--expect',self.p/'expected.json')),2)
    def test_json_duplicate_rejected(self):
        p=self.p/'bad.json';p.write_text('{"x":1,"x":2}')
        with self.assertRaises(FormatError):read_json(p)
    def test_path_escape_rejected(self):
        with self.assertRaises(FormatError):safe_child(self.p,'../secret')
    def test_symlink_escape_rejected(self):
        (self.p/'link').symlink_to('/etc')
        with self.assertRaises(FormatError):safe_child(self.p,'link/passwd')
    def test_unlisted_object_not_ignored(self):
        m=self.collect();(m.parent/'objects'/('0'*64+'.hsaco')).write_bytes(FIXTURE)
        self.assertEqual(self.audit(m),2)
    def test_bad_manifest_count(self):
        m=self.collect();j=read_json(m);j['expected_kernels']=9;write_json(m,j)
        self.assertEqual(self.audit(m),2)
    def test_capture_matches_hash_and_kernel(self):
        m=read_json(self.collect());p=self.p/'capture.jsonl'
        p.write_text(json.dumps({'event':'resolve','file':'input.hsaco','bytes':len(FIXTURE),'sha256':sha256(FIXTURE),'kernel':'smoke'})+'\n')
        r=runtime_reconcile(p,m);self.assertFalse(r['errors']);self.assertEqual(r['matched'],1)
    def test_capture_missing_kernel(self):
        m=read_json(self.collect());p=self.p/'capture.jsonl'
        p.write_text(json.dumps({'event':'resolve','file':'input.hsaco','bytes':len(FIXTURE),'sha256':sha256(FIXTURE),'kernel':'missing'})+'\n')
        self.assertTrue(runtime_reconcile(p,m)['errors'])
    def test_capture_same_size_tampering(self):
        m=read_json(self.collect());p=self.p/'capture.jsonl'
        p.write_text(json.dumps({'event':'resolve','file':'input.hsaco','bytes':len(FIXTURE),'sha256':'0'*64,'kernel':'smoke'})+'\n')
        self.assertTrue(runtime_reconcile(p,m)['errors'])
    def test_capture_truncation(self):
        m=read_json(self.collect());p=self.p/'capture.jsonl';p.write_text('{')
        self.assertTrue(runtime_reconcile(p,m)['errors'])
    def test_errors_have_priority(self):self.assertEqual(classify([{'severity':'error'},{'severity':'incomplete'}]),('FAIL',1))
    def test_strict_review_blocks(self):self.assertEqual(classify([{'severity':'review'}],True),('INCOMPLETE',2))
