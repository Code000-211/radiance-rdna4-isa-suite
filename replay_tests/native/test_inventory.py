"""Real native metadata inventory; not dispatch or GPU coverage."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
from radreplay.format import file_hash
class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.exe=Path(os.environ['RAD_REPLAY_TEST_BUILD'])/'rad-replay-inventory'
        self.plugin=Path(os.environ['RAD_REPLAY_TEST_LIBREF'])
        self.assertTrue(self.exe.is_file(),'native inventory is required, not skipped')
    def tearDown(self):self.tmp.cleanup()
    def test_actual_libref_rows(self):
        out=self.root/'rows.json';r=subprocess.run([str(self.exe),str(self.plugin),str(out)],capture_output=True,timeout=20)
        self.assertEqual(r.returncode,0,r.stderr);j=json.loads(out.read_text())
        self.assertEqual(j['plugin_sha256'],file_hash(self.plugin));self.assertEqual(j['declared_row_count'],len(j['rows']))
        self.assertGreater(j['declared_row_count'],1);self.assertIn('ref_gemm_nt',{x['name'] for x in j['rows']});self.assertFalse(j['runtime_executed'])
    def test_output_overwrite_refused(self):
        out=self.root/'rows.json';out.write_text('keep')
        r=subprocess.run([str(self.exe),str(self.plugin),str(out)],capture_output=True,timeout=20)
        self.assertEqual(r.returncode,2);self.assertEqual(out.read_text(),'keep')
    def test_not_a_plugin_refused(self):
        p=self.root/'fake.so';p.write_bytes(b'not an ELF')
        r=subprocess.run([str(self.exe),str(p),str(self.root/'rows.json')],capture_output=True,timeout=20)
        self.assertEqual(r.returncode,2);self.assertFalse((self.root/'rows.json').exists())
