import random
import struct
import unittest
from pathlib import Path
from radisa.binary import ELF,FormatError,MessagePack,extract_code_objects,kernel_inventory,span,unpack_bundle,ar_members
from support import FIXTURE,ROOT,pack,bundle,archive

class MessagePackTests(unittest.TestCase):
    def test_nested_roundtrip(self):
        x={'amdhsa.kernels':[{'.args':[],'.name':'K<int, x<y>>','.wavefront_size':32}], 'a':True,'n':None,'v':[-33,-1,256,2**40]}
        self.assertEqual(MessagePack(pack(x)).decode(),x)
    def test_duplicate_key_rejected(self):
        with self.assertRaises(FormatError):MessagePack(b'\x82\xa1a\x01\xa1a\x02').decode()
    def test_trailing_rejected(self):
        with self.assertRaises(FormatError):MessagePack(b'\x01\x02').decode()
    def test_extension_rejected(self):
        with self.assertRaises(FormatError):MessagePack(b'\xd4\0\0').decode()
    def test_resource_depth(self):
        with self.assertRaises(FormatError):MessagePack(b'\x91'*66+b'\x00').decode()
    def test_array_count_bound(self):
        with self.assertRaises(FormatError):MessagePack(b'\xdd\xff\xff\xff\xff').decode()
    def test_invalid_utf8(self):
        with self.assertRaises(FormatError):MessagePack(b'\xa1\xff').decode()
    def test_truncations(self):
        b=pack({'hello':list(range(12))})
        for i in range(len(b)):
            with self.subTest(i=i),self.assertRaises(FormatError):MessagePack(b[:i]).decode()
    def test_deterministic_fuzz_1000(self):
        r=random.Random(1201)
        for _ in range(1000):
            b=r.randbytes(r.randrange(64))
            try:MessagePack(b).decode()
            except FormatError:pass

class ELFTests(unittest.TestCase):
    def test_real_kernel_inventory(self):
        k,err=kernel_inventory(ELF(FIXTURE));self.assertFalse(err);self.assertEqual(len(k),1)
        self.assertEqual(k[0]['name'],'smoke');self.assertFalse(k[0]['issues'])
    def test_metadata_args_before_name(self):
        m=ELF(FIXTURE).metadata()['amdhsa.kernels'][0]
        self.assertEqual(next(iter(m)),'.args');self.assertEqual(m['.wavefront_size'],32)
    def test_physical_not_descriptor_encoding(self):
        k,_=kernel_inventory(ELF(FIXTURE));self.assertEqual(k[0]['descriptor']['encoded_vgprs'],8)
    def test_padding_evidence(self):
        k,_=kernel_inventory(ELF(FIXTURE));self.assertTrue(k[0]['padding_file_backed'])
        self.assertIn(k[0]['padding_bytes_proved'],(0,256))
    def test_relocatable_is_incomplete(self):
        k,_=kernel_inventory(ELF((ROOT/'fixtures/smoke-relocatable.o').read_bytes()))
        self.assertTrue(any('relocatable' in m for m in k[0]['issues']))
    def test_wrong_endianness(self):
        b=bytearray(FIXTURE);b[5]=2
        with self.assertRaises(FormatError):ELF(bytes(b))
    def test_wrong_osabi(self):
        b=bytearray(FIXTURE);b[7]=0
        with self.assertRaises(FormatError):kernel_inventory(ELF(bytes(b)))
    def test_unknown_abi(self):
        b=bytearray(FIXTURE);b[8]=99
        with self.assertRaises(FormatError):kernel_inventory(ELF(bytes(b)))
    def test_wrong_target(self):
        b=bytearray(FIXTURE);b[48]=0x48
        with self.assertRaises(FormatError):kernel_inventory(ELF(bytes(b)))
    def test_truncated_header(self):
        for n in range(64):
            with self.subTest(n=n),self.assertRaises(FormatError):ELF(FIXTURE[:n])
    def test_truncated_section_table(self):
        with self.assertRaises(FormatError):ELF(FIXTURE[:-80])
    def test_sections_missing(self):
        b=bytearray(FIXTURE);b[40:48]=bytes(8)
        with self.assertRaises(FormatError):ELF(bytes(b))
    def test_negative_span(self):
        with self.assertRaises(FormatError):span(b'x',-1,1)
    def test_overrun_span(self):
        with self.assertRaises(FormatError):span(b'x',0,2)
    def test_named_symbol_missing(self):
        with self.assertRaises(FormatError):ELF(FIXTURE).named_symbol('missing')
    def test_descriptor_wave_mismatch(self):
        e=ELF(FIXTURE);s=e.named_symbol('smoke.kd');sec=e.sections[s.section]
        b=bytearray(FIXTURE);off=sec.offset+s.value-sec.address+56
        v=int.from_bytes(b[off:off+2],'little')^(1<<10);b[off:off+2]=v.to_bytes(2,'little')
        k,_=kernel_inventory(ELF(bytes(b)));self.assertIn('wave sizes disagree',' '.join(k[0]['issues']))

class ContainerTests(unittest.TestCase):
    def test_direct_object(self):
        a,b=extract_code_objects(FIXTURE,'one');self.assertEqual(len(a),1);self.assertEqual(b,[])
    def test_bundle_with_host(self):
        b=bundle([('host-x86_64',b'x'),('hip-amdgcn-amd-amdhsa--gfx1201',FIXTURE)])
        a,e=extract_code_objects(b,'bundle');self.assertEqual(len(a),1);self.assertEqual(e[0]['status'],'host_bundle')
    def test_bundle_family_label(self):
        # ROCm device libraries carry one object for the whole family ("gfx12").
        b=bundle([('hip-amdgcn-amd-amdhsa--gfx12',FIXTURE)])
        a,_=extract_code_objects(b,'bundle');self.assertEqual(len(a),1)
    def test_bundle_wrong_target_rejected(self):
        b=bundle([('hip-amdgcn-amd-amdhsa--gfx11',FIXTURE)])
        self.assertRaises(FormatError,extract_code_objects,b,'bundle')
    def test_generic_target_flags(self):
        # ROCm ships one object per generic family under a matching bundle label.
        # The name has to come out of e_flags, not out of the bundle.
        for flags,name in ((0x51,'gfx9-generic'),(0x52,'gfx10-1-generic'),(0x53,'gfx10-3-generic'),
                           (0x54,'gfx11-generic'),(0x59,'gfx12-generic'),(0x5f,'gfx9-4-generic')):
            data=bytearray(FIXTURE);data[0x30:0x34]=(flags).to_bytes(4,'little')
            self.assertEqual(ELF(bytes(data)).target,name)
            b=bundle([(f'hip-amdgcn-amd-amdhsa--{name}',bytes(data))])
            self.assertEqual(len(extract_code_objects(b,'bundle')[0]),1)
    def test_concatenated_bundles(self):
        b=bundle([('hip-amdgcn-amd-amdhsa--gfx1201',FIXTURE)])
        a,e=extract_code_objects(b+b'\0'*8+b,'both');self.assertEqual(len(a),2)
    def test_nested_archive(self):
        a,e=extract_code_objects(archive([('a',archive([('b',FIXTURE)]))]),'ar');self.assertEqual(len(a),1)
    def test_archive_alignment(self):
        a=list(ar_members(archive([('a',b'1'),('b',b'22')])));self.assertEqual(a,[('a',b'1'),('b',b'22')])
    def test_compressed_not_silent(self):
        a,e=extract_code_objects(b'CCOBjunk','x');self.assertFalse(a);self.assertEqual(e[0]['status'],'unsupported')
    def test_thin_archive_not_silent(self):
        a,e=extract_code_objects(b'!<thin>\n','x');self.assertFalse(a);self.assertEqual(e[0]['status'],'unsupported')
    def test_bundle_id_mismatch(self):
        b=bundle([('hip-amdgcn-amd-amdhsa--gfx1200',FIXTURE)])
        with self.assertRaises(FormatError):extract_code_objects(b,'bad')
    def test_duplicate_bundle_id(self):
        b=bundle([('one',b'x'),('one',b'y')])
        with self.assertRaises(FormatError):unpack_bundle(b)
    def test_overlap_bundle(self):
        b=bytearray(bundle([('one',b'x')]))
        struct.pack_into('<Q',b,32,0)
        with self.assertRaises(FormatError):unpack_bundle(b)
    def test_truncated_bundle(self):
        b=bundle([('one',b'x')])
        with self.assertRaises(FormatError):unpack_bundle(b[:-1])
    def test_nested_depth_limit(self):
        b=FIXTURE
        for _ in range(10):b=archive([('nested',b)])
        with self.assertRaises(FormatError):extract_code_objects(b,'deep')

class ArchiveLongNameTests(unittest.TestCase):
    def header(self,name,body):
        h=name.ljust(16).encode()+b'0'.ljust(12)+b'0'.ljust(6)+b'0'.ljust(6)+b'100644'.ljust(8)+str(len(body)).ljust(10).encode()+b'`\n'
        return h+body+(b'\n' if len(body)&1 else b'')
    def test_gnu_long_names(self):
        name='very_long_compiled_kernel_file.hip.o'
        data=b'!<arch>\n'+self.header('//',name.encode()+b'/\n')+self.header('/0',FIXTURE)
        result=list(ar_members(data));self.assertEqual(result,[(name,FIXTURE)])
    def test_bsd_long_names(self):
        name='another_long_kernel_file.hip.o'
        data=b'!<arch>\n'+self.header('#1/'+str(len(name)),name.encode()+FIXTURE)
        self.assertEqual(list(ar_members(data)),[(name,FIXTURE)])
    def test_bad_long_name_offset(self):
        data=b'!<arch>\n'+self.header('//',b'name/\n')+self.header('/999',FIXTURE)
        with self.assertRaises(FormatError):list(ar_members(data))
