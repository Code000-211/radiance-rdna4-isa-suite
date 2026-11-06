import unittest
from radisa import vopd
from radisa.binary import FormatError
from support import vopd_bytes

class VOPDTests(unittest.TestCase):
    def codes(self,raw,**kw):return {r for r,_ in vopd.validate(vopd.decode(raw),kw.get('wave',32),kw.get('allocated',256))}
    def test_legal_pair(self):self.assertEqual(self.codes(vopd_bytes()),set())
    def test_binary_fields(self):
        p=vopd.decode(vopd_bytes(dx=34,dy=75,sx=279,sy=277,vx=18,vy=39))
        self.assertEqual((p.x.dst,p.y.dst,p.x.src0,p.y.src0,p.x.src1,p.y.src1),(34,75,279,277,18,39))
    def test_wave64_rejected(self):self.assertIn('WAVE32',self.codes(vopd_bytes(),wave=64))
    def test_src0_same_bank(self):self.assertIn('BANK',self.codes(vopd_bytes(sx=260,sy=264)))
    def test_src0_identical_exception(self):self.assertNotIn('BANK',self.codes(vopd_bytes(sx=260,sy=260)))
    def test_src1_same_bank(self):self.assertIn('BANK',self.codes(vopd_bytes(vx=0,vy=4)))
    def test_src1_identical_exception(self):self.assertNotIn('BANK',self.codes(vopd_bytes(vx=0,vy=0)))
    def test_y_mov_port_reroute(self):self.assertEqual(self.codes(vopd_bytes(x=8,y=8,sx=260,sy=264)),set())
    def test_fmamk_port2_parity(self):
        self.assertIn('SRC2',self.codes(vopd_bytes(x=2,y=2,vx=0,vy=2,literal=0)))
    def test_fmamk_good_port2_parity(self):
        self.assertNotIn('SRC2',self.codes(vopd_bytes(x=2,y=2,vx=0,vy=1,literal=0)))
    def test_accum_dest_ports(self):
        self.assertEqual(self.codes(vopd_bytes(x=0,y=0)),set())
    def test_x_writes_y_input(self):self.assertIn('ANTIDEP',self.codes(vopd_bytes(dx=6,sy=262)))
    def test_x_writes_y_input_dpmacc_is_error(self):self.assertIn('RAW_DPMACC',self.codes(vopd_bytes(x=12,dx=6,sy=262)))
    def test_y_writes_x_input_allowed(self):self.assertNotIn('ANTIDEP',self.codes(vopd_bytes(sx=263)))
    def test_unused_mov_src1_ignored(self):
        self.assertEqual(self.codes(vopd_bytes(x=8,y=8,vx=255,vy=255),allocated=8),set())
    def test_dest_range(self):self.assertIn('REGISTER_RANGE',self.codes(vopd_bytes(),allocated=4))
    def test_dpp_prohibited(self):self.assertIn('DPP',self.codes(vopd_bytes(sx=233)))
    def test_reserved_source(self):self.assertIn('SOURCE',self.codes(vopd_bytes(sx=230)))
    def test_vcc_hi_high_half_is_review(self):self.assertIn('SOURCE_HI',self.codes(vopd_bytes(sx=107)))
    def test_exec_hi_high_half_is_review(self):self.assertIn('SOURCE_HI',self.codes(vopd_bytes(sx=127)))
    def test_scalar_bus_implicit_vcc(self):
        self.assertIn('SCALAR_BUS',self.codes(vopd_bytes(x=9,y=9,sx=0,sy=1)))
    def test_null_does_not_consume_constant_bus(self):
        self.assertNotIn('SCALAR_BUS',self.codes(vopd_bytes(x=9,y=9,sx=124,sy=0)))
    def test_shared_scalar(self):self.assertNotIn('SCALAR_BUS',self.codes(vopd_bytes(x=9,y=9,sx=0,sy=0)))
    def test_literal_size(self):self.assertEqual(vopd.decode(vopd_bytes(sx=255,literal=123)).size,12)
    def test_literal_share(self):self.assertEqual(vopd.decode(vopd_bytes(sx=255,sy=255,literal=123)).literal,123)
    def test_fmaak_requires_literal(self):
        with self.assertRaises(FormatError):vopd.decode(vopd_bytes(x=1))
    def test_truncated_pair(self):
        with self.assertRaises(FormatError):vopd.decode(vopd_bytes()[:4])
    def test_not_vopd(self):
        with self.assertRaises(FormatError):vopd.decode(bytes(8))
    def test_reserved_x_opcode(self):self.assertIn('OPCODE',self.codes(vopd_bytes(x=14)))
    def test_reserved_y_opcode(self):self.assertIn('OPCODE',self.codes(vopd_bytes(y=31)))
    def test_x_extension_not_legal(self):self.assertIn('OPCODE',self.codes(vopd_bytes(x=15)))
    def test_all_supported_opcode_pairs(self):
        # Enumerate every X/Y opcode combination; operands selected independently
        # of the validator. Some pairs are intentionally rejected for hazards.
        n=0
        for x in range(14):
            for y in vopd.OPS:
                lit=0 if x in (1,2) or y in (1,2) else None
                p=vopd.decode(vopd_bytes(x=x,y=y,literal=lit))
                self.assertEqual((p.x.op,p.y.op),(x,y));vopd.validate(p,32,256);n+=1
        self.assertEqual(n,238)
    def test_all_destination_encodings(self):
        for x in range(256):
            for h in range(128):
                y=2*h+((x&1)^1)
                p=vopd.decode(vopd_bytes(dx=x,dy=y))
                self.assertEqual((p.x.dst,p.y.dst),(x,y))
                self.assertEqual((p.x.dst^p.y.dst)&1,1)
