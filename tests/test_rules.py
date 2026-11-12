import unittest
from radisa.rules import occupancy,regs,operands,cfg,local_checks,deallocation_checks,waterfall_checks
from radisa.binary import FormatError
from support import seq,instruction

class ResourceTests(unittest.TestCase):
    def test_wave32_boundaries(self):
        for n,w in [(1,16),(96,16),(97,12),(104,12),(120,12),(121,10),(128,10),(145,9),(241,5),(256,5)]:
            self.assertEqual(occupancy(n,32)['register_limited_waves_per_simd'],w)
    def test_wave64_capacity_not1536(self):
        self.assertEqual(occupancy(128,64)['register_limited_waves_per_simd'],5)
        self.assertEqual(occupancy(128,64)['wave_equivalent_capacity'],768)
    def test_zero_live_vgprs_still_allocate(self):self.assertEqual(occupancy(0,32)['physical_allocation'],24)
    def test_descriptor_reservation_honored(self):self.assertEqual(occupancy(96,32,104)['physical_allocation'],120)
    def test_unsupported_wave(self):
        with self.assertRaises(FormatError):occupancy(10,16)
    def test_invalid_count(self):
        with self.assertRaises(FormatError):occupancy(257,32)
    def test_no_clipping_264(self):self.assertEqual(occupancy(256,32)['physical_allocation'],264)

class OperandTests(unittest.TestCase):
    def test_register_ranges(self):self.assertEqual(regs('v[1:3], -s2'),{('v',1),('v',2),('v',3),('s',2)})
    def test_constants_not_registers(self):self.assertEqual(regs('0x12, offset:4'),set())
    def test_vcc_aliases(self):self.assertEqual(regs('vcc_lo'),{('s',106)})
    def test_nested_operands(self):self.assertEqual(operands('v0, v[1:4], neg(v2)'),['v0','v[1:4]','neg(v2)'])
    def test_invalid_range(self):
        with self.assertRaises(FormatError):regs('v[30:2]')

class FlowTests(unittest.TestCase):
    def test_signed_backedge(self):
        x=seq(('s_nop','0'),('s_cbranch_execnz','-2',-2),('s_endpgm',))
        e,r,f=cfg(x);self.assertEqual(e[4],[0,8]);self.assertFalse(f)
    def test_out_of_function_branch(self):
        e,r,f=cfg(seq(('s_branch','10',10)));self.assertEqual(f[0]['rule'],'CFG_TARGET')
    def test_indirect_not_certified(self):
        e,r,f=cfg(seq(('s_setpc_b64','s[0:1]')));self.assertEqual(f[0]['severity'],'incomplete')
    def test_missing_terminator(self):
        e,r,f=cfg(seq(('v_mov_b32','v0, 1')));self.assertEqual(f[0]['rule'],'CFG_FALLTHROUGH')
        self.assertEqual(f[0]['severity'],'error')
        # A conditional branch as the last instruction falls through as well. A final call is taken as
        # a call to a noreturn function (rules.cfg docstring; see tests/test_hazards.py CfgTests).
        for last in (('s_cbranch_scc1','0',0),):
            e,r,f=cfg(seq(('v_mov_b32','v0, 1'),last));self.assertIn(('CFG_FALLTHROUGH','error'),[(x['rule'],x['severity'])for x in f],last)
        # Not reached by direct branches: the path cannot be decided.
        e,r,f=cfg(seq(('s_endpgm',),('v_mov_b32','v0, 1')));self.assertEqual([(x['rule'],x['severity'])for x in f],[('CFG_FALLTHROUGH','incomplete')])
    def test_reachable_code_end(self):
        e,r,f=cfg(seq(('s_code_end',)));self.assertEqual(f[0]['rule'],'REACHABLE_CODE_END')
    def test_unreachable_code_end_is_padding(self):
        e,r,f=cfg(seq(('s_endpgm',),('s_code_end',)));self.assertFalse(f);self.assertEqual(r,{0})
    def test_dealloc_then_end_is_allowed(self):
        x=seq(('s_sendmsg','sendmsg(MSG_DEALLOC_VGPRS)',3),('s_wait_storecnt','0'),('s_endpgm',))
        e,r,f=cfg(x);self.assertFalse(deallocation_checks(x,e,r))
    def test_dealloc_then_vector_is_error(self):
        x=seq(('s_sendmsg','sendmsg(MSG_DEALLOC_VGPRS)',3),('v_mov_b32','v0, 1'),('s_endpgm',))
        e,r,f=cfg(x);self.assertIn('USE_AFTER_DEALLOC',{x['rule']for x in deallocation_checks(x,e,r)})
    def test_dealloc_branch_path_checked(self):
        x=seq(('s_sendmsg','sendmsg(MSG_DEALLOC_VGPRS)',3),('s_branch','1',1),('s_endpgm',),('v_mov_b32','v0,1'),('s_endpgm',))
        e,r,f=cfg(x);self.assertIn('USE_AFTER_DEALLOC',{x['rule']for x in deallocation_checks(x,e,r)})
    def test_no_dealloc_is_not_error(self):
        x=seq(('v_mov_b32','v0,1'),('s_endpgm',));e,r,f=cfg(x);self.assertFalse(deallocation_checks(x,e,r))
    def test_waterfall_requires_backedge(self):
        x=seq(('v_readfirstlane_b32','s0,v0'),('s_and_saveexec_b32','s1,s2'),('v_movrels_b32','v0,v1'),('s_cbranch_execnz','0',0),('s_endpgm',))
        e,r,f=cfg(x);self.assertFalse(waterfall_checks(x,e))
    def test_waterfall_candidate(self):
        x=seq(('v_readfirstlane_b32','s0,v0'),('s_and_saveexec_b32','s1,s2'),('v_movrels_b32','v0,v1'),('s_cbranch_execnz','-4',-4),('s_endpgm',))
        e,r,f=cfg(x);self.assertEqual(waterfall_checks(x,e)[0]['severity'],'info')
    def test_uniform_relative_not_waterfall(self):
        x=seq(('v_movrels_b32','v0,v1'),('s_endpgm',));e,r,f=cfg(x);self.assertFalse(waterfall_checks(x,e))

class HazardTests(unittest.TestCase):
    def rules(self,x):
        e,r,f=cfg(x);return {z['rule']for z in local_checks(x,e,r)[0]}
    def test_load_consume_unwaited_candidate(self):
        self.assertIn('LOCAL_MEMORY_HAZARD',self.rules(seq(('global_load_b32','v0,v[2:3],off'),('v_add_f32','v1,v0,v2'),('s_endpgm',))))
    def test_correct_load_wait(self):
        self.assertNotIn('LOCAL_MEMORY_HAZARD',self.rules(seq(('global_load_b32','v0,v[2:3],off'),('s_wait_loadcnt','0'),('v_add_f32','v1,v0,v2'),('s_endpgm',))))
    def test_wrong_counter_not_accepted(self):
        self.assertIn('LOCAL_MEMORY_HAZARD',self.rules(seq(('global_load_b32','v0,v[2:3],off'),('s_wait_dscnt','0'),('v_add_f32','v1,v0,v2'),('s_endpgm',))))
    def test_flat_needs_both_domains(self):
        self.assertIn('LOCAL_MEMORY_HAZARD',self.rules(seq(('flat_load_b32','v0,v[2:3]'),('s_wait_loadcnt','0'),('v_add_f32','v1,v0,v2'),('s_endpgm',))))
    def test_flat_combined_wait(self):
        self.assertNotIn('LOCAL_MEMORY_HAZARD',self.rules(seq(('flat_load_b32','v0,v[2:3]'),('s_wait_loadcnt_dscnt','0'),('v_add_f32','v1,v0,v2'),('s_endpgm',))))
    def test_smem_requires_km(self):
        self.assertIn('LOCAL_MEMORY_HAZARD',self.rules(seq(('s_load_b32','s0,s[2:3],0'),('s_wait_loadcnt','0'),('s_mov_b32','s4,s0'),('s_endpgm',))))
    def test_nonzero_wait_not_assumed_complete(self):
        self.assertIn('LOCAL_MEMORY_HAZARD',self.rules(seq(('global_load_b32','v0,v[2:3],off'),('s_wait_loadcnt','1',1),('v_add_f32','v1,v0,v2'),('s_endpgm',))))
    def test_wmma_d_to_ab(self):
        self.assertIn('WMMA_D_TO_AB',self.rules(seq(('v_wmma_f32_16x16x16_bf16','v[0:7],v[8:11],v[12:15],v[0:7]'),('v_wmma_f32_16x16x16_bf16','v[16:23],v[0:3],v[12:15],v[16:23]'),('s_endpgm',))))
    def test_wmma_d_to_c_allowed(self):
        self.assertNotIn('WMMA_D_TO_AB',self.rules(seq(('v_wmma_f32_16x16x16_bf16','v[0:7],v[8:11],v[12:15],v[0:7]'),('v_wmma_f32_16x16x16_bf16','v[0:7],v[8:11],v[12:15],v[0:7]'),('s_endpgm',))))
    def test_vnop_resolves_wmma(self):
        self.assertNotIn('WMMA_D_TO_AB',self.rules(seq(('v_wmma_f32_16x16x16_bf16','v[0:7],v[8:11],v[12:15],v[0:7]'),('v_nop',),('v_wmma_f32_16x16x16_bf16','v[16:23],v[0:3],v[12:15],v[16:23]'),('s_endpgm',))))
    def test_snop_not_wmma_valu_separator(self):
        self.assertIn('WMMA_D_TO_AB',self.rules(seq(('v_wmma_f32_16x16x16_bf16','v[0:7],v[8:11],v[12:15],v[0:7]'),('s_nop','0'),('v_wmma_f32_16x16x16_bf16','v[16:23],v[0:3],v[12:15],v[16:23]'),('s_endpgm',))))
    def test_permlane_cmpx(self):
        self.assertIn('PERMLANE_AFTER_CMPX',self.rules(seq(('v_cmpx_eq_u32','v0,v1'),('v_permlane16_b32','v0,v1,s0,s1'),('s_endpgm',))))
    def test_permlane_vnop(self):
        # LLVM GCNHazardRecognizer::fixVcmpxPermlaneHazards: V_NOP does not expire the hazard ("V_NOP will be
        # discarded by SQ"); its fix is a v_mov_b32 of the permlane's src0 onto itself.
        self.assertIn('PERMLANE_AFTER_CMPX',self.rules(seq(('v_cmpx_eq_u32','v0,v1'),('v_nop',),('v_permlane16_b32','v0,v1,s0,s1'),('s_endpgm',))))
        self.assertNotIn('PERMLANE_AFTER_CMPX',self.rules(seq(('v_cmpx_eq_u32','v0,v1'),('v_mov_b32','v1,v1'),('v_permlane16_b32','v0,v1,s0,s1'),('s_endpgm',))))
    def test_sgpr_documented_pattern(self):
        self.assertIn('SGPR_PAIR_GUARD',self.rules(seq(('v_mov_b32','v0,s1'),('s_mov_b32','s0,0x42'),('s_mov_b32','s2,s0'),('s_endpgm',))))
    def test_sgpr_different_pair(self):
        self.assertNotIn('SGPR_PAIR_GUARD',self.rules(seq(('v_mov_b32','v0,s3'),('s_mov_b32','s0,0x42'),('s_mov_b32','s2,s0'),('s_endpgm',))))
    def test_plain_salu_not_blanket_hazard(self):
        self.assertNotIn('SGPR_PAIR_GUARD',self.rules(seq(('s_mov_b32','s1,0'),('s_mov_b32','s0,0x42'),('s_mov_b32','s2,s0'),('s_endpgm',))))
