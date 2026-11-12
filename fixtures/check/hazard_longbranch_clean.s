; BranchRelaxation's long branch (SIInstrInfo::insertIndirectBranch), as llc emits it for gfx1201.
; s_setpc_b64 never falls through: .LBB0_2 is reached only from the s_cbranch_scc0, without the cmpx.
; The long branch reaches .LBB0_3 after v_cmpx; the v_mov_b32 there is the fix LLVM inserts.
	s_cbranch_scc0 .LBB0_2
	v_cmpx_gt_u32 v0, v1
	s_getpc_b64 s[2:3]
.Lpost_getpc0:
	s_add_co_u32 s2, s2, (.LBB0_3-.Lpost_getpc0)&4294967295
	s_add_co_ci_u32 s3, s3, (.LBB0_3-.Lpost_getpc0)>>32
	s_setpc_b64 s[2:3]
.LBB0_2:
	v_permlane16_b32 v2, v3, s0, s1
	s_endpgm
.LBB0_3:
	v_mov_b32 v3, v3
	v_permlane16_b32 v2, v3, s0, s1
	s_endpgm
