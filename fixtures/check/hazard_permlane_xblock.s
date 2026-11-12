; v_cmpx writes EXEC; on the branch-taken path only s_nop and v_nop lie before the permlane.
; LLVM fixVcmpxPermlaneHazards inserts v_mov_b32 v3, v3 before it.
	v_cmpx_gt_u32 v0, v1
	s_cbranch_execz L1
	v_add_f32 v4, v5, v6
L1:
	s_nop 0
	v_nop
	v_permlane16_b32 v2, v3, s0, s1
	s_endpgm
