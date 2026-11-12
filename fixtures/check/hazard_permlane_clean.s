; No PERMLANE_AFTER_CMPX: a non-NOP VALU follows every EXEC-writing compare on every path,
; and a compare that writes VCC (not EXEC) is no producer.
	v_cmpx_gt_u32 v0, v1
	s_cbranch_execz L1
	v_add_f32 v4, v5, v6
L1:
	v_mov_b32 v3, v3
	v_permlane16_b32 v2, v3, s0, s1
	v_cmp_gt_u32 vcc_lo, v0, v1
	v_permlanex16_b32 v2, v3, s0, s1
	s_endpgm
