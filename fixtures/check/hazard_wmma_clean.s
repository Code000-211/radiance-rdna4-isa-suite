; No WMMA_D_TO_AB: every path from the first WMMA to the A/B readers passes a VALU (v_nop counts),
; and D feeding C is allowed on GFX12 (fixWMMAHazards: the hardware stalls).
	v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15]
	v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15]
	s_cbranch_scc0 L1
	v_nop
	s_branch L2
L1:
	v_add_f32 v24, v25, v26
L2:
	v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[12:15], v[16:23]
	s_endpgm
