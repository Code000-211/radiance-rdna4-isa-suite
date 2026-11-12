; A loop whose back edge carries the WMMA D (v[8:15]) into the B operand of the same WMMA:
; only SALU lies between them on the back edge.
L0:
	v_wmma_f32_16x16x16_f16 v[16:23], v[0:3], v[8:11], v[16:23]
	v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15]
	s_sub_co_u32 s0, s0, 1
	s_cbranch_scc1 L0
	s_endpgm
