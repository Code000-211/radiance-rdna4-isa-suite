; SWMMAC index operand (v8) overlaps the D of the previous WMMA (v[8:15]): a GFX12 hazard in
; fixWMMAHazards even though A and B do not overlap.
	v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15]
	v_swmmac_f32_16x16x32_f16 v[16:23], v[0:3], v[24:31], v8
	s_endpgm
