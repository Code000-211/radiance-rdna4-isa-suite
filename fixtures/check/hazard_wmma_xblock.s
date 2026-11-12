; WMMA D (v[8:15]) feeds the A operand of a WMMA in another basic block. On the branch-taken path
; nothing lies in between (the fall-through path has a VALU): LLVM fixWMMAHazards inserts v_nop at L1.
	v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15]
	s_cbranch_scc0 L1
	v_mov_b32 v24, v25
	s_nop 0
L1:
	v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[4:7], v[16:23]
	s_endpgm
