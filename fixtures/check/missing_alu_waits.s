; clean_alu_waits.s without its three s_wait_alu: lines 4, 6 and 10 read an SGPR too early.
v_add_co_ci_u32_e64 v14, null, s15, v14, s0
v_add_co_u32 v15, s0, s14, v15
v_add_co_ci_u32_e64 v16, null, s15, v16, s0
s_mov_b32 s14, 4
s_add_co_u32 s1, s14, 1
v_cmp_eq_u32_e32 vcc_lo, 0, v8
v_cndmask_b32_e32 v0, v1, v2, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 0, v9
v_cndmask_b32_e32 v3, v1, v2, vcc_lo
s_endpgm
