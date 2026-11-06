global_load_b32 v1, v[2:3], off
global_load_b32 v5, v[2:3], off offset:4
s_wait_loadcnt 0x1
v_add_f32 v4, v1, v1
s_wait_loadcnt 0x0
v_add_f32 v6, v5, v5
s_load_b32 s4, s[0:1], 0x0
s_wait_kmcnt 0x0
v_mov_b32 v7, s4
s_endpgm
