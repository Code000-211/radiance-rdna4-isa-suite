; loop: load issued in the body, consumed after the back edge without a wait
  s_mov_b32 s2, 4
.Lloop:
  v_add_f32 v4, v1, v4
  global_load_b32 v1, v[2:3], off
  s_add_co_i32 s2, s2, -1
  s_cmp_lg_u32 s2, 0
  s_cbranch_scc1 .Lloop
  s_wait_loadcnt 0x0
  ; two loads, wait 1 only proves the older one
  global_load_b32 v5, v[2:3], off
  global_load_b32 v6, v[2:3], off offset:4
  s_wait_loadcnt 0x1
  v_add_f32 v7, v6, v5
  ; scalar loads return out of order: kmcnt 1 proves nothing
  s_load_b32 s4, s[0:1], 0x0
  s_load_b32 s5, s[0:1], 0x4
  s_wait_kmcnt 0x1
  s_add_co_u32 s6, s4, 1
  s_endpgm
