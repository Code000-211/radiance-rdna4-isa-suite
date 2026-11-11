; The store of line 5 reads the low half of v3 while the load that writes it is outstanding.
global_load_d16_hi_b16 v3, v[6:7], off
global_load_d16_b16 v3, v[4:5], off
s_wait_loadcnt 0x1
global_store_b16 v[8:9], v3, off
; A VALU instruction reads the whole register: both halves must have landed.
global_load_d16_b16 v12, v[4:5], off
global_load_d16_hi_b16 v12, v[6:7], off
s_wait_loadcnt 0x1
v_add_nc_u32 v13, v12, v12
s_endpgm
