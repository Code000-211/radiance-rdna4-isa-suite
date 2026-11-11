; Two D16 loads fill the two halves of v3; each store reads one half only, so it has to wait
; for the load of that half and not for the other one (LLVM SIInsertWaitcnts emits exactly this).
global_load_d16_b16 v3, v[4:5], off
global_load_d16_hi_b16 v3, v[6:7], off
s_wait_loadcnt 0x1
global_store_b16 v[8:9], v3, off
s_wait_loadcnt 0x0
global_store_d16_hi_b16 v[10:11], v3, off
s_endpgm
