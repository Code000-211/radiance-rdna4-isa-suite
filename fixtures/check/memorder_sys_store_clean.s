; SYS_STORE_WAIT clean case: the waits LLVM inserts before a non-atomic scope:SCOPE_SYS store on
; GFX12.0 are present. An outstanding LDS store (DScnt) is not part of that rule, and a
; system-scope atomic is not a plain store.
	global_load_b32 v1, v0, s[0:1]
	s_load_b32 s4, s[2:3], 0x0
	global_store_b32 v0, v3, s[0:1] offset:4
	ds_store_b32 v0, v3
	s_wait_loadcnt 0x0
	s_wait_samplecnt 0x0
	s_wait_bvhcnt 0x0
	s_wait_kmcnt 0x0
	s_wait_storecnt 0x0
	global_store_b32 v0, v2, s[0:1] scope:SCOPE_SYS
	global_atomic_add_u32 v0, v1, s[0:1] scope:SCOPE_SYS
	s_endpgm
