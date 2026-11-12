; SYS_STORE_WAIT (opt-in) failing case: a non-atomic scope:SCOPE_SYS store is reached while a
; global load (LOADcnt), a scalar load (KMcnt) and a store (STOREcnt) may be outstanding; LLVM
; (SIGfx12CacheControl::insertWaitsBeforeSystemScopeStore) puts s_wait_loadcnt/samplecnt/
; bvhcnt/kmcnt/storecnt 0 in front of such a store. No register dependency: MISSING_WAIT is clean.
	global_load_b32 v1, v0, s[0:1]
	s_load_b32 s4, s[2:3], 0x0
	global_store_b32 v0, v3, s[0:1] offset:4
	global_store_b32 v0, v2, s[0:1] scope:SCOPE_SYS
	s_endpgm
