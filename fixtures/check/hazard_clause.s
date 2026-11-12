; Three s_clause bodies LLVM's SIInsertHardClauses never emits for gfx1201:
; 33 instructions (limit 32), a load and a store in one clause, and a branch into a clause body.
	s_clause 0x20
	global_load_b32 v1, v0, s[0:1]
	global_load_b32 v1, v0, s[0:1] offset:4
	global_load_b32 v1, v0, s[0:1] offset:8
	global_load_b32 v1, v0, s[0:1] offset:12
	global_load_b32 v1, v0, s[0:1] offset:16
	global_load_b32 v1, v0, s[0:1] offset:20
	global_load_b32 v1, v0, s[0:1] offset:24
	global_load_b32 v1, v0, s[0:1] offset:28
	global_load_b32 v1, v0, s[0:1] offset:32
	global_load_b32 v1, v0, s[0:1] offset:36
	global_load_b32 v1, v0, s[0:1] offset:40
	global_load_b32 v1, v0, s[0:1] offset:44
	global_load_b32 v1, v0, s[0:1] offset:48
	global_load_b32 v1, v0, s[0:1] offset:52
	global_load_b32 v1, v0, s[0:1] offset:56
	global_load_b32 v1, v0, s[0:1] offset:60
	global_load_b32 v1, v0, s[0:1] offset:64
	global_load_b32 v1, v0, s[0:1] offset:68
	global_load_b32 v1, v0, s[0:1] offset:72
	global_load_b32 v1, v0, s[0:1] offset:76
	global_load_b32 v1, v0, s[0:1] offset:80
	global_load_b32 v1, v0, s[0:1] offset:84
	global_load_b32 v1, v0, s[0:1] offset:88
	global_load_b32 v1, v0, s[0:1] offset:92
	global_load_b32 v1, v0, s[0:1] offset:96
	global_load_b32 v1, v0, s[0:1] offset:100
	global_load_b32 v1, v0, s[0:1] offset:104
	global_load_b32 v1, v0, s[0:1] offset:108
	global_load_b32 v1, v0, s[0:1] offset:112
	global_load_b32 v1, v0, s[0:1] offset:116
	global_load_b32 v1, v0, s[0:1] offset:120
	global_load_b32 v1, v0, s[0:1] offset:124
	global_load_b32 v1, v0, s[0:1] offset:128
	s_clause 0x1
	global_load_b32 v1, v0, s[0:1]
	global_store_b32 v0, v5, s[0:1]
	s_cbranch_scc0 L1
	s_clause 0x1
	global_load_b32 v2, v0, s[0:1]
L1:
	global_load_b32 v3, v0, s[0:1] offset:4
	s_wait_loadcnt 0x0
	s_endpgm
