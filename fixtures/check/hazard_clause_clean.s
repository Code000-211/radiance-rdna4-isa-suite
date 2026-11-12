; Clauses as LLVM's SIInsertHardClauses forms them for gfx1201: one type per clause, s_nop allowed
; in the middle, at most 32 instructions.
	s_clause 0x2
	s_load_b64 s[4:5], s[0:1], 0x0
	s_nop 0
	s_load_b64 s[6:7], s[0:1], 0x8
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b32 v1, v0, s[4:5]
	global_load_b32 v2, v0, s[4:5] offset:4
	s_wait_loadcnt 0x0
	s_endpgm
