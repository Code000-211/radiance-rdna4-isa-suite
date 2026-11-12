; radisa check inline suppression: line 6 is suppressed by the comment above it, line 9 has a
; suppression without a reason (inactive, SUPPRESSION_NO_REASON), line 12 is not suppressed, and
; the comment on line 14 suppresses nothing (UNUSED_SUPPRESSION).
	global_load_b32 v1, v[2:3], off
	; radisa: ignore[MISSING_WAIT] v4 is recomputed after the wait below; this value is discarded
	v_add_f32 v4, v1, v4
	s_wait_loadcnt 0x0
	global_load_b32 v5, v[2:3], off
	v_add_f32 v7, v5, v5 ; radisa: ignore[MISSING_WAIT]
	s_wait_loadcnt 0x0
	s_load_b32 s4, s[0:1], 0x0
	s_add_co_u32 s6, s4, 1
	s_wait_kmcnt 0x0
	; radisa: ignore[MISSING_ALU_WAIT] nothing here needs it
	s_endpgm
