; radisa check: a trailing directive covers its own line only. The one on line 4 annotates s_nop
; and must not hide the MISSING_WAIT on line 5 (it is UNUSED_SUPPRESSION instead).
	global_load_b32 v1, v[2:3], off
	s_nop 0 ; radisa: ignore[MISSING_WAIT] v4 discarded
	v_add_f32 v4, v1, v4
	s_wait_loadcnt 0x0
	s_endpgm
