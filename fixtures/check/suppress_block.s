; radisa check: a standalone /* */ directive (line 4) and one whose reason continues on the next
; line of the block comment (lines 8-9) each suppress the MISSING_WAIT on the following code line.
	global_load_b32 v1, v[2:3], off
	/* radisa: ignore[MISSING_WAIT] v4 is discarded later */
	v_add_f32 v4, v1, v4
	s_wait_loadcnt 0x0
	global_load_b32 v5, v[2:3], off
	/* radisa: ignore[MISSING_WAIT]
	   v6 is dead */
	v_add_f32 v6, v5, v5
	s_wait_loadcnt 0x0
	s_endpgm
