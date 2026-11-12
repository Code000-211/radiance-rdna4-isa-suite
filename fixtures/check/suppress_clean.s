; radisa check: a used suppression with a reason next to clean code produces no hygiene finding.
	global_load_b32 v1, v[2:3], off
	s_wait_loadcnt 0x0
	v_add_f32 v4, v1, v1
	global_load_b32 v5, v[2:3], off
	v_add_f32 v6, v5, v5 /* radisa: ignore[MISSING_WAIT] fixture: v6 is dead */
	s_wait_loadcnt 0x0
	s_endpgm
