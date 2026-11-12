; radisa check file-level suppression: both MISSING_WAIT findings are suppressed by the
; ignore-file comment below; the run passes and the summary names the hidden errors.
// radisa: ignore-file[MISSING_WAIT] test fixture: the loaded values are never used for results
	global_load_b32 v1, v[2:3], off
	v_add_f32 v4, v1, v4
	s_load_b32 s4, s[0:1], 0x0
	s_add_co_u32 s6, s4, 1
	s_endpgm
