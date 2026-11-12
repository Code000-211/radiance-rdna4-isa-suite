; A snippet whose bytes are only partly decoded: the dword after v_add_f32 is no instruction, so the
; decoder stops there and the missing wait below it is never analysed. `radisa check` must not report
; PASS for it (status INCOMPLETE, exit 3).
	v_add_f32 v2, v1, v1
	.long 0xffffffff
	global_load_b32 v1, v[4:5], off
	v_add_f32 v2, v1, v1
	s_endpgm
