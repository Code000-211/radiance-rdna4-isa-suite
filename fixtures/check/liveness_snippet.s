; tests/test_liveness.py: a snippet has no kernel descriptor, so no entry ABI is known and
; UNINITIALIZED_READ stays off unless pass option `liveness_entry` names the registers defined at
; entry. With ['s[40:41]', 'v0'] the reads of v200 and v201 are reported.
	v_add_f32 v2, v200, v201
	global_store_b32 v0, v2, s[40:41]
	s_endpgm
