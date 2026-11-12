; As LLVM emits it for gfx1201: s_nop 0 right before MSG_DEALLOC_VGPRS.
	v_mov_b32 v1, v2
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
