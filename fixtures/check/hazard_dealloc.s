; MSG_DEALLOC_VGPRS without the s_nop 0 that LLVM (SIInsertWaitcnts) puts before it for gfx1201.
	v_mov_b32 v1, v2
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
