; Calls to a noreturn function and llvm.trap, as clang emits them for gfx1201 (cf. a HIP
; `[[noreturn]] void die()` ending in __builtin_trap()). The block of a noreturn call ends with
; s_swappc_b64; llvm.trap is s_trap 2 (AMDGPUUsage Trap Handler ABI: the wave is halted).
; Neither falls through into the next function or the s_code_end padding.
	s_cbranch_scc0 .LBB0_2
	s_swappc_b64 s[30:31], s[0:1]
.LBB0_2:
	s_trap 2
	s_code_end
	s_code_end
