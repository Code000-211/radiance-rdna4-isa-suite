; callee_rdc.s with every s_wait_* removed: the entry waits, the wait for the two flat loads
; before v_fmac_f32 (MISSING_WAIT) and so the return leaves them outstanding (CALLEE_RETURN_OUTSTANDING).
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.hidden	_Z4leafPKfi                     ; -- Begin function _Z4leafPKfi
	.globl	_Z4leafPKfi
	.p2align	2
	.type	_Z4leafPKfi,@function
_Z4leafPKfi:                            ; @_Z4leafPKfi
; %bb.0:
	v_ashrrev_i32_e32 v3, 31, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v1, vcc_lo, v1, v3, vcc_lo
	s_clause 0x1
	flat_load_b32 v2, v[0:1]
	flat_load_b32 v0, v[0:1] offset:28
	v_fmac_f32_e32 v0, 2.0, v2
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	_Z4leafPKfi, .Lfunc_end0-_Z4leafPKfi
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 88
; NumSgprs: 34
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.hidden	__oclc_ABI_version              ; @__oclc_ABI_version
	.type	__oclc_ABI_version,@object
	.section	.rodata,"a",@progbits
	.weak	__oclc_ABI_version
	.p2align	2, 0x0
__oclc_ABI_version:
	.long	500                             ; 0x1f4
	.size	__oclc_ABI_version, 4

	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
