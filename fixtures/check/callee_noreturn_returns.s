; Kernel k ends with its call to helper (nothing follows the s_swappc_b64), so the call is taken
; as a call to a noreturn function; helper does return (s_setpc_b64 s[30:31]), and the wave would
; then run past the end of k: NORETURN_CALL_RETURNS. Derived from callee_missing_waits.s.
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.text
	.globl	k
	.p2align	8
	.type	k,@function
k:
	global_load_b32 v1, v0, s[0:1]
	s_getpc_b64 s[4:5]
	s_add_co_u32 s4, s4, helper@rel32@lo+4
	s_add_co_ci_u32 s5, s5, helper@rel32@hi+12
	s_swappc_b64 s[30:31], s[4:5]
.Lfunc_end_k:
	.size	k, .Lfunc_end_k-k
	.p2align	2
	.type	helper,@function
helper:
	v_add_f32 v1, 1.0, v1
	global_load_b32 v1, v0, s[0:1]
	s_setpc_b64 s[30:31]
.Lfunc_end_helper:
	.size	helper, .Lfunc_end_helper-helper
	.rodata
	.p2align	6
	.amdhsa_kernel k
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name:           k
    .symbol:         k.kd
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .max_flat_workgroup_size: 256
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
