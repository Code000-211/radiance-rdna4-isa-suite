; The smallest complete gfx1201 kernel (descriptor + metadata) that both LLVM 18 and newer llvm-mc
; assemble: tests/support.py kernel_asm() writes the same shape. o[t] = o[t] + 1.0.
; Code object v5 and the 3-component target id: LLVM 18 rejects the AMD clang 20+ form
; ("amdgcn-amd-amdhsa-unknown-gfx1201", .amdhsa_inst_pref_size with instprefsize()).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.text
	.globl	minimal
	.p2align	8
	.type	minimal,@function
minimal:
	s_load_b64 s[4:5], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_wait_kmcnt 0x0
	global_load_b32 v2, v1, s[4:5]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, 1.0, v2
	global_store_b32 v1, v2, s[4:5]
	s_endpgm
.Lfunc_end_minimal:
	.size	minimal, .Lfunc_end_minimal-minimal
	.rodata
	.p2align	6
	.amdhsa_kernel minimal
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
  - .name:           minimal
    .symbol:         minimal.kd
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
