; tests/test_liveness.py (radisa/liveness.py); same shape as tests/support.py kernel_asm().
; No UNINITIALIZED_READ: x^x / x-x idioms, a self-move, TTMP9 (architected work-group id), s2, v0, EXEC;
; v6 is written on both sides of a divergent if/else (the execz edges carry no lanes); a D16 load
; defines v4; v5 is the undef upper half of the v[4:5] tuple; after s_swappc_b64 every register
; counts as written (the callee is not visible).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.text
	.globl	lv_clean
	.p2align	8
	.type	lv_clean,@function
lv_clean:
	s_load_b64 s[4:5], s[0:1], 0x0
	v_xor_b32_e32 v1, v1, v1
	v_sub_nc_u32_e32 v2, v2, v2
	s_xor_b32 s8, s8, s8
	v_mov_b32_e32 v3, v3
	v_mov_b32_e32 v3, ttmp9
	v_add_nc_u32_e32 v3, s2, v3
	v_lshlrev_b32_e32 v2, 2, v0
	v_cmp_gt_u32_e32 vcc_lo, 64, v0
	s_and_saveexec_b32 s9, vcc_lo
	s_xor_b32 s9, exec_lo, s9
	s_cbranch_execz .Lelse
	v_mov_b32_e32 v6, 2.0
	.Lelse:
	s_and_not1_saveexec_b32 s9, s9
	s_cbranch_execz .Lend
	v_mov_b32_e32 v6, 1.0
	.Lend:
	s_or_b32 exec_lo, exec_lo, s9
	v_add_f32_e32 v6, v6, v3
	v_add_nc_u32_e32 v6, v1, v6
	s_wait_kmcnt 0x0
	global_load_d16_b16 v4, v2, s[4:5]
	s_wait_loadcnt 0x0
	v_add_f16_e32 v4, v4, v4
	global_store_b64 v2, v[4:5], s[4:5]
	global_store_b32 v2, v6, s[4:5]
	s_mov_b64 s[10:11], 0
	s_swappc_b64 s[30:31], s[10:11]
	v_add_nc_u32_e32 v7, v7, v8
	global_store_b32 v2, v7, s[4:5]
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end_lv_clean:
	.size	lv_clean, .Lfunc_end_lv_clean-lv_clean
	.rodata
	.p2align	6
	.amdhsa_kernel lv_clean
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 16
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name:           lv_clean
    .symbol:         lv_clean.kd
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .max_flat_workgroup_size: 256
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .vgpr_count:     16
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
