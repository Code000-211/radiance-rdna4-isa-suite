; tests/test_liveness.py (radisa/liveness.py); same shape as tests/support.py kernel_asm().
; UNINITIALIZED_READ 'review': v1 (gfx1201 packs the work-item IDs into v0; v1 is not set up), s7 (never
; written; the entry state is s[0:1] kernarg pointer and s2 work-group id). 'info': v4 (written only when
; s2 != 0: maybe undefined, as LLVM's phi(implicit-def, x) would be).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.text
	.globl	lv_uninit
	.p2align	8
	.type	lv_uninit,@function
lv_uninit:
	s_load_b64 s[4:5], s[0:1], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	v_add_nc_u32_e32 v3, v1, v2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lskip
	v_mov_b32_e32 v4, 1.0
	.Lskip:
	v_add_f32_e32 v5, v4, v3
	s_add_co_u32 s6, s7, s2
	v_add_nc_u32_e32 v5, s6, v5
	s_wait_kmcnt 0x0
	global_store_b32 v2, v5, s[4:5]
	s_endpgm
.Lfunc_end_lv_uninit:
	.size	lv_uninit, .Lfunc_end_lv_uninit-lv_uninit
	.rodata
	.p2align	6
	.amdhsa_kernel lv_uninit
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
  - .name:           lv_uninit
    .symbol:         lv_uninit.kd
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
