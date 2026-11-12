; llc 18.1.3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -O2 output (unedited below this header) for
; a kernel that calls a noinline function: LLVM's callee prologue (s_wait_loadcnt_dscnt 0x0,
; s_wait_expcnt/samplecnt/bvhcnt/kmcnt 0x0) and the wait before s_setpc_b64 s[30:31] return.
; Callee check clean case.
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.hidden	helper                          ; -- Begin function helper
	.globl	helper
	.p2align	2
	.type	helper,@function
helper:                                 ; @helper
; %bb.0:
	s_wait_loadcnt_dscnt 0x0
	s_wait_expcnt 0x0
	s_wait_samplecnt 0x0
	s_wait_bvhcnt 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v3, v[0:1], off
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v3, v2
	global_store_b32 v[0:1], v2, off
	v_mov_b32_e32 v0, v2
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	helper, .Lfunc_end0-helper
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 60
; NumSgprs: 32
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	k                               ; -- Begin function k
	.p2align	8
	.type	k,@function
k:                                      ; @k
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s32, 0
	s_getpc_b64 s[2:3]
	s_sext_i32_i16 s3, s3
	s_add_co_u32 s2, s2, helper@rel32@lo+8
	s_add_co_ci_u32 s3, s3, helper@rel32@hi+16
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, s1
	s_load_b32 s4, s[0:1], 0x0
	v_mov_b32_e32 v0, s0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, s4
	s_swappc_b64 s[30:31], s[2:3]
	global_store_b32 v4, v0, s[0:1] offset:4
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 1
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 33
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	k, .Lfunc_end1-k
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 88
; NumSgprs: 33
; NumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 5
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           p
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           k
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         k.kd
    .uses_dynamic_stack: true
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
