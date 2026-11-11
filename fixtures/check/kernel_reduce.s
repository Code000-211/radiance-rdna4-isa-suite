; A whole kernel as the gfx1201 compiler emits it (AMD clang 24, -O3), checked in so that the
; tests do not depend on a compiler: s[t] = a[t] * b[t]; barrier; if (t < n) o[t] = s[t] + s[t+128].
; tests/test_check.py removes each wait and the s_endpgm in turn; every such mutant must fail.
	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	reduce                  ; -- Begin function reduce
	.globl	reduce
	.p2align	8
	.type	reduce,@function
reduce:                                 ; @reduce
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b96 s[0:2], s[0:1], 0x10
	v_lshlrev_b32_e32 v1, 2, v0
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b32 v2, v1, s[6:7]
	global_load_b32 v3, v1, s[0:1]
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v2, v2, v3
	ds_store_b32 v1, v2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_cmpx_gt_i32_e32 s2, v0
	s_cbranch_execz .LBB0_2
; %bb.1:
	ds_load_2addr_stride64_b32 v[2:3], v1 offset1:2
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v2, v3
	global_store_b32 v1, v0, s[4:5]
.LBB0_2:
	s_endpgm
.Lfunc_end0:
	.size	reduce, .Lfunc_end0-reduce
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel reduce
		.amdhsa_group_segment_fixed_size 1024
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-reduce)<<4)&4080)>>4
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
                                        ; -- End function
	.set .Lreduce.num_vgpr, 4
	.set .Lreduce.num_agpr, 0
	.set .Lreduce.numbered_sgpr, 8
	.set .Lreduce.num_named_barrier, 0
	.set .Lreduce.private_seg_size, 0
	.set .Lreduce.uses_vcc, 0
	.set .Lreduce.uses_flat_scratch, 0
	.set .Lreduce.has_dyn_sized_stack, 0
	.set .Lreduce.has_recursion, 0
	.set .Lreduce.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 256
    .name:           reduce
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .sgpr_spill_count: 0
    .symbol:         reduce.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
