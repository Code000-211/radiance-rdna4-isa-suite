; clang 18 -O2 output of: __device__ noinline noreturn void die(int* p) { p[0] = 1; __builtin_trap(); }
; int chk(int* p, int x) { if (x > 5) die(p); return p[x]; }  (noinline)
; extern "C" __global__ void kn(int* p, int x) { if (x < 0) die(p); p[1] = chk(p, x); }
; kn ends with the noreturn call to die, die ends with s_trap 2 (llvm.trap).
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.p2align	2
	.type	_Z3diePi,@function
_Z3diePi:
	s_wait_loadcnt_dscnt 0x0
	s_wait_expcnt 0x0
	s_wait_samplecnt 0x0
	s_wait_bvhcnt 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, 1
	flat_store_b32 v[0:1], v2
	s_trap 2
.Lfunc_end0:
	.size	_Z3diePi, .Lfunc_end0-_Z3diePi
	.section	.AMDGPU.csdata,"",@progbits
	.text
	.p2align	2
	.type	_Z3chkPii,@function
_Z3chkPii:
	s_wait_loadcnt_dscnt 0x0
	s_wait_expcnt 0x0
	s_wait_samplecnt 0x0
	s_wait_bvhcnt 0x0
	s_wait_kmcnt 0x0
	s_mov_b32 s1, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b32 s0, -1
	scratch_store_b32 off, v4, s33
	s_mov_b32 exec_lo, s0
	v_writelane_b32 v4, s30, 0
	s_mov_b32 s0, exec_lo
	s_add_co_i32 s32, s32, 16
	v_writelane_b32 v4, s31, 1
	v_cmpx_gt_i32_e32 6, v2
	s_xor_b32 s0, exec_lo, s0
	s_cbranch_execz .LBB1_2
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v1, vcc_lo, v1, v3, vcc_lo
	flat_load_b32 v3, v[0:1]
.LBB1_2:
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB1_4
	s_getpc_b64 s[2:3]
	s_sext_i32_i16 s3, s3
	s_add_co_u32 s2, s2, _Z3diePi@rel32@lo+8
	s_add_co_ci_u32 s3, s3, _Z3diePi@rel32@hi+16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_swappc_b64 s[30:31], s[2:3]
.LBB1_4:
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x0
	v_mov_b32_e32 v0, v3
	v_readlane_b32 s31, v4, 1
	v_readlane_b32 s30, v4, 0
	s_xor_saveexec_b32 s0, -1
	scratch_load_b32 v4, off, s33
	s_mov_b32 exec_lo, s0
	s_add_co_i32 s32, s32, -16
	s_mov_b32 s33, s1
	s_wait_loadcnt 0x0
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	_Z3chkPii, .Lfunc_end1-_Z3chkPii
	.section	.AMDGPU.csdata,"",@progbits
	.text
	.protected	kn
	.globl	kn
	.p2align	8
	.type	kn,@function
kn:
	s_load_b96 s[4:6], s[0:1], 0x0
	s_mov_b32 s32, 0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s6, -1
	s_cbranch_scc0 .LBB2_2
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v0, s4
	v_dual_mov_b32 v1, s5 :: v_dual_mov_b32 v2, s6
	s_getpc_b64 s[0:1]
	s_sext_i32_i16 s1, s1
	s_add_co_u32 s0, s0, _Z3chkPii@rel32@lo+8
	s_add_co_ci_u32 s1, s1, _Z3chkPii@rel32@hi+16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_swappc_b64 s[30:31], s[0:1]
	global_store_b32 v5, v0, s[4:5] offset:4
	s_endpgm
.LBB2_2:
	v_dual_mov_b32 v0, s4 :: v_dual_mov_b32 v1, s5
	s_getpc_b64 s[0:1]
	s_sext_i32_i16 s1, s1
	s_add_co_u32 s0, s0, _Z3diePi@rel32@lo+8
	s_add_co_ci_u32 s1, s1, _Z3diePi@rel32@hi+16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_swappc_b64 s[30:31], s[0:1]
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kn
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 16
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 34
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
.Lfunc_end2:
	.size	kn, .Lfunc_end2-kn
	.section	.AMDGPU.csdata,"",@progbits
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .max_flat_workgroup_size: 1024
    .name:           kn
    .private_segment_fixed_size: 16
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         kn.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
