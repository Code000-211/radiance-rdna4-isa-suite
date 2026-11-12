// Must pass: LDS write, then a release fence at the "workgroup-one-as" scope and a barrier. The fence
// orders no second address space, so LLVM (SIMemoryLegalizer.cpp SIGfx12CacheControl::insertWait,
// DSCnt |= IsCrossAddrSpaceOrdering; same in ROCm d6f6cb69) emits no s_wait_dscnt before the barrier:
// LDS_STORE_BEFORE_BARRIER is a warning here, never an error.
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))

extern "C" __global__ void __launch_bounds__(256) reduce(float* o, const float* a) {
  __shared__ float s[256];
  int t = __builtin_amdgcn_workitem_id_x();
  s[t] = a[t];
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup-one-as");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup-one-as");
  if (t < 128) o[t] = s[t] + s[t + 128];
}
