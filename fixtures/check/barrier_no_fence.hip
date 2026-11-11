// A workgroup barrier without a release fence: the compiler inserts no wait, so the LDS write
// of line 12 may still be outstanding when the barrier releases the other waves. Must fail.
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))

extern "C" __global__ void __launch_bounds__(256) reduce(float* o, const float* a) {
  __shared__ float s[256];
  int t = __builtin_amdgcn_workitem_id_x();
  s[t] = a[t];
  __builtin_amdgcn_s_barrier();
  if (t < 128) o[t] = s[t] + s[t + 128];
}
