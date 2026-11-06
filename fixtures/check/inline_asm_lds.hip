// Self-contained: builds with -nogpuinc, no ROCm headers needed.
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))

// BUG: the asm statement issues an LDS load, the compiler does not insert
// s_wait_dscnt for it, and the add below reads the destination too early.
extern "C" __global__ void __launch_bounds__(256) racy(float* o, const float* a) {
  __shared__ float s[256];
  int t = __builtin_amdgcn_workitem_id_x();
  s[t] = a[t];
  __builtin_amdgcn_s_barrier();
  float v;
  asm volatile("ds_load_b32 %0, %1" : "=v"(v) : "v"(((t + 1) & 255) * 4));
  o[t] = v + 1.0f;
}

// Fixed: the wait sits in the same asm statement.
extern "C" __global__ void __launch_bounds__(256) fixed(float* o, const float* a) {
  __shared__ float s[256];
  int t = __builtin_amdgcn_workitem_id_x();
  s[t] = a[t];
  __builtin_amdgcn_s_barrier();
  float v;
  asm volatile("ds_load_b32 %0, %1\n\ts_wait_dscnt 0x0" : "=v"(v) : "v"(((t + 1) & 255) * 4));
  o[t] = v + 1.0f;
}
