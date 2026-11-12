// Must pass: ordinary compiler-scheduled code that clang 18 and newer build for gfx1201
// (no 3-argument __builtin_amdgcn_fence, no ROCm headers). Used by tests/test_compiler_clean.py:
// every finding of severity 'error' on compiler output is a false positive of radisa check.
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))

// Barrier with the release/acquire fences around it (2-argument form: all address spaces).
__device__ static inline __attribute__((always_inline)) void sync_block() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

extern "C" __global__ void __launch_bounds__(256) saxpy(float* y, const float* x, float a, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x();
  if (i < n) y[i] = a * x[i] + y[i];
}

extern "C" __global__ void __launch_bounds__(256) block_sum(float* o, const float* a) {
  __shared__ float s[256];
  int t = __builtin_amdgcn_workitem_id_x();
  s[t] = a[__builtin_amdgcn_workgroup_id_x() * 256 + t];
  sync_block();
  for (int w = 128; w > 0; w >>= 1) {
    if (t < w) s[t] += s[t + w];
    sync_block();
  }
  if (t == 0) o[__builtin_amdgcn_workgroup_id_x()] = s[0];
}

extern "C" __global__ void __launch_bounds__(128) histogram(unsigned* bins, const unsigned char* d, int n) {
  for (int i = __builtin_amdgcn_workitem_id_x(); i < n; i += 128)
    __atomic_fetch_add(&bins[d[i]], 1u, __ATOMIC_RELAXED);
}

extern "C" __global__ void __launch_bounds__(64) gather(float* o, const float* a, const int* idx, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  float acc = 0.0f;
  for (int k = 0; k < n; ++k) acc += a[idx[k * 64 + t]] * (float)k;
  o[t] = acc;
}

extern "C" __global__ void __launch_bounds__(256) scalar_branchy(int* o, const int* p, int n) {
  int s = 0;
  for (int k = 0; k < n; ++k) s = (p[k] & 1) ? s + p[k] : s ^ p[k];   // uniform: SALU + SMEM
  o[__builtin_amdgcn_workitem_id_x()] = s;
}
