// Ordinary compiler-scheduled code: loads, LDS, loops, WMMA, atomics. Must pass.
// radisa-test: wave32-only (__builtin_amdgcn_wmma_*_w32_gfx12 needs wave32; clang rejects it with -mwavefrontsize64)
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));

extern "C" __global__ void __launch_bounds__(256) wmma(f8* o, const h8* a, const h8* b, int n) {
  int t = __builtin_amdgcn_workitem_id_x();
  f8 c = {};
  for (int k = 0; k < n; ++k) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[t + k], b[t + k], c);
  o[t] = c;
}

// A workgroup barrier with the fences __syncthreads() puts around it, scoped to local memory:
// the release fence is what makes the compiler wait for the LDS writes before the barrier.
__attribute__((device)) static inline __attribute__((always_inline)) void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

extern "C" __global__ void __launch_bounds__(256) reduce(float* o, const float* a, unsigned* cnt, int n) {
  __shared__ float s[256];
  int t = __builtin_amdgcn_workitem_id_x();
  float v = 0;
  for (int i = t; i < n; i += 256) v += a[i] * a[i + 1];
  s[t] = v;
  lds_barrier();
  for (int w = 128; w > 0; w >>= 1) {
    if (t < w) s[t] += s[t + w];
    lds_barrier();
  }
  unsigned old = __atomic_fetch_add(cnt, 1u, __ATOMIC_RELAXED);
  if (t == 0) o[0] = s[0] + (float)old;
}
