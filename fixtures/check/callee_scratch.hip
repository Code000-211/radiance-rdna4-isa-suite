#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define NI __attribute__((noinline))
#define __shared__ __attribute__((shared))
__device__ NI float leaf(const float* p, int i) { return p[i] * p[i + 32]; }
__device__ NI float mid(const float* p, int n, int i) {
  float acc = 0.f;
  if (i >= n) return -1.f;
  for (int j = 0; j < n; ++j) acc += leaf(p, (i + j) & 1023);
  return acc;
}
__device__ NI void lds_helper(float* q, int i, float v) {
  __shared__ float s[256];
  s[i & 255] = v;
  __builtin_amdgcn_s_barrier();
  q[i] = s[(i + 1) & 255];
}
__device__ NI int atom(int* c, int v) { return __atomic_fetch_add(c, v, __ATOMIC_RELAXED); }
__device__ NI double dbl(const double* d, int i) { return d[i] * 3.0 + d[i+1]; }
extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1,256)))
k(const float* p, float* q, int n, int* c, const double* d) {
  int i = __builtin_amdgcn_workitem_id_x();
  float v = mid(p, n, i);
  v += (float)atom(c, i);
  v += (float)dbl(d, i);
  lds_helper(q, i, v);
}
