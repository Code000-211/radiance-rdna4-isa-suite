// Compiled with -mllvm -amdgpu-s-branch-bits=5 (tests/test_hazards.py) so BranchRelaxation emits long branches.
#define __global__ __attribute__((global))
extern "C" __global__ void lb(float* p, int n) {
  float a = p[0];
  for (int i = 0; i < n; ++i) { a = a * 1.5f + p[i]; if (a > 3.0f) a = __builtin_sqrtf(a) * p[i+1]; }
  p[1] = a;
}
