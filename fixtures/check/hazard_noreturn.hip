// Unedited clang output must not report CFG_FALLTHROUGH / REACHABLE_CODE_END (tests/test_hazards.py):
// die() ends in s_trap 2 (llvm.trap); kn ends in the s_swappc_b64 to die().
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
__device__ __attribute__((noinline, noreturn)) void die(int* p) { p[0] = 1; __builtin_trap(); }
__device__ __attribute__((noinline)) int chk(int* p, int x) { if (x > 5) die(p); return p[x]; }
extern "C" __global__ void kn(int* p, int x) { if (x < 0) die(p); p[1] = chk(p, x); }
