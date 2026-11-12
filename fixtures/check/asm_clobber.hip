// Self-contained: builds with -nogpuinc, no ROCm headers needed.
// Inline asm that writes registers it does not declare (ASM_UNDECLARED_CLOBBER).
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

#define ZERO_V9() __asm__ __volatile__("v_mov_b32 v9, 0")

__device__ __attribute__((always_inline)) inline int helper(int a) {
  int r;
  asm volatile("v_mov_b32 v8, %1\n\t"
               "v_add_nc_u32 %0, v8, 1" : "=v"(r) : "v"(a));
  return r;
}

// BUG: v7, SCC and VCC are written but not declared; $1 is an input and is overwritten.
extern "C" __global__ void undeclared(int* o, int a) {
  int v, w;
  asm volatile("v_mov_b32 v7, %1\n\t"
               "v_add_nc_u32 %0, v7, %1" : "=v"(v) : "v"(a));
  __asm__ __volatile__("s_add_u32 %0, %1, 1\n"
                       "v_cmp_eq_u32 0, %2" : "=s"(w) : "s"(a), "v"(v));
  __asm volatile("v_mov_b32 %0, 0" :: "v"(a));
  ZERO_V9();
  o[0] = v + w + helper(a);
}

// Fixed: every written register is an output or a clobber.
extern "C" __global__ void declared(int* o, int a) {
  int v, w, t = a;
  asm volatile("v_mov_b32 v7, %1\n\t"
               "v_add_nc_u32 %0, v7, %1" : "=v"(v) : "v"(a) : "v7");
  __asm__ __volatile__("s_add_u32 %0, %1, 1\n"
                       "v_cmp_eq_u32 0, %2" : "=s"(w) : "s"(a), "v"(v) : "scc", "vcc");
  __asm volatile("v_add_nc_u32 %0, %0, 1" : "+v"(t));
  int r;
  asm volatile("v_mov_b32 %0, 1" : "={v1}"(r));
  o[0] = v + w + t + r;
}
