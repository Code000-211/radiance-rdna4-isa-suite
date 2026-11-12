// Host-only stand-in for the few HIP runtime calls hwprobe/host_runner.cpp makes, so that
// tests/test_hwprobe.py can build and run the runner with g++ where no ROCm/GPU exists.
// NOT a GPU: "launching" kernel K of code object H copies the file H.K.mock (raw little-endian words,
// written by the test) into the output buffer; without that file the buffer stays as filled.
// Only the runner's parsing, comparison, JSON and exit codes are exercised this way.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

typedef int hipError_t;
enum { hipSuccess = 0, hipErrorNotFound = 500 };
enum hipMemcpyKind { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
struct hipDeviceProp_t { char name[256]; char gcnArchName[256]; };
struct hip_mock_module { std::string path; };
struct hip_mock_function { std::string path, name; };
typedef hip_mock_module *hipModule_t;
typedef hip_mock_function *hipFunction_t;
typedef void *hipStream_t;

inline std::map<void *, size_t> &hip_mock_sizes() { static std::map<void *, size_t> m; return m; }
inline const char *hipGetErrorString(hipError_t e) { return e == hipSuccess ? "hipSuccess" : "hip mock error"; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *p, int) {
  std::strcpy(p->name, "hip mock (no GPU)");
  const char *a = std::getenv("HWPROBE_MOCK_ARCH");
  std::strcpy(p->gcnArchName, a ? a : "gfx1201");
  return hipSuccess;
}
inline hipError_t hipModuleLoad(hipModule_t *m, const char *path) {
  if (FILE *f = std::fopen(path, "rb")) { std::fclose(f); *m = new hip_mock_module{path}; return hipSuccess; }
  return hipErrorNotFound;
}
inline hipError_t hipModuleUnload(hipModule_t m) { delete m; return hipSuccess; }
inline hipError_t hipModuleGetFunction(hipFunction_t *f, hipModule_t m, const char *name) {
  *f = new hip_mock_function{m->path, name};
  return hipSuccess;
}
inline hipError_t hipMalloc(void **p, size_t n) { *p = std::malloc(n); hip_mock_sizes()[*p] = n; return hipSuccess; }
inline hipError_t hipFree(void *p) { hip_mock_sizes().erase(p); std::free(p); return hipSuccess; }
inline hipError_t hipMemcpy(void *d, const void *s, size_t n, hipMemcpyKind) { std::memcpy(d, s, n); return hipSuccess; }
inline hipError_t hipMemset(void *d, int v, size_t n) { std::memset(d, v, n); return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                        unsigned, hipStream_t, void **params, void **) {
  void *out = *(void **)params[0];
  std::string mock = f->path + "." + f->name + ".mock";
  if (FILE *in = std::fopen(mock.c_str(), "rb")) {
    size_t n = std::fread(out, 1, hip_mock_sizes()[out], in);
    (void)n;
    std::fclose(in);
  }
  return hipSuccess;
}
