; CPU-generated fixture. Never executed on a GPU by this suite.
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @smoke(ptr addrspace(1) %out) {
entry:
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}
