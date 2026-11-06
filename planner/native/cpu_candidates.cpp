// CPU-only comparison candidates for the planner experiment. These are NOT R9700
// kernel emulations. Real Radiance ABI, contiguous f32 GEMM NT, no hidden state.
#include "rad_plugin.h"
#include <algorithm>
#include <cstddef>

template<int PANEL>
static int panel(const RadArgs* a, RadStream) {
    if (!a || a->n_t!=3) return RAD_E_INVAL;
    const auto M=rad_args_geti_or(a,"M",0), N=rad_args_geti_or(a,"N",0), K=rad_args_geti_or(a,"K",0);
    if (M<1 || N<1 || K<1) return RAD_E_SHAPE;
    for (int t=0;t<3;++t) if(a->t[t].dtype!=RAD_F32 || !a->t[t].data || a->t[t].rank!=2 || a->t[t].stride[1]!=1) return RAD_E_SHAPE;
    if (a->t[0].shape[0]!=M || a->t[0].shape[1]!=K || a->t[1].shape[0]!=N || a->t[1].shape[1]!=K ||
        a->t[2].shape[0]!=M || a->t[2].shape[1]!=N) return RAD_E_SHAPE;
    const float* A=static_cast<const float*>(a->t[0].data);
    const float* B=static_cast<const float*>(a->t[1].data);
    float* C=static_cast<float*>(a->t[2].data);
    for (long long m=0;m<M;++m) for(long long n=0;n<N;n+=PANEL) {
        float acc[PANEL]={};
        const int width=static_cast<int>(std::min<long long>(PANEL,N-n));
        for(long long k=0;k<K;++k) {
            const float x=A[m*a->t[0].stride[0]+k];
            for(int j=0;j<width;++j) acc[j] += x*B[(n+j)*a->t[1].stride[0]+k];
        }
        for(int j=0;j<width;++j) C[m*a->t[2].stride[0]+n+j]=acc[j];
    }
    return RAD_OK;
}
static RadKernelInfo make(const char* name, RadLaunchFn f) {
    RadKernelInfo r{};r.name=name;r.op="gemm_nt";r.family="planner_cpu_demo";
    r.computes="CPU-only f32 GEMM candidate; not GPU emulation";r.shape="contiguous rank-2";r.dtypes="f32";
    r.domain=RAD_DOMAIN_HOST;r.launch=f;return r;
}
static RadKernelInfo rows[]={make("cpu_panel1",panel<1>),make("cpu_panel4",panel<4>),make("cpu_panel16",panel<16>)};
static RadParamSpec params[]={{"M",RAD_P_INT,1,0},{"N",RAD_P_INT,1,0},{"K",RAD_P_INT,1,0},{"dtype",RAD_P_STR,1,0}};
static RadOperandSpec operands[]={{"a",RAD_OPD_IN,0},{"b",RAD_OPD_IN,0},{"y",RAD_OPD_OUT,0}};
static RadOpSchema schema={"gemm_nt",params,4,operands,3,"CPU f32 C=A@B.T"};
static RadPluginInfo info{RAD_PLUGIN_KERNEL,"planner_cpu","0.5.0","Planner CPU calibration experiment only","host"};
extern "C" uint32_t rad_plugin_abi_version(){return RAD_ABI_VERSION;}
extern "C" const RadPluginInfo* rad_plugin_info(){return &info;}
extern "C" int rad_kernel_schema_count(){return 1;}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i){return i==0?&schema:nullptr;}
extern "C" int rad_kernel_count(){return 3;}
extern "C" const RadKernelInfo* rad_kernel_at(int i){return i>=0&&i<3?&rows[i]:nullptr;}
extern "C" int rad_kernel_concurrent(int i){return i>=0&&i<3;}
