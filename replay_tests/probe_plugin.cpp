// HOST-ONLY negative fixtures for the replay infrastructure. Never run on a GPU.
#include "rad_plugin.h"
#include <cstring>
#include <thread>
#include <chrono>
#include <limits>
#include <cstdint>
static size_t n(const RadArgs* a){return (size_t)a->t[1].shape[0];}
static int copy_(const RadArgs* a,RadStream){std::memcpy(a->t[1].data,a->t[0].data,n(a));return RAD_OK;}
static int bad_(const RadArgs* a,RadStream){copy_(a,nullptr);((uint8_t*)a->t[1].data)[0]^=1;return RAD_OK;}
static int guard_(const RadArgs* a,RadStream){copy_(a,nullptr);((uint8_t*)a->t[1].data)[n(a)]=0;return RAD_OK;}
static int mutate_(const RadArgs* a,RadStream){copy_(a,nullptr);((uint8_t*)a->t[0].data)[0]^=1;return RAD_OK;}
static int indirect_(const RadArgs* a,RadStream){uint8_t* p=nullptr;std::memcpy(&p,a->t[0].data,sizeof p);std::memcpy(a->t[1].data,p,n(a));return RAD_OK;}
static int scratch_(const RadArgs* a,RadStream){if(a->scratch_bytes<16)return RAD_E_SHAPE;for(int i=0;i<16;++i)if(((uint8_t*)a->scratch)[i])return RAD_E_STATE;std::memset(a->scratch,0x11,16);return copy_(a,nullptr);}
static int scratchbad_(const RadArgs* a,RadStream){scratch_(a,nullptr);((uint8_t*)a->scratch)[16]=0;return RAD_OK;}
static int hang_(const RadArgs*,RadStream){std::this_thread::sleep_for(std::chrono::seconds(30));return RAD_OK;}
static int nan_(const RadArgs* a,RadStream){float x=std::numeric_limits<float>::quiet_NaN();std::memcpy(a->t[1].data,&x,4);return RAD_OK;}
static int inf_(const RadArgs* a,RadStream){float x=std::numeric_limits<float>::infinity();std::memcpy(a->t[1].data,&x,4);return RAD_OK;}
// Additional HOST-ONLY audit probes. Deliberately violate a logical kernel contract.
static volatile uint8_t audit_sink = 0;
static int audit_oob_read_(const RadArgs* a,RadStream){
    audit_sink = static_cast<volatile uint8_t*>(a->t[0].data)[n(a)];
    return copy_(a,nullptr);
}
static int audit_no_store_(const RadArgs*,RadStream){return RAD_OK;}
static int audit_scratch_read_(const RadArgs* a,RadStream){
    if(a->scratch_bytes<16)return RAD_E_SHAPE;
    const uint8_t stale = static_cast<volatile uint8_t*>(a->scratch)[0];
    for(size_t i=0;i<n(a);++i) static_cast<uint8_t*>(a->t[1].data)[i] = static_cast<uint8_t*>(a->t[0].data)[i] + stale;
    return RAD_OK;
}
static int scratch_overwrite_(const RadArgs* a,RadStream){std::memset(a->scratch,0,16);return audit_scratch_read_(a,nullptr);}
static int partial_store_(const RadArgs* a,RadStream){if(n(a)>1)std::memcpy(a->t[1].data,a->t[0].data,n(a)-1);return RAD_OK;}
static int prefix_read_(const RadArgs* a,RadStream){audit_sink=static_cast<volatile uint8_t*>(a->t[0].data)[-1];return copy_(a,nullptr);}
static int64_t scratchsize_(const RadArgs*){return 16;}
static int64_t axis_values[]={1,2};
static RadTunable axes[]={ {"tile",axis_values,2,1,"test axis"} };
static RadKernelInfo row(const char* name,RadLaunchFn f,bool scratch=false,const char* op="replay_probe"){
 RadKernelInfo r{};r.name=name;r.op=op;r.family="test";r.computes="HOST-ONLY replay infrastructure fixture";r.shape="flat";r.dtypes="u8/f32";r.domain=RAD_DOMAIN_HOST;r.launch=f;r.scratch=scratch?scratchsize_:nullptr;if(std::strcmp(name,"tuned")==0){r.tunables=axes;r.n_tunables=1;}return r;
}
static RadKernelInfo rows[]={row("copy",copy_),row("bad",bad_),row("guard",guard_),row("mutate",mutate_),row("scratch",scratch_,true),row("scratchbad",scratchbad_,true),row("hang",hang_),row("nan",nan_),row("inf",inf_),row("indirect",indirect_,false,"replay_indirect"),row("stateful",copy_),row("tuned",copy_),row("audit_oob_read",audit_oob_read_),row("audit_no_store",audit_no_store_),row("audit_scratch_read",audit_scratch_read_,true),row("scratch_overwrite",scratch_overwrite_,true),row("partial_store",partial_store_),row("prefix_read",prefix_read_)};
static RadOperandSpec direct[]={{"x",RAD_OPD_IN,0},{"out",RAD_OPD_OUT,0}};
static RadOperandSpec indirect[]={{"ptr",RAD_OPD_WTAB,0},{"out",RAD_OPD_OUT,0}};
static RadOpSchema schemas[]={{"replay_probe",nullptr,0,direct,2,"host fixture"},{"replay_indirect",nullptr,0,indirect,2,"host relocation fixture"}};
static RadPluginInfo info{RAD_PLUGIN_KERNEL,"replay_test","0.4.0","HOST-ONLY test plugin","host"};
extern "C" uint32_t rad_plugin_abi_version(){return RAD_ABI_VERSION;}
extern "C" const RadPluginInfo* rad_plugin_info(){return &info;}
extern "C" int rad_kernel_schema_count(){return 2;}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i){return i>=0&&i<2?&schemas[i]:nullptr;}
extern "C" int rad_kernel_count(){return sizeof rows/sizeof rows[0];}
extern "C" const RadKernelInfo* rad_kernel_at(int i){return i>=0&&i<rad_kernel_count()?&rows[i]:nullptr;}
extern "C" int rad_kernel_concurrent(int i){return i!=10;}
