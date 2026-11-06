#pragma once
// Opt-in kBench pre/post-operand exporter, not a serving-engine interceptor.
// Captures full bytes from known operands; WTAB/instance/collective paths fail closed.
#include "common.h"
#include "plugin/registry.h"
#include "rad_device.h"
#include "device/device.h"
#include <set>
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace rad_replay {
struct CaptureState {
    bool initialized=false;
    fs::path root;
    std::ofstream log;
    uint64_t serial=0, captured=0, max_bytes=256ull<<20, max_cases=100;
    std::mutex mutex;
};
inline CaptureState& capture_state() {static auto* s=new CaptureState;return *s;}
inline bool capture_enabled() {const char* s=std::getenv("RAD_REPLAY_CAPTURE_DIR");return s && *s;}
inline uint64_t capture_limit(const char* name,uint64_t fallback) {
    const char* s=std::getenv(name);if(!s || !*s) return fallback;
    std::string v=s;need(v.find_first_not_of("0123456789")==std::string::npos,"invalid capture limit");
    auto n=std::stoull(v);need(n>0 && n<(1ull<<43),"invalid capture limit");return n;
}
inline void initialize_capture(CaptureState& s) {
    if(s.initialized) return;
    s.root=std::getenv("RAD_REPLAY_CAPTURE_DIR");
    need(!fs::exists(s.root),"snapshot output already exists; use a fresh directory");
    fs::create_directories(s.root);fs::permissions(s.root,fs::perms::owner_all,fs::perm_options::replace);
    s.log.open(s.root/"capture.jsonl",std::ios::app);need(bool(s.log),"cannot create snapshot ledger");
    s.max_bytes=capture_limit("RAD_REPLAY_CAPTURE_MAX_BYTES",s.max_bytes);
    s.max_cases=capture_limit("RAD_REPLAY_CAPTURE_MAX_CASES",s.max_cases);
    s.initialized=true;
}
class Capture {
    struct Range {uintptr_t lo,hi;bool mut;std::string id;};
    bool active_=false;
    bool device_=false;
    RadStream stream_=nullptr;
    fs::path root_;
    Json m_;
    std::vector<Range> ranges_;
    std::vector<size_t> sizes_;
    std::vector<RadTensor> tensors_;
    const RadOpSchema* schema_=nullptr;
    uint64_t serial_=0;
    Bytes read(const void* p,size_t n) {
        Bytes b(n);
        if(device_) {
            need(stream_!=nullptr,"capture has no device stream");
            need(rad_memcpy_async(b.data(),p,(int64_t)n,stream_)>=0 && rad_stream_sync(stream_)>=0,"snapshot device read failed");
        } else if(n) std::memcpy(b.data(),p,n);
        return b;
    }
    void log(const std::string& status,const std::string& reason="") {
        auto& s=capture_state();std::lock_guard<std::mutex> lock(s.mutex);
        s.log<<Json{{"serial",serial_},{"status",status},{"reason",reason},{"package",root_.filename().string()}}.dump()<<'\n';s.log.flush();
        need(bool(s.log),"snapshot ledger write failed");
    }
public:
    Capture(const rad::Registry& reg,const rad::KernelRow* row,const RadArgs& a,
            const RadOpSchema* schema,const std::vector<size_t>& sizes,RadStream stream,
            const std::string& provenance) {
        if(!capture_enabled()) return;
        auto& state=capture_state();
        {
            std::lock_guard<std::mutex> lock(state.mutex);initialize_capture(state);
            serial_=++state.serial;
        }
        need(row && row->info && schema,"capture missing kernel/schema");
        if(state.captured>=state.max_cases){log("NOT_CAPTURED","configured case cap");return;}
        if(a.world_size>1 || a.rank!=0 || row->collective || row->releases_link) {log("NOT_CAPTURED","multi-rank protocol");return;}
        if((!row->concurrent && !row->reference) || a.instance || row->info->init || row->info->fini) {log("NOT_CAPTURED","hidden instance/lifecycle state");return;}
        if(a.n_t!=schema->n_operands || sizes.size()!=(size_t)a.n_t) {log("NOT_CAPTURED","schema/size mismatch");return;}
        if(a.scratch_bytes<0 || (uint64_t)a.scratch_bytes>state.max_bytes){log("NOT_CAPTURED","scratch budget");return;}
        uint64_t bytes=a.scratch_bytes;
        for(int i=0;i<a.n_t;++i) {
            if(!a.t[i].data) continue;
            if(schema->operands[i].role==RAD_OPD_WTAB){log("NOT_CAPTURED","WTAB needs explicit target-allocation relocation inventory");return;}
            if(!sizes[i] || sizes[i]>state.max_bytes || bytes>state.max_bytes-sizes[i]) {log("NOT_CAPTURED","snapshot byte budget or unknown span");return;}
            bytes+=sizes[i];
            auto lo=reinterpret_cast<uintptr_t>(a.t[i].data);
            need(lo+sizes[i]>=lo,"operand range overflow");
            int role=schema->operands[i].role;
            ranges_.push_back(Range{lo,lo+sizes[i],role==RAD_OPD_OUT || role==RAD_OPD_INOUT,""});
        }
        if(ranges_.empty()){log("NOT_CAPTURED","empty operands");return;}
        // Known operand intervals, not memory scanning, recover aliases/overlapping views.
        std::sort(ranges_.begin(),ranges_.end(),[](const Range& x,const Range& y){return x.lo<y.lo;});
        std::vector<Range> merged;
        for(auto r:ranges_) {
            if(!merged.empty() && r.lo<merged.back().hi) {
                merged.back().hi=std::max(merged.back().hi,r.hi);merged.back().mut|=r.mut;
            } else merged.push_back(r);
        }
        ranges_=std::move(merged);
        device_=row->info->domain==RAD_DOMAIN_DEVICE;stream_=stream;
        if(device_) {
            // No public current-device query in this ABI. Refuse heterogeneous visible
            // targets rather than infer an ordinal from operand addresses.
            bool supported=std::string(rad::device_backend_name())=="aql" && rad_dev_count()>0;
            for(int ordinal=0;ordinal<rad_dev_count();++ordinal) {
                RadDeviceProps props{};
                supported=supported && rad_dev_props(ordinal,&props)>=0 && std::string(props.arch)=="gfx1201";
            }
            if(!supported){log("NOT_CAPTURED","requires a homogeneous visible gfx1201 AQL device set");return;}
        }
        tensors_.assign(a.t,a.t+a.n_t);sizes_=sizes;schema_=schema;
        if(a.scratch_bytes && a.scratch) {
            auto b=read(a.scratch,a.scratch_bytes);
            if(!std::all_of(b.begin(),b.end(),[](uint8_t x){return x==0;})){log("NOT_CAPTURED","nonzero initial scratch state");return;}
        }
        std::ostringstream name;name<<"case-"<<std::setw(8)<<std::setfill('0')<<serial_;
        root_=state.root/name.str();need(fs::create_directory(root_),"snapshot collision");
        fs::permissions(root_,fs::perms::owner_all,fs::perm_options::replace);
        const auto* plugin=reg.plugin(row->plugin);need(plugin,"capture plugin not in registry");
        auto plugin_hash=hash_file(plugin->path);
        m_={{"schema","radiance-kernel-replay/1"},{"id",name.str()},
            {"provenance",{{"kind",provenance},{"reference","captured-baseline"},
                           {"note","Exact kBench invocation bytes; output is a regression baseline, not an independent oracle. Original kBench verdict is separate."}}},
            {"target",{{"backend",device_?"aql":"host"},{"arch",device_?"gfx1201":"host"},{"world_size",1}}},
            {"contract",{{"pointer_fields_complete",true},{"hidden_state","none"},{"execution","ordered-launchers"}}},
            {"plugins",Json::array({{{"id","p0"},{"name",row->plugin},{"abi",RAD_ABI_VERSION},{"sha256",plugin_hash}}})},
            {"allocations",Json::array()},{"views",Json::array()},{"nodes",Json::array()},{"checks",Json::array()},
            {"streams",Json::array({{{"id","q0"},{"priority",0}}})}};
        for(size_t i=0;i<ranges_.size();++i) {
            auto& r=ranges_[i];r.id="a"+std::to_string(i);
            // Preserve observed operand address residue up to 256 bytes via a prefix. It avoids
            // silently granting an unaligned view stronger alignment during replay.
            size_t pad=r.lo%256;Bytes raw(pad,0);auto bytes0=read(reinterpret_cast<const void*>(r.lo),r.hi-r.lo);raw.insert(raw.end(),bytes0.begin(),bytes0.end());
            m_["allocations"].push_back({{"id",r.id},{"kind",device_?"device":"host"},{"bytes",raw.size()},
                {"alignment",256},{"initial",save_blob(root_,raw)},{"mutable",r.mut},{"relocations",Json::array()}});
        }
        Json tensor_ids=Json::array(), accesses=Json::array(),params=Json::array();std::set<std::string> present;
        for(int i=0;i<a.n_t;++i) {
            if(!a.t[i].data){tensor_ids.push_back(nullptr);continue;}
            auto address=reinterpret_cast<uintptr_t>(a.t[i].data);const Range* found=nullptr;
            for(auto& r:ranges_) if(address>=r.lo && address+sizes[i]<=r.hi) {found=&r;break;}
            need(found,"operand not contained in recovered allocation");
            std::string id="t"+std::to_string(i);Json shape=Json::array(),stride=Json::array();
            for(uint32_t j=0;j<a.t[i].rank;++j){shape.push_back(a.t[i].shape[j]);stride.push_back(a.t[i].stride[j]);}
            m_["views"].push_back({{"id",id},{"allocation",found->id},{"offset",address-found->lo+found->lo%256},
                {"bytes",sizes[i]},{"dtype",a.t[i].dtype},{"shape",shape},{"stride",stride}});
            tensor_ids.push_back(id);int role=schema->operands[i].role;
            accesses.push_back({{"view",id},{"mode",role==RAD_OPD_OUT?"w":role==RAD_OPD_INOUT?"rw":"r"}});
        }
        for(int i=0;i<a.n_p;++i) {
            const auto& p=a.p[i];present.insert(p.key);
            if(p.kind==RAD_P_INT){need(std::string(p.key)!="chain" || !p.ival,"chained snapshot unsupported");params.push_back({{"key",p.key},{"kind","int"},{"value",p.ival}});}
            else if(p.kind==RAD_P_STR) params.push_back({{"key",p.key},{"kind","str"},{"value",p.sval}});
            else if(p.kind==RAD_P_F64) params.push_back({{"key",p.key},{"kind","f64"},{"value",p.dval}});
            else throw Incomplete("capture unresolved parameter");
        }
        for(int i=0;i<row->info->n_tunables;++i) {const auto& t=row->info->tunables[i];if(!present.count(t.key))params.push_back({{"key",t.key},{"kind","int"},{"value",t.deflt}});}
        m_["nodes"].push_back({{"id","kernel0"},{"kind","kernel"},{"stream","q0"},{"phase","measure"},
            {"plugin","p0"},{"row",row->info->name},{"op",row->info->op},{"params",params},
            {"tensors",tensor_ids},{"accesses",accesses},{"scratch_bytes",a.scratch_bytes}});
        active_=true;log("CAPTURING");
    }
    bool active()const{return active_;}
    void finish() {
        if(!active_) return;
        for(size_t i=0;i<tensors_.size();++i) {
            int role=schema_->operands[i].role;
            if(!tensors_[i].data || (role!=RAD_OPD_OUT && role!=RAD_OPD_INOUT))continue;
            auto b=read(tensors_[i].data,sizes_[i]);
            m_["checks"].push_back({{"view","t"+std::to_string(i)},{"expected",save_blob(root_,b)},
                {"mode","bytes"},{"atol",0},{"rtol",0},{"rel_l2",0},{"nonfinite","reject"}});
        }
        need(!m_["checks"].empty(),"captured invocation has no observable output");
        write_json(root_/"manifest.json",m_);active_=false;
        {auto& s=capture_state();std::lock_guard<std::mutex> lock(s.mutex);++s.captured;}
        log("CAPTURED");
    }
    ~Capture(){if(active_) {try{log("INCOMPLETE","invocation did not finalize snapshot");}catch(...){}}}
};
}
