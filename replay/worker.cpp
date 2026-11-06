// Engine-free launcher replay, linked to Radiance's existing registry/device backend.
// Internal worker: invoke through replay.sh, which validates/stages the package first.
#include "common.h"
#include "diagnostics.h"
#include "dispatch_trace.h"
#include "device/device.h"
#include "plugin/registry.h"
#include "rad_plugin.h"
#include "kbench_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>

using namespace rad_replay;
namespace {
void ok(int rc, const std::string& what) {
    if(rc<0) {
        const char* err=rad_dev_last_error();
        throw Incomplete(what+": "+rad_strerror(rc)+(err?"; "+std::string(err):""));
    }
}
struct Allocation {
    void* base=nullptr;
    uint8_t* data=nullptr;
    size_t bytes=0, total=0, guard=256;
    int kind=RAD_MEM_HOST;
    bool mutable_data=false;
    Bytes pristine;
};
struct Operation {
    enum Kind { Kernel, Copy, Fill, Record, Wait } kind=Kernel;
    bool measured=false;
    RadStream stream=nullptr;
    RadEvent event=nullptr;
    void* src=nullptr;
    void* dst=nullptr;
    int fill=0;
    size_t bytes=0;
    std::string id;
    Json desc;
    const rad::KernelRow* row=nullptr;
    rad::Geometry geom;
    std::vector<RadTensor> tensors;
    Allocation scratch;
};
struct Check { Json desc; RadTensor tensor{}; uint8_t* data=nullptr; Bytes expected; };

// Process-lifetime objects deliberately do not drain/free queues on error. The supervisor
// kills a timed-out process; an error path must not deadlock again in a destructor.
struct Worker {
    Json plan, m, result;
    fs::path out;
    bool gpu=false;
    rad::Registry* registry=new rad::Registry;
    std::map<std::string,Allocation> allocations;
    std::map<std::string,Json> views;
    std::map<std::string,RadStream> streams;
    std::map<std::string,RadEvent> events;
    std::map<std::string,std::string> plugin_names;
    std::vector<Operation> ops;
    std::vector<Check> checks;
    std::map<std::string,std::vector<Check>> checkpoints;
    std::map<std::string,Json> baseline_checkpoints;
    bool memory_check=false;
    bool trace_dispatches=false;
    size_t diagnostic_output_bytes=0;
    std::map<std::string,std::vector<std::pair<size_t,size_t>>> writable;
    RadStream control=nullptr;
    RadEvent start=nullptr,stop=nullptr;
    std::map<std::string,RadEvent> ready,done;
    std::ofstream progress;
    uint64_t budget=0, used=0;

    Bytes get_blob(const Json& b) {
        auto p=plan.at("blob_paths").at(b.at("sha256").get<std::string>()).get<std::string>();
        Bytes raw=read_bytes(p,budget);
        need(raw.size()==b.at("bytes").get<size_t>() && hash(raw)==b.at("sha256").get<std::string>(),"staged blob changed");
        return raw;
    }
    void copy(void* dst,const void* src,size_t n) {
        if(gpu) { ok(rad_memcpy_async(dst,src,(int64_t)n,control),"read/write snapshot");ok(rad_stream_sync(control),"snapshot sync"); }
        else if(n) std::memcpy(dst,src,n);
    }
    Allocation allocate(size_t bytes,size_t align,int kind,bool mut) {
        need(bytes>0 && bytes<=budget && align>=8 && align<=65536 && !(align&(align-1)),"bad allocation size/alignment");
        Allocation a; a.bytes=bytes;a.kind=kind;a.mutable_data=mut;
        a.total=bytes+512+align;
        need(a.total>=bytes && used<=budget && a.total<=budget-used,"physical allocation/redzone budget exceeded"); used+=a.total;
        if(kind==RAD_MEM_HOST) { void* b=nullptr;need(posix_memalign(&b,align,a.total)==0,"host allocation failed");a.base=b; }
        else a.base=rad_dev_alloc((int64_t)a.total,kind);
        need(a.base!=nullptr,"device allocation failed");
        auto address=(reinterpret_cast<uintptr_t>(a.base)+256+align-1)&~uintptr_t(align-1);
        a.data=reinterpret_cast<uint8_t*>(address);
        a.pristine.assign(bytes,0);
        return a;
    }
    void reset(Allocation& a) {
        if(!a.base) return;
        if(gpu) { ok(rad_memset_async(a.base,0xa5,(int64_t)a.total,control),"poison guards");ok(rad_stream_sync(control),"poison sync"); }
        else std::memset(a.base,0xa5,a.total);
        copy(a.data,a.pristine.data(),a.bytes);
    }
    RadTensor tensor(const std::string& id) {
        auto& v=views.at(id);auto& a=allocations.at(v.at("allocation").get<std::string>());
        auto off=v.at("offset").get<size_t>(), n=v.at("bytes").get<size_t>();
        need(off<=a.bytes && n<=a.bytes-off,"view exceeds physical allocation");
        RadTensor t{};t.data=a.data+off;t.dtype=v.at("dtype");
        auto sh=v.at("shape").get<std::vector<int64_t>>(),st=v.at("stride").get<std::vector<int64_t>>();
        need(!sh.empty() && sh.size()<=RAD_MAX_RANK && st.size()==sh.size(),"invalid tensor shape");
        t.rank=sh.size();
        for(size_t i=0;i<sh.size();++i) { t.shape[i]=sh[i];t.stride[i]=st[i]; }
        return t;
    }
    std::string access_for(const Json& n,const std::string& view) {
        for(const auto& a:n.at("accesses")) if(a.at("view")==view) return a.at("mode");
        throw Incomplete("tensor missing from access contract");
    }
    void load() {
        need(plan.at("schema")=="radiance-replay-worker/1","bad internal plan");
        m=plan.at("fixture");out=plan.at("out").get<std::string>();
        need(m.at("schema")=="radiance-kernel-replay/1","bad fixture schema");
        budget=plan.at("max_bytes").get<uint64_t>();
        trace_dispatches=plan.value("trace_dispatches",false);
        need(!trace_dispatches || (m.at("target").at("backend")=="aql" && plan.value("diagnostic_only",false)),
             "dispatch tracing is AQL diagnostic-only, not a benchmark mode");
        if(trace_dispatches)rad_replay_trace::begin(plan.at("allowed_dispatches").get<std::set<std::string>>());
        memory_check=plan.value("memory_check",std::string("none"))=="host-asan";
        need(!memory_check || (RAD_REPLAY_HAS_ASAN && m.at("target").at("backend")=="host"),
             "host ASan requires an instrumented host worker; GPU read checking is not implemented");
        need(!memory_check || (plan.value("diagnostics",std::string("none"))=="contracts" && plan.value("diagnostic_only",false)),
             "memory checking must be diagnostic-only");
        need(budget>0 && budget<(1ull<<45),"bad memory budget");
        need(m.at("target").at("world_size")==1,"multi-rank replay unsupported");
        need(m.at("contract").at("execution")=="ordered-launchers" && m.at("contract").at("hidden_state")=="none" && m.at("contract").at("pointer_fields_complete")==true,"unsupported execution contract");
        gpu=m.at("target").at("backend")=="aql";
        result={{"schema","radiance-replay-worker-result/1"},{"status","INCOMPLETE"},
                {"fixture_sha256",plan.at("fixture_sha256")},{"backend",rad::device_backend_name()},
                {"samples_us",Json::array()},{"checked_iterations",0},{"checks_per_iteration",m.at("checks").size()}};
        need((gpu && !rad::device_is_host() && std::string(rad::device_backend_name())=="aql") ||
             (!gpu && rad::device_is_host()),"requested backend not present; silent fallback refused");
        int device=plan.at("device");need(device>=0,"negative device");ok(rad_dev_set(device),"set device");
        RadDeviceProps props{};ok(rad_dev_props(device,&props),"device properties");
        need(std::string(props.arch)==m.at("target").at("arch").get<std::string>(),"device architecture differs from fixture");
        result["device"]={{"ordinal",device},{"name",props.name},{"arch",props.arch},{"n_cu",props.n_cu},{"vram_bytes",props.vram_bytes}};
        result["device"]["pci_address"]=rad::dev_pci_address(device);
        result["device"]["runtime_version"]=rad::dev_runtime_version();
        result["device"]["dma_lane"]=rad::dev_dma_lane(device);
        result["worker_build"]={{"compiler",__VERSION__},{"cxx",__cplusplus},{"radiance_abi",RAD_ABI_VERSION},{"host_asan",bool(RAD_REPLAY_HAS_ASAN)}};
        registry->set_code_device(device);
        int order=0;
        for(const auto& p:m.at("plugins")) {
            auto id=p.at("id").get<std::string>();auto path=plan.at("plugin_paths").at(id).get<std::string>();
            need(hash_file(path)==p.at("sha256").get<std::string>(),"plugin hash changed before load");
            need(p.at("abi")==RAD_ABI_VERSION,"fixture plugin ABI mismatch");
            need(registry->load_plugin(path,order++)==RAD_OK,"plugin declined or failed to load");
            auto name=p.at("name").get<std::string>();need(registry->plugin(name)!=nullptr,"plugin identity mismatch");
            plugin_names[id]=name;
        }
        ok(registry->check_complete(),"plugin schema inventory");
        for(const auto& s:m.at("streams")) {
            RadStream q=nullptr;ok(rad_stream_create(&q,s.at("priority")),"create stream");streams[s.at("id")]=q;
        }
        need(!streams.empty(),"empty stream inventory");
        ok(rad_stream_create(&control,0),"create control stream");
        progress.open(out/"progress.jsonl",std::ios::out|std::ios::app);need(bool(progress),"progress log failed");
        for(const auto& a:m.at("allocations")) {
            auto kind=a.at("kind").get<std::string>();
            need(gpu ? kind=="device" || kind=="host_pinned" : kind=="host","unsupported memory kind");
            int k=kind=="device"?RAD_MEM_DEVICE:kind=="host_pinned"?RAD_MEM_HOST_PINNED:RAD_MEM_HOST;
            auto alloc=allocate(a.at("bytes"),a.at("alignment"),k,a.at("mutable"));
            alloc.pristine=get_blob(a.at("initial"));need(alloc.pristine.size()==alloc.bytes,"allocation snapshot size mismatch");
            need(allocations.emplace(a.at("id").get<std::string>(),std::move(alloc)).second,"duplicate allocation");
        }
        for(const auto& a:m.at("allocations")) {
            auto& dst=allocations.at(a.at("id"));
            for(const auto& r:a.at("relocations")) {
                need(!dst.mutable_data,"mutable pointer table unsupported");
                auto& src=allocations.at(r.at("target"));size_t off=r.at("offset"),add=r.at("addend");
                need(dst.bytes>=8 && off<=dst.bytes-8 && !(off%8) && add<src.bytes,"invalid relocation");
                uint64_t p=reinterpret_cast<uintptr_t>(src.data+add);
                std::memcpy(dst.pristine.data()+off,&p,8);
            }
        }
        for(const auto& v:m.at("views")) need(views.emplace(v.at("id"),v).second,"duplicate view");
        for(const auto& n:m.at("nodes")) {
            Operation op;op.desc=n;op.id=n.at("id");op.measured=n.at("phase")=="measure";op.stream=streams.at(n.at("stream"));
            auto record_write=[&](const std::string& id) {auto& v=views.at(id);size_t off=v.at("offset");writable[v.at("allocation")].emplace_back(off,off+v.at("bytes").get<size_t>());};
            if(n.at("kind")=="kernel") {
                for(const auto& a:n.at("accesses")) if(a.at("mode").get<std::string>().find('w')!=std::string::npos) record_write(a.at("view"));
                for(const auto& p:n.at("params")) {
                    auto key=p.at("key").get<std::string>();
                    if(p.at("kind")=="int") op.geom.set_i(key,p.at("value").get<int64_t>());
                    else if(p.at("kind")=="str") op.geom.set_s(key,p.at("value").get<std::string>());
                    else if(p.at("kind")=="f64") op.geom.set_f(key,p.at("value").get<double>());
                    else throw Incomplete("unresolved parameter kind");
                }
                long long chain=0;need(!op.geom.get_i("chain",&chain) || chain==0,"device-side chain replay unsupported");
                auto name=n.at("row").get<std::string>();auto opname=n.at("op").get<std::string>();
                rad::Registry::RowFilter filter=[&](const rad::KernelRow& r,const rad::Geometry&){return r.info && name==r.info->name?"":"not captured row";};
                rad::SelectionTrace trace;
                op.row=registry->select_in(plugin_names.at(n.at("plugin")),opname,op.geom,gpu?RAD_DOMAIN_DEVICE:RAD_DOMAIN_HOST,&filter,&trace);
                need(op.row && op.row->info && op.row->info->launch,"captured row not selectable: "+name+"; "+trace.reason);
                need(!op.row->collective && !op.row->releases_link,"collective/link protocol requires a multi-rank runner");
                need(!op.row->info->init && !op.row->info->fini,"instance lifecycle/state unsupported in v0.4");
                need(op.row->concurrent || (!gpu && streams.size()==1 && op.row->reference),"row has no stateless/concurrent promise (single-stream host reference excepted)");
                std::vector<RadParam> choice;
                for(int ti=0;ti<op.row->info->n_tunables;++ti) {
                    const auto& t=op.row->info->tunables[ti];long long value=0;
                    need(op.geom.get_i(t.key,&value),"snapshot must freeze every tunable");
                    bool valid=false;for(int vi=0;vi<t.n_values;++vi) valid|=t.values[vi]==value;
                    need(valid,"tunable value outside declared axis");
                    choice.push_back(RadParam{t.key,RAD_P_INT,value,0,nullptr,0});
                }
                need(!op.row->info->tune_valid || op.row->info->tune_valid(op.geom.params(),op.geom.n_params(),choice.data(),(int)choice.size())==1,"invalid tuned variant");
                for(auto& d:trace.derived) need(d.known && op.geom.has(d.key),"captured geometry missing derived parameter");
                for(const auto& v:n.at("tensors")) op.tensors.push_back(v.is_null()?RadTensor{}:tensor(v.get<std::string>()));
                const auto* schema=registry->schema(opname);need(schema && schema->n_operands==(int)op.tensors.size(),"operand schema mismatch");
                // Tunable axes are row-local, not op-schema geometry. Validate the two
                // vocabularies separately; otherwise legitimate tile/sk knobs get rejected.
                std::vector<RadParam> geometry_params;
                for(int pi=0;pi<op.geom.n_params();++pi) {
                    const auto& param=op.geom.params()[pi];bool in_schema=false,in_tuning=false;
                    for(int si=0;si<schema->n_params;++si)in_schema|=std::strcmp(param.key,schema->params[si].key)==0;
                    for(int ti=0;ti<op.row->info->n_tunables;++ti)in_tuning|=std::strcmp(param.key,op.row->info->tunables[ti].key)==0;
                    if(in_schema)geometry_params.push_back(param);
                    else need(in_tuning || (std::string(param.key)=="chain" && param.kind==RAD_P_INT && param.ival==0),"unknown parameter: "+std::string(param.key));
                }
                std::string validation_error;
                int valid_geometry=registry->validate(opname,geometry_params.data(),(int)geometry_params.size(),&validation_error);
                need(valid_geometry==RAD_OK,"invalid geometry: "+validation_error);
                for(size_t i=0;i<op.tensors.size();++i) {
                    if(!op.tensors[i].data) { need(schema->operands[i].optional,"required operand absent");continue; }
                    auto mode=access_for(n,n.at("tensors")[i]);int role=schema->operands[i].role;
                    if(role==RAD_OPD_OUT) need(mode.find('w')!=std::string::npos,"output not declared writable");
                    else if(role==RAD_OPD_INOUT) need(mode=="rw","inout not declared read/write");
                    else need(mode.find('r')!=std::string::npos,"input not declared readable");
                }
                RadArgs args{op.tensors.data(),(int)op.tensors.size(),op.geom.params(),op.geom.n_params(),nullptr,0,nullptr,0,1};
                int64_t want=op.row->info->scratch?op.row->info->scratch(&args):0;
                int64_t supplied=n.at("scratch_bytes");
                need(want>=0 && supplied>=want,"captured scratch smaller than current hook requires");
                if(supplied) op.scratch=allocate(supplied,256,gpu?RAD_MEM_DEVICE:RAD_MEM_HOST,true);
            } else if(n.at("kind")=="record") {
                op.kind=Operation::Record;
                RadEvent e=nullptr;ok(rad_event_create(&e),"create event");
                need(events.emplace(n.at("event"),e).second,"duplicate event generation");op.event=e;
            } else if(n.at("kind")=="wait") {
                op.kind=Operation::Wait;op.event=events.at(n.at("event"));
            } else if(n.at("kind")=="copy" || n.at("kind")=="fill") {
                op.dst=tensor(n.at("dst")).data;op.bytes=n.at("bytes");
                const auto& dv=views.at(n.at("dst"));size_t begin=dv.at("offset");
                writable[dv.at("allocation")].emplace_back(begin,begin+op.bytes);
                if(n.at("kind")=="copy") {op.kind=Operation::Copy;op.src=tensor(n.at("src")).data;}
                else {op.kind=Operation::Fill;op.fill=n.at("value");}
            } else throw Incomplete("unknown node kind");
            ops.push_back(std::move(op));
        }
        for(const auto& c:m.at("checks")) {
            Check check;check.desc=c;check.tensor=tensor(c.at("view"));check.data=static_cast<uint8_t*>(check.tensor.data);
            check.expected=get_blob(c.at("expected"));need(check.expected.size()==views.at(c.at("view")).at("bytes"),"check size mismatch");
            checks.push_back(std::move(check));
        }
        for(const auto& cp:m.value("checkpoints",Json::array())) {
            for(const auto& c:cp.at("checks")) {
                Check check;check.desc=c;check.tensor=tensor(c.at("view"));
                check.data=static_cast<uint8_t*>(check.tensor.data);check.expected=get_blob(c.at("expected"));
                checkpoints[cp.at("after")].push_back(std::move(check));
            }
        }
        need(checkpoints.empty() || plan.value("diagnostics",std::string("none"))=="contracts",
             "explicit checkpoints require contract diagnostics");
        need(!checks.empty(),"no checks");
        ok(rad_event_create(&start),"start event");ok(rad_event_create(&stop),"stop event");
        for(const auto& [id,q]:streams) { ok(rad_event_create(&ready[id]),"ready event");ok(rad_event_create(&done[id]),"done event"); }
    }
    void issue(Operation& o) {
        // Typed plan: no JSON parsing, symbol resolution, logging or allocation on this path.
        switch(o.kind) {
        case Operation::Kernel: {
            RadArgs a{o.tensors.data(),(int)o.tensors.size(),o.geom.params(),o.geom.n_params(),o.scratch.data,(int64_t)o.scratch.bytes,nullptr,0,1};
            int rc=o.row->info->launch(&a,gpu?o.stream:nullptr);
            if(rc!=RAD_OK) {ok(rc,"launch "+o.id);throw Incomplete("unexpected positive/link-ended launch status: "+o.id);}
            break;
        }
        case Operation::Record: ok(rad_event_record(o.event,o.stream),"record event");break;
        case Operation::Wait: ok(rad_event_wait(o.stream,o.event),"wait event");break;
        case Operation::Copy:
            if(gpu) ok(rad_memcpy_async(o.dst,o.src,o.bytes,o.stream),"copy node");
            else std::memcpy(o.dst,o.src,o.bytes);
            break;
        case Operation::Fill:
            if(gpu) ok(rad_memset_async(o.dst,o.fill,o.bytes,o.stream),"fill node");
            else std::memset(o.dst,o.fill,o.bytes);
            break;
        }
    }
    void guard_check(Allocation& a,const std::string& name) {
        if(!a.base) return;
        Bytes lo(a.guard),hi(a.guard);
        copy(lo.data(),a.data-a.guard,a.guard);copy(hi.data(),a.data+a.bytes,a.guard);
        auto check=[](const Bytes& v){return std::all_of(v.begin(),v.end(),[](uint8_t b){return b==0xa5;});};
        if(!check(lo)||!check(hi)) throw Failure("red-zone write: "+name);
        if(!a.mutable_data) {
            Bytes b(a.bytes);copy(b.data(),a.data,a.bytes);
            if(b!=a.pristine) throw Failure("immutable allocation changed: "+name);
        }
    }
    void compare_list(const std::vector<Check>& list) {
        for(const auto& c:list) {
            Bytes b(c.expected.size());copy(b.data(),c.data,b.size());
            auto what=c.desc.at("view").get<std::string>();
            if(c.desc.at("mode")=="bytes") {
                if(b!=c.expected) throw Failure("byte comparison failed: "+what);
            } else {
                size_t count=(size_t)rad_tensor_numel(&c.tensor);need(count>0,"empty numeric output");
                std::vector<float> got(count),ref(count);
                bool match=c.desc.at("nonfinite")=="match";
                double atol=c.desc.at("atol"),rtol=c.desc.at("rtol"),l2=c.desc.at("rel_l2");
                for(size_t i=0;i<count;++i) {
                    got[i]=rad_load_f32(b.data(),c.tensor.dtype,i);ref[i]=rad_load_f32(c.expected.data(),c.tensor.dtype,i);
                    if(!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
                        bool same=(std::isnan(got[i]) && std::isnan(ref[i])) || got[i]==ref[i];
                        if(!match || !same) throw Failure("nonfinite mismatch: "+what);
                    } else if(std::abs(double(got[i])-ref[i])>atol+rtol*std::abs(double(ref[i]))) {
                        throw Failure("elementwise numeric comparison failed: "+what+" at "+std::to_string(i));
                    }
                }
                long double diff2=0,ref2=0;
                for(size_t i=0;i<count;++i) if(std::isfinite(got[i]) && std::isfinite(ref[i])) {
                    long double d=(long double)got[i]-ref[i];diff2+=d*d;ref2+=(long double)ref[i]*ref[i];
                }
                // Matches the reference-zero convention of kBench; nonfinite values were checked above.
                const long double relative=ref2>0?std::sqrt(diff2/ref2):(diff2>0?1.0L:0.0L);
                if(!std::isfinite(relative) || relative>l2) throw Failure("relative L2 comparison failed: "+what);
            }
        }
    }
    void compare_checks() {
        compare_list(checks);
        for(auto& [id,a]:allocations) {
            guard_check(a,id);
            if(a.mutable_data) {
                Bytes b(a.bytes);copy(b.data(),a.data,a.bytes);
                auto intervals=writable[id];std::sort(intervals.begin(),intervals.end());size_t pos=0;
                // Bound first: clang before 16 cannot capture a structured binding in a lambda.
                const auto& pristine=a.pristine;
                auto same=[&](size_t begin,size_t end) {return std::equal(b.begin()+begin,b.begin()+end,pristine.begin()+begin);};
                for(auto [begin,end]:intervals) {
                    if(begin>pos && !same(pos,begin))throw Failure("undeclared write inside allocation: "+id);
                    pos=std::max(pos,end);
                }
                if(pos<a.bytes && !same(pos,a.bytes))throw Failure("undeclared write inside allocation: "+id);
            }
        }
        for(auto& o:ops) guard_check(o.scratch,o.desc.at("id").get<std::string>()+" scratch");
    }

    using RangeMap=std::map<std::string,Ranges>;
    RangeMap ranges(const Operation& o,char kind) {
        RangeMap r;
        auto add=[&](const std::string& id,size_t size=0) {
            const auto& v=views.at(id);size_t lo=v.at("offset");
            r[v.at("allocation")].emplace_back(lo,lo+(size?size:v.at("bytes").get<size_t>()));
        };
        if(o.kind==Operation::Kernel) {
            for(const auto& a:o.desc.at("accesses"))
                if(kind=='a' || a.at("mode").get<std::string>().find(kind)!=std::string::npos)add(a.at("view"));
        } else if(o.kind==Operation::Copy || o.kind==Operation::Fill) {
            if(kind=='a' || kind=='w')add(o.desc.at("dst"),o.bytes);
            if(o.kind==Operation::Copy && (kind=='a' || kind=='r'))add(o.desc.at("src"),o.bytes);
        }
        for(auto& [id,rr]:r)rr=merge_ranges(std::move(rr));
        return r;
    }
    bool packed_view(const Json& v) {
        auto sh=v.at("shape").get<std::vector<size_t>>(),st=v.at("stride").get<std::vector<size_t>>();
        size_t stride=1;
        for(size_t i=sh.size();i-->0;) {if(sh[i]!=1 && st[i]!=stride)return false;stride*=sh[i];}
        // Do not claim full overwrite of padding, custom encodings or sub-byte rows.
        int bits=0;switch(v.at("dtype").get<int>()) {
        case 1:case 11:case 13:bits=32;break;
        case 2:case 3:case 10:case 37:bits=16;break;
        case 4:case 5:case 8:case 9:case 14:case 21:bits=8;break;
        case 12:bits=64;break;default:return false;
        }
        return stride*(size_t(bits)/8)==v.at("bytes").get<size_t>();
    }
    RangeMap output_ranges(const Operation& o) {
        RangeMap r, reads=ranges(o,'r');
        if(o.kind==Operation::Kernel) {
            const auto* schema=registry->schema(o.desc.at("op").get<std::string>());
            for(size_t i=0;i<o.tensors.size();++i) {
                if(!o.tensors[i].data || schema->operands[i].role!=RAD_OPD_OUT)continue;
                const auto& v=views.at(o.desc.at("tensors")[i]);
                if(!packed_view(v))continue;
                size_t lo=v.at("offset");r[v.at("allocation")].emplace_back(lo,lo+v.at("bytes").get<size_t>());
            }
        } else if(o.kind==Operation::Copy || o.kind==Operation::Fill)r=ranges(o,'w');
        for(auto& [id,rr]:r)rr=subtract_ranges(rr,reads[id]); // preserve IN/INOUT aliases
        return r;
    }
    void sync_all() {
        for(auto [id,q]:streams)ok(rad_stream_sync(q),"diagnostic completion: "+id);
        ok(rad_stream_sync(control),"diagnostic control completion");
    }
    void poison_outputs(Operation& o,unsigned pattern) {
        for(auto& [id,rr]:output_ranges(o)) {
            auto& a=allocations.at(id);
            for(auto [lo,hi]:rr) {
                // Complementary patterns: no byte can remain correct in both passes by coincidence.
                Bytes b(hi-lo);
                for(size_t i=0;i<b.size();++i)b[i]=uint8_t((i+lo)*131u+0x39u) ^ uint8_t(pattern==1?0:0xff);
                copy(a.data+lo,b.data(),b.size());diagnostic_output_bytes+=b.size();
            }
        }
    }
    void arm_bounds(Operation& o) {
        if(!memory_check)return;
        auto allowed=ranges(o,'a');
        for(auto& [id,a]:allocations) {
            host_poison(a.base,a.total);
            for(auto [lo,hi]:allowed[id]) {
                // ASan shadow cannot represent arbitrary unaligned lower interval boundaries.
                need(lo%8==0,"ASan exact view lower bound is not 8-byte aligned: "+id);
                host_unpoison(a.data+lo,hi-lo);
                need(!host_is_poisoned(a.data+lo) && !host_is_poisoned(a.data+hi-1),"ASan failed to enable declared view");
            }
            for(auto [lo,hi]:subtract_ranges({{0,a.bytes}},allowed[id]))
                need(host_is_poisoned(a.data+lo) && host_is_poisoned(a.data+hi-1),"ASan shadow cannot represent exact declared gap");
            need(host_is_poisoned(a.data-1) && host_is_poisoned(a.data+a.bytes),"ASan allocation guards unavailable");
        }
        for(auto& other:ops) if(other.scratch.base) {
            host_poison(other.scratch.base,other.scratch.total);
            if(&other==&o)host_unpoison(other.scratch.data,other.scratch.bytes);
        }
    }
    void disarm_bounds() {
        if(!memory_check)return;
        for(auto& [id,a]:allocations)host_unpoison(a.base,a.total);
        for(auto& o:ops)if(o.scratch.base)host_unpoison(o.scratch.base,o.scratch.total);
    }
    void check_node_writes(Operation& o,const std::map<std::string,Bytes>& before) {
        auto allowed=ranges(o,'w');
        for(auto& [id,a]:allocations) {
            guard_check(a,id);
            Bytes after(a.bytes);copy(after.data(),a.data,a.bytes);
            for(auto [lo,hi]:subtract_ranges({{0,a.bytes}},allowed[id]))
                if(!std::equal(after.begin()+lo,after.begin()+hi,before.at(id).begin()+lo))
                    throw Failure("operation write contract violated: "+o.id+" changed read-only/undeclared bytes in "+id);
        }
        for(auto& op:ops)guard_check(op.scratch,op.id+" scratch");
    }
    Json node_output_hashes(Operation& o) {
        Json hashes=Json::array();
        RangeMap explicit_checks;
        // The checkpoint comparison has just run. Its declared tolerance may allow
        // legitimate floating-point variation; do not override it with a byte hash.
        if(checkpoints.count(o.id))for(const auto& c:checkpoints.at(o.id)) {
            const auto& v=views.at(c.desc.at("view"));size_t lo=v.at("offset");
            explicit_checks[v.at("allocation")].emplace_back(lo,lo+v.at("bytes").get<size_t>());
        }
        for(auto& [id,rr]:ranges(o,'w')) for(auto [lo,hi]:subtract_ranges(rr,explicit_checks[id])) {
            Bytes b(hi-lo);copy(b.data(),allocations.at(id).data+lo,b.size());
            hashes.push_back(Json{{"allocation",id},{"offset",lo},{"bytes",hi-lo},{"sha256",hash(b)}});
        }
        return hashes;
    }
    void diagnostic_pass(unsigned pattern) {
        if(trace_dispatches)rad_replay_trace::context("", "diagnostic-support", pattern);
        for(auto& [id,a]:allocations)reset(a);
        for(auto& o:ops) if(o.scratch.base) {
            auto policy=o.desc.value("scratch_policy",std::string("unspecified"));
            need(policy!="unspecified","scratch contract required for diagnostics: "+o.id);
            std::fill(o.scratch.pristine.begin(),o.scratch.pristine.end(),
                      policy=="zeroed" || pattern==0?0:pattern==1?0x3d:0xc2);
            reset(o.scratch);
        }
        for(auto& o:ops) {
            sync_all();
            if(pattern)poison_outputs(o,pattern);
            std::map<std::string,Bytes> before;
            for(auto& [id,a]:allocations) {before[id].resize(a.bytes);copy(before[id].data(),a.data,a.bytes);}
            progress<<Json{{"phase","diagnostic-node"},{"pattern",pattern},{"node",o.id}}.dump()<<'\n';progress.flush();
            try {
                arm_bounds(o);
                if(trace_dispatches)rad_replay_trace::context(o.id,"diagnostic-node",pattern);
                issue(o);sync_all();disarm_bounds();
                if(trace_dispatches)rad_replay_trace::context(o.id,"diagnostic-support",pattern);
            }
            catch(...) {disarm_bounds();throw;}
            check_node_writes(o,before);
            if(checkpoints.count(o.id))compare_list(checkpoints.at(o.id));
            auto observed=node_output_hashes(o);
            if(pattern==0)baseline_checkpoints[o.id]=observed;
            else if(observed!=baseline_checkpoints.at(o.id))
                throw Failure("diagnostic output/scratch invariance failed at "+o.id+
                              "; exact intermediate bytes differ (not a performance or race-freedom claim)");
        }
        compare_checks();
    }
    void diagnostics() {
        result["diagnostics"]={{"status","NOT_RUN"},{"read_checks","NOT_CHECKED"},
            {"per_operation_write_contracts",false},{"output_poison_variants",0},
            {"scratch_contracts",false},{"intermediate_equality","explicit checkpoint tolerances, then exact bytes for remaining writes against same-candidate baseline"},
            {"race_freedom_proved",false},{"allocation_byte_bounds_not_scalar_element_bounds",true}};
        if(plan.value("diagnostics",std::string("none"))!="contracts")return;
        result["diagnostics"]["status"]="RUNNING";
        for(unsigned pattern=0;pattern<3;++pattern) {
            diagnostic_pass(pattern);result["diagnostics"]["completed_passes"]=pattern+1;
        }
        result["diagnostics"]["status"]="PASS";
        result["diagnostics"]["read_checks"]=memory_check?"HOST_ASAN_INSTRUMENTED_PATHS":"NOT_CHECKED";
        result["diagnostics"]["per_operation_write_contracts"]=true;
        result["diagnostics"]["output_poison_variants"]=2;
        result["diagnostics"]["poisoned_output_bytes_total"]=diagnostic_output_bytes;
        result["diagnostics"]["scratch_contracts"]=true;
        result["diagnostics"]["checked_nodes"]=ops.size();
        result["diagnostics"]["explicit_checkpoints"]=checkpoints.size();
        result["diagnostics"]["output_poison_scope"]="packed exact-span output-only operands, excluding all readable aliases; other views not covered";
        // Timing uses original contract state, not the last diagnostic perturbation.
        for(auto& o:ops)if(o.scratch.base)std::fill(o.scratch.pristine.begin(),o.scratch.pristine.end(),0);
    }
    void save_dispatch_trace() {
        if(!trace_dispatches)return;
        rad_replay_trace::end();auto& s=rad_replay_trace::state();
        Json t={{"schema","radiance-dispatch-trace/1"},{"fixture_sha256",plan.at("fixture_sha256")},
            {"scope","ordered AQL dispatch entry, diagnostic passes only; not raw tape/SDMA/TP2"},
            {"completed",result.value("status",std::string("INCOMPLETE"))=="RUNTIME_CHECKS_PASS" && s.error.empty()},
            {"error",s.error},{"entries",Json::array()}};
        for(const auto& e:s.entries)t["entries"].push_back(Json{{"identity",e.identity},{"node",e.node},{"phase",e.phase},
            {"pattern",e.pattern},{"device",e.device},{"queue",e.queue},{"grid",e.grid},{"group",e.group},
            {"group_bytes",e.group_bytes},{"private_bytes",e.private_bytes}});
        write_json(out/"dispatch-trace.json",t);
        result["dispatch_trace"]={{"file","dispatch-trace.json"},{"sha256",hash_file(out/"dispatch-trace.json")},
            {"entry_count",s.entries.size()},{"completed",t.at("completed")}};
    }
    void run() {
        diagnostics();
        if(trace_dispatches) {
            auto& s=rad_replay_trace::state();
            need(s.error.empty() && !s.entries.empty(),"dispatch trace absent or refused; build/apply the optional AQL trace patch");
            for(const auto& op:ops)if(op.kind==Operation::Kernel)for(int pass=0;pass<3;++pass) {
                bool found=false;for(const auto& e:s.entries)found|=e.phase=="diagnostic-node" && e.node==op.id && e.pattern==pass;
                need(found,"unobserved GPU kernel node: "+op.id);
            }
        }
        if(plan.value("diagnostic_only",false)) {
            need(result.at("diagnostics").at("status")=="PASS","diagnostic-only run has no completed diagnostics");
            result["measurement"]="NOT_MEASURED_DIAGNOSTIC_ONLY";
            result["statistics"]=nullptr;result["reference"]=m.at("provenance").at("reference");
            result["status"]="RUNTIME_CHECKS_PASS";return;
        }
        need(!RAD_REPLAY_HAS_ASAN,"instrumented worker timings are not performance evidence; use --diagnostic-only");
        int warm=plan.at("warmups"), reps=plan.at("repetitions");
        need(0<=warm && warm<=1000 && 1<=reps && reps<=10000,"bad repetition count");
        bool gated=plan.at("gate");need(!gated || (gpu && streams.size()==1),"gated mode requires one AQL stream");
        result["measurement"]=gpu?(gated?"single-stream gated queue-window GPU event time":"multi/single-stream host-paced queue-window GPU event time"):
                                         "host ordered launcher-window steady_clock time";
        result["restore_policy"]="restore every allocation and zero scratch before every checked iteration";
        result["dispatch_plan"]="typed pre-resolved launcher plan; no JSON/disk I/O in measured submit loop";
        result["state_policy"]="no hidden state; warmups and timed iterations use identical pristine bytes";
        result["reference"]=m.at("provenance").at("reference");
        result["physical_bytes_with_guards"]=used;
        for(int iteration=0;iteration<warm+reps;++iteration) {
            for(auto& [id,a]:allocations) reset(a);
            for(auto& o:ops) reset(o.scratch);
            progress<<Json{{"iteration",iteration},{"phase","prefix"}}.dump()<<'\n';progress.flush();
            for(auto& o:ops) if(!o.measured) issue(o);
            void* gate=nullptr;
            double us=0;
            progress<<Json{{"iteration",iteration},{"phase","measure-submit"}}.dump()<<'\n';progress.flush();
            try {
                if(gpu && gated) {
                    auto q=streams.begin()->second;
                    // Prefix executes before the gate. There is no host wait between prefix
                    // and measured work; gate opening is outside the device timestamp interval.
                    ok(rad_stream_gate(q,&gate),"create gate");
                    need(rad_gate_room(q)>4,"insufficient gated queue room");
                    ok(rad_event_record(start,q),"record start");
                    for(auto& o:ops) if(o.measured) issue(o);
                    ok(rad_event_record(stop,q),"record stop");
                    void* g=gate;gate=nullptr;ok(rad_gate_open(g),"open gate");ok(rad_event_sync(stop),"window completion");
                    float ms=0;ok(rad_event_elapsed_ms(start,stop,&ms),"window timestamps");us=double(ms)*1000;
                } else if(gpu) {
                    for(auto& [id,q]:streams) { ok(rad_event_record(ready[id],q),"prefix completion");ok(rad_event_wait(control,ready[id]),"prefix join"); }
                    ok(rad_event_record(start,control),"window start");
                    for(auto& [id,q]:streams) ok(rad_event_wait(q,start),"common start");
                    for(auto& o:ops) if(o.measured) issue(o);
                    for(auto& [id,q]:streams) { ok(rad_event_record(done[id],q),"queue completion");ok(rad_event_wait(control,done[id]),"window join"); }
                    ok(rad_event_record(stop,control),"window stop");ok(rad_event_sync(stop),"window completion");
                    float ms=0;ok(rad_event_elapsed_ms(start,stop,&ms),"window timestamps");us=double(ms)*1000;
                } else {
                    auto begin=std::chrono::steady_clock::now();
                    for(auto& o:ops) if(o.measured) issue(o);
                    us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
                }
            } catch(...) {
                if(gate) (void)rad_gate_open(gate);
                throw;
            }
            need(std::isfinite(us) && us>=0,"invalid device timestamps");
            compare_checks(); // never time a dataset that is not also checked
            result["checked_iterations"]=iteration+1;
            if(iteration>=warm) result["samples_us"].push_back(us);
            progress<<Json{{"iteration",iteration},{"phase","checked"},{"window_us",us}}.dump()<<'\n';progress.flush();
        }
        auto samples=result.at("samples_us").get<std::vector<double>>();std::sort(samples.begin(),samples.end());
        auto q=[&](double f){size_t i=(size_t)std::ceil(f*samples.size());return samples[std::min(samples.size()-1,i?i-1:0)];};
        result["statistics"]={{"unit","us per complete window"},{"n",samples.size()},{"min",samples.front()},{"median",q(.5)},{"p90",q(.9)},
                              {"p90_is_per_dispatch",false},{"tokens_per_second_estimated",false}};
        result["status"]="RUNTIME_CHECKS_PASS";
    }
};
}
int main(int argc,char** argv) {
    if(argc!=2) {std::cerr<<"Use replay.sh run; worker expects one validated plan file\n";return 2;}
    auto* w=new Worker;int code=2;
    try {w->plan=read_json(argv[1]);w->out=w->plan.at("out").get<std::string>();w->load();w->run();code=0;}
    catch(const Failure& e){w->result["status"]="FAIL";w->result["error"]=e.what();code=1;}
    catch(const std::exception& e){w->result["status"]="INCOMPLETE";w->result["error"]=e.what();}
    try {w->save_dispatch_trace();write_json(w->out/"worker-result.json",w->result);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';code=2;}
    std::cout<<w->result.dump(2)<<'\n';std::cout.flush();std::cerr.flush();
    // Do not invoke backend/plugin destructors after a failed queue; process isolation is deliberate.
    std::_Exit(code);
}
