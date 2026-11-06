// Enumerates declared plugin rows without starting an inference engine.
#include "common.h"
#include "plugin/registry.h"
#include "device/device.h"
#include <dlfcn.h>
#include <iostream>
using namespace rad_replay;
int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"usage: rad-replay-inventory TRUSTED_PLUGIN NEW_OUTPUT.json\n";return 2;}
    try {
        auto* reg=new rad::Registry;
        const std::string path=fs::canonical(argv[1]).string();
        const auto before=hash_file(path);
        need(reg->load_plugin(path,0)==RAD_OK,"plugin load failed");
        need(reg->plugins().size()==1,"expected one plugin");
        const auto& p=reg->plugins().front();
        using Count=int(*)();using At=const RadKernelInfo*(*)(int);
        auto count=reinterpret_cast<Count>(dlsym(p.handle,"rad_kernel_count"));
        auto at=reinterpret_cast<At>(dlsym(p.handle,"rad_kernel_at"));
        need(count && at,"plugin has no kernel table");
        int n=count();need(n>0 && n<=100000,"invalid declared kernel count");
        const auto h=hash_file(path);need(h==before,"plugin changed during load");
        Json j={{"schema","radiance-declared-rows/1"},{"plugin",p.name},{"plugin_sha256",h},
                {"backend",rad::device_backend_name()},{"rows",Json::array()},
                {"runtime_executed",false},{"valid_tuning_combinations_enumerated",false}};
        for(int i=0;i<n;++i) {
            const auto* r=at(i);need(r && r->name && r->op,"invalid declared row");
            Json row={{"index",i},{"identity",h+":"+r->name},{"name",r->name},{"op",r->op},
                {"domain",r->domain==RAD_DOMAIN_HOST?"host":"device"},
                {"shape",r->shape?r->shape:""},{"dtypes",r->dtypes?r->dtypes:""},
                {"has_launch",bool(r->launch)},{"has_instance",bool(r->init || r->fini)},
                {"has_shape_adapter",bool(r->opd_shape)},{"has_scratch",bool(r->scratch)},
                {"has_cross_axis_predicate",bool(r->tune_valid)},
                {"constraints",Json::array()},{"tunables",Json::array()}};
            need(r->n_constraints>=0 && r->n_constraints<=4096 && r->n_tunables>=0 && r->n_tunables<=128,"invalid table size");
            need((!r->n_constraints || r->constraints) && (!r->n_tunables || r->tunables),"missing metadata tables");
            for(int q=0;q<r->n_constraints;++q) {
                const auto& c=r->constraints[q];need(c.key,"null constraint key");
                row["constraints"].push_back(Json{{"key",c.key},{"op",c.op},{"integer",c.ival},{"strings",c.sval?c.sval:""}});
            }
            for(int q=0;q<r->n_tunables;++q) {
                const auto& t=r->tunables[q];need(t.key && t.values && t.n_values>0 && t.n_values<=100000,"invalid tunable axis");
                Json vals=Json::array();for(int v=0;v<t.n_values;++v)vals.push_back(t.values[v]);
                row["tunables"].push_back(Json{{"key",t.key},{"values",vals},{"default",t.deflt}});
            }
            j["rows"].push_back(std::move(row));
        }
        need(hash_file(path)==h,"plugin changed during enumeration");
        j["declared_row_count"]=n;write_json(argv[2],j);std::cout<<n<<" declared rows; none executed\n";return 0;
    } catch(const std::exception& e){std::cerr<<"INCOMPLETE: "<<e.what()<<'\n';return 2;}
}
