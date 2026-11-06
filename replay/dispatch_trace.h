#pragma once
#define RAD_REPLAY_DISPATCH_COLLECTOR
#include "../integration/rad_replay_dispatch.h"
#undef RAD_REPLAY_DISPATCH_COLLECTOR
#include <array>
#include <atomic>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
namespace rad_replay_trace {
struct Entry {
    std::string identity,node,phase;
    int pattern=-1,device=-1;
    unsigned queue=0,group_bytes=0,private_bytes=0;
    std::array<unsigned,3> grid{},group{};
};
struct State {
    std::atomic<bool> enabled{false};
    std::mutex mutex;
    std::set<std::string> allowed;
    std::vector<Entry> entries;
    std::string node,phase,error;
    int pattern=-1;
    size_t cap=1000000;
    std::thread::id owner;
};
inline State& state(){static State s;return s;}
inline void begin(const std::set<std::string>& allowed,size_t cap=1000000) {
    auto& s=state();std::lock_guard<std::mutex> l(s.mutex);
    if(allowed.empty() || !cap)throw std::runtime_error("empty dispatch whitelist or capacity");
    s.allowed=allowed;s.cap=cap;s.entries.clear();s.node="";s.phase="setup";s.pattern=-1;s.error="";s.owner=std::this_thread::get_id();s.enabled=true;
}
inline void context(std::string node,std::string phase,int pattern) {
    auto& s=state();if(!s.enabled)return;std::lock_guard<std::mutex> l(s.mutex);
    s.node=std::move(node);s.phase=std::move(phase);s.pattern=pattern;
}
inline void end(){state().enabled=false;}
inline int observe(const RadReplayDispatchV1* e) noexcept {
    auto& s=state();if(!s.enabled)return 0;
    try {
        std::lock_guard<std::mutex> l(s.mutex);
        if(!s.error.empty())return 1;
        auto fail=[&](const char* why){s.error=why;return 1;};
        if(s.owner!=std::this_thread::get_id())return fail("unscoped background-thread dispatch refused");
        if(!e || e->version!=1 || !e->object_sha256 || !e->descriptor_symbol)return fail("invalid dispatch-hook ABI");
        std::string h=e->object_sha256,sym=e->descriptor_symbol;
        if(h.size()!=64 || h.find_first_not_of("0123456789abcdef")!=std::string::npos || sym.empty() || sym.size()>65536)return fail("invalid dispatch identity");
        const std::string identity=h+":"+sym;
        if(!s.allowed.count(identity)){s.error="unapproved dispatch refused before queue reservation: "+identity;return 1;}
        if(s.entries.size()>=s.cap)return fail("dispatch trace capacity reached; no silent truncation");
        Entry v;v.identity=identity;v.node=s.node;v.phase=s.phase;v.pattern=s.pattern;
        v.device=e->device;v.queue=e->queue_id;v.group_bytes=e->group_bytes;v.private_bytes=e->private_bytes;
        if(v.device<0)return fail("invalid dispatch device");
        uint64_t group_size=1;
        for(unsigned i=0;i<3;++i){if(!e->grid[i] || !e->group[i] || e->group[i]>65535)return fail("invalid dispatch dimensions");v.grid[i]=e->grid[i];v.group[i]=e->group[i];group_size*=e->group[i];}
        if(group_size>1024 || v.group_bytes>65536)return fail("invalid workgroup resources");
        s.entries.push_back(std::move(v));return 0;
    } catch(...) {
        // Allocation failure itself refuses dispatch. No C++ exception crosses the C hook.
        return 1;
    }
}
}
extern "C" int rad_replay_trace_enabled_v1() noexcept {return rad_replay_trace::state().enabled?1:0;}
extern "C" int rad_replay_observe_dispatch_v1(const RadReplayDispatchV1* e) noexcept {return rad_replay_trace::observe(e);}
