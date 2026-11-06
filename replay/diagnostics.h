#pragma once
// Host-only instrumentation and interval arithmetic. No automatic GPU-ASan claim.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#ifndef __has_feature
#define __has_feature(x) 0
#endif
#if defined(__SANITIZE_ADDRESS__) || __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define RAD_REPLAY_HAS_ASAN 1
#else
#define RAD_REPLAY_HAS_ASAN 0
#endif
namespace rad_replay {
using Ranges=std::vector<std::pair<size_t,size_t>>;
inline Ranges merge_ranges(Ranges r) {
    std::sort(r.begin(),r.end());Ranges out;
    for(auto [a,b]:r) {
        if(a>=b) continue;
        if(out.empty() || a>out.back().second) out.emplace_back(a,b);
        else out.back().second=std::max(out.back().second,b);
    }
    return out;
}
inline Ranges subtract_ranges(Ranges a,Ranges b) {
    a=merge_ranges(std::move(a));b=merge_ranges(std::move(b));Ranges out;
    for(auto [lo,hi]:a) {
        for(auto [x,y]:b) {
            if(y<=lo || x>=hi) continue;
            if(x>lo)out.emplace_back(lo,x);
            lo=std::max(lo,y);if(lo>=hi)break;
        }
        if(lo<hi)out.emplace_back(lo,hi);
    }
    return out;
}
inline void host_poison(void* p,size_t n) {
#if RAD_REPLAY_HAS_ASAN
    __asan_poison_memory_region(p,n);
#else
    (void)p;(void)n;
#endif
}
inline void host_unpoison(void* p,size_t n) {
#if RAD_REPLAY_HAS_ASAN
    __asan_unpoison_memory_region(p,n);
#else
    (void)p;(void)n;
#endif
}
inline bool host_is_poisoned(const void* p) {
#if RAD_REPLAY_HAS_ASAN
    return __asan_address_is_poisoned(p)!=0;
#else
    (void)p;return false;
#endif
}
}
