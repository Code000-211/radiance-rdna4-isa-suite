#pragma once
// Optional CPU-side diagnostic hook. Versioned ABI; never dereference GPU pointers.
// Backends call BEFORE queue reservation. Nonzero observer result refuses dispatch.
#include <stdint.h>
struct RadReplayDispatchV1 {
    uint32_t version;
    const char* object_sha256;
    const char* descriptor_symbol;
    int32_t device;
    uint32_t queue_id;
    uint32_t grid[3];
    uint32_t group[3];
    uint32_t group_bytes;
    uint32_t private_bytes;
};
#ifndef RAD_REPLAY_DISPATCH_COLLECTOR
# define RAD_REPLAY_DISPATCH_WEAK __attribute__((weak))
#else
# define RAD_REPLAY_DISPATCH_WEAK
#endif
extern "C" int rad_replay_trace_enabled_v1() noexcept RAD_REPLAY_DISPATCH_WEAK;
extern "C" int rad_replay_observe_dispatch_v1(const RadReplayDispatchV1*) noexcept RAD_REPLAY_DISPATCH_WEAK;
#undef RAD_REPLAY_DISPATCH_WEAK
