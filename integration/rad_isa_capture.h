#pragma once
// Optional, Linux-only, CPU-side evidence capture. No GPU commands are issued.
// Include before HIP headers. Disabled unless RAD_ISA_CAPTURE_DIR is nonempty.
// A capture failure must propagate to the caller (see the accompanying patch).
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rad_isa_capture {
// SHA-256 binds each resolution record to the bytes observed by the loader.
// Cross-checked against Python hashlib in the host integration tests.
inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32-n)); }
inline std::string sha256(const void* input, size_t size) {
    static constexpr uint32_t K[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const auto* data=static_cast<const unsigned char*>(input);
    const size_t blocks=size/64+((size%64)<56?1:2);
    for(size_t block=0;block<blocks;++block) {
        unsigned char b[64]{};
        for(size_t j=0;j<64;++j) {
            const size_t pos=block*64+j;
            if(pos<size) b[j]=data[pos];
            else if(pos==size) b[j]=0x80;
            else if(block==blocks-1 && j>=56) b[j]=static_cast<unsigned char>((uint64_t(size)*8)>>(8*(63-j)));
        }
        uint32_t w[64]{};
        for(int j=0;j<16;++j) w[j]=(uint32_t(b[j*4])<<24)|(uint32_t(b[j*4+1])<<16)|(uint32_t(b[j*4+2])<<8)|b[j*4+3];
        for(int j=16;j<64;++j) {
            uint32_t s0=rotr(w[j-15],7)^rotr(w[j-15],18)^(w[j-15]>>3);
            uint32_t s1=rotr(w[j-2],17)^rotr(w[j-2],19)^(w[j-2]>>10);
            w[j]=w[j-16]+s0+w[j-7]+s1;
        }
        uint32_t a=h[0],c1=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for(int j=0;j<64;++j) {
            uint32_t t1=z+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^((~e)&g))+K[j]+w[j];
            uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&c1)^(a&c)^(c1&c));
            z=g;g=f;f=e;e=d+t1;d=c;c=c1;c1=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=c1;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
    }
    std::string out;
    for(auto x:h) { char b[9];std::snprintf(b,sizeof b,"%08x",x);out+=b; }
    return out;
}
struct Image { std::string file; std::string hash; };
inline std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += c; }
        else if (c < 32) { char b[7]; std::snprintf(b,sizeof b,"\\u%04x",c); out += b; }
        else out += c;
    }
    return out + '"';
}
inline bool write_all(int fd, const void* data, size_t size) {
    const char* p = static_cast<const char*>(data);
    while (size) {
        const ssize_t n = ::write(fd,p,size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n; size -= static_cast<size_t>(n);
    }
    return true;
}
struct State {
    const pid_t owner = ::getpid();
    std::mutex mutex;
    int directory = -1;
    int log = -1;
    pid_t pid = 0;
    unsigned long long serial = 0;
    bool initialized = false;
    bool enabled = false;
    bool failed = false;
    std::map<std::tuple<uintptr_t,size_t,int>,Image> images;
    ~State() { if (log >= 0) ::close(log); if (directory >= 0) ::close(directory); }
};
// Process-lifetime state avoids callbacks into a destroyed static during shutdown.
// The OS closes the descriptors; each record/snapshot is fsynced before success.
inline State& state() { static State* s = new State; return *s; }
inline bool failure(State& s, const char* message) {
    s.failed = true;
    std::fprintf(stderr,"RAD_ISA_CAPTURE_ERROR: %s (errno=%d)\n",message,errno);
    return false;
}
inline bool initialize(State& s) {
    if (s.initialized) {
        if (s.enabled && s.pid != ::getpid()) return failure(s,"fork after capture initialization is unsupported");
        return !s.failed;
    }
    s.initialized = true;
    const char* path = std::getenv("RAD_ISA_CAPTURE_DIR");
    if (!path || !*path) return true;
    s.enabled = true; s.pid = ::getpid();
    s.directory = ::open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    if (s.directory < 0) return failure(s,"capture directory must exist");
    struct stat st{};
    if (::fstat(s.directory,&st) || !S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() || (st.st_mode & 0022))
        return failure(s,"capture directory must be owned by the process user and not group/world writable");
    const std::string name="resolved-"+std::to_string(s.pid)+".jsonl";
    s.log=::openat(s.directory,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if (s.log < 0) return failure(s,"cannot exclusively create capture log; use a fresh directory");
    return true;
}
// Called after successful executable loading, including reloads. Replaces the
// pointer-key mapping, avoiding stale-pointer reuse across module unload/reload.
inline bool object(const void* data, size_t size, int ordinal) {
    const char* configured=std::getenv("RAD_ISA_CAPTURE_DIR");
    if (!configured || !*configured) return true;
    State& s=state();
    // Check before taking the inherited mutex: fork may inherit a locked mutex.
    if (s.owner != ::getpid()) {
        std::fprintf(stderr,"RAD_ISA_CAPTURE_ERROR: fork after capture initialization is unsupported\n");
        return false;
    }
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!initialize(s)) return false;
    if (!s.enabled) return true;
    if (!data || !size) return failure(s,"empty code object");
    const std::string name="code-"+std::to_string(s.pid)+"-"+std::to_string(++s.serial)+".hsaco";
    const std::string hash=sha256(data,size);
    int fd=::openat(s.directory,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if (fd < 0) return failure(s,"cannot exclusively create code object snapshot");
    bool ok=write_all(fd,data,size);
    if (ok && ::fsync(fd)) ok=false;
    if (::close(fd)) ok=false;
    if (!ok) { ::unlinkat(s.directory,name.c_str(),0); return failure(s,"incomplete code object snapshot"); }
    s.images[std::make_tuple(reinterpret_cast<uintptr_t>(data),size,ordinal)]=Image{name,hash};
    return true;
}
// Records first resolution per Func/ordinal, NOT every kernel dispatch.
inline bool resolved(const void* data, size_t size, int ordinal, const std::string& name) {
    const char* configured=std::getenv("RAD_ISA_CAPTURE_DIR");
    if (!configured || !*configured) return true;
    State& s=state();
    // Check before taking the inherited mutex: fork may inherit a locked mutex.
    if (s.owner != ::getpid()) {
        std::fprintf(stderr,"RAD_ISA_CAPTURE_ERROR: fork after capture initialization is unsupported\n");
        return false;
    }
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!initialize(s)) return false;
    if (!s.enabled) return true;
    const auto it=s.images.find(std::make_tuple(reinterpret_cast<uintptr_t>(data),size,ordinal));
    if (it==s.images.end()) return failure(s,"resolved kernel has no captured object");
    const std::string line="{\"event\":\"resolve\",\"pid\":"+std::to_string(s.pid)+
        ",\"device_ordinal\":"+std::to_string(ordinal)+",\"bytes\":"+std::to_string(size)+
        ",\"file\":"+quote(it->second.file)+",\"sha256\":"+quote(it->second.hash)+",\"kernel\":"+quote(name)+"}\n";
    if (!write_all(s.log,line.data(),line.size()) || ::fsync(s.log)) return failure(s,"cannot persist resolution record");
    return true;
}
} // namespace rad_isa_capture
