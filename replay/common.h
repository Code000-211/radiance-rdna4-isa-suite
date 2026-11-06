#pragma once
// Host-only replay I/O. Uses the already tested v0.2 SHA-256 implementation.
#include "../integration/rad_isa_capture.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
namespace rad_replay {
using Json = nlohmann::json;
using Bytes = std::vector<uint8_t>;
namespace fs = std::filesystem;
struct Incomplete : std::runtime_error { using std::runtime_error::runtime_error; };
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };
inline void need(bool ok, const std::string& what) { if (!ok) throw Incomplete(what); }
inline Bytes read_bytes(const fs::path& p, uint64_t max = 2ull<<30) {
    const auto n=fs::file_size(p); need(n<=max,"file exceeds read budget: "+p.string());
    std::ifstream f(p,std::ios::binary); need(bool(f),"cannot open "+p.string());
    Bytes b(static_cast<size_t>(n));
    if(n) f.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(n));
    need(bool(f),"short read: "+p.string()); return b;
}
inline std::string hash(const Bytes& b) { return rad_isa_capture::sha256(b.data(),b.size()); }
inline std::string hash_file(const fs::path& p) { return hash(read_bytes(p)); }
inline Json read_json(const fs::path& p) {
    auto b=read_bytes(p,16ull<<20); return Json::parse(b.begin(),b.end());
}
inline void write_bytes(const fs::path& p, const void* data, size_t n) {
    int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    need(fd>=0,"cannot exclusively create "+p.string());
    bool ok=rad_isa_capture::write_all(fd,data,n);
    if(::fsync(fd)) ok=false;
    if(::close(fd)) ok=false;
    need(ok,"short write: "+p.string());
}
inline void write_json(const fs::path& p, const Json& j) {
    auto s=j.dump(2)+"\n"; write_bytes(p,s.data(),s.size());
}
inline Json save_blob(const fs::path& root, const Bytes& b) {
    need(!b.empty(),"empty snapshot blob");
    auto h=hash(b); auto dir=root/"blobs"; fs::create_directory(dir);
    fs::permissions(dir,fs::perms::owner_all,fs::perm_options::replace);
    auto p=dir/h;
    if(!fs::exists(p)) write_bytes(p,b.data(),b.size());
    else need(fs::file_size(p)==b.size() && hash_file(p)==h,"existing blob changed");
    return Json{{"path","blobs/"+h},{"sha256",h},{"bytes",b.size()}};
}
}
